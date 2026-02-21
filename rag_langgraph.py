from __future__ import annotations
import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

load_dotenv()

# ------------------- 1. LLM + Embeddings -------------------
api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(model="openai/gpt-oss-20b", api_key=api_key)
embeddings = HuggingFaceEmbeddings()

# ------------------- 2. PDF Retriever Store -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: str):
    if thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        _THREAD_RETRIEVERS[thread_id] = retriever
        _THREAD_METADATA[thread_id] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        return _THREAD_METADATA[thread_id]
    finally:
        try: os.remove(temp_path)
        except OSError: pass

# ------------------- 3. Tools -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform basic arithmetic: add, sub, mul, div
    """
    try:
        if operation == "add": result = first_num + second_num
        elif operation == "sub": result = first_num - second_num
        elif operation == "mul": result = first_num * second_num
        elif operation == "div":
            if second_num == 0: return {"error": "Division by zero not allowed"}
            result = first_num / second_num
        else: return {"error": f"Unsupported operation '{operation}'"}
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Get latest stock price for a symbol (e.g. AAPL, TSLA)
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

@tool
def rag_tool(query: str, thread_id: str) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Must include thread_id.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {"error": "No document indexed for this chat. Upload a PDF first.", "query": query}
    result = retriever.invoke(query)
    return {
        "query": query,
        "context": [doc.page_content for doc in result],
        "metadata": [doc.metadata for doc in result],
        "source_file": _THREAD_METADATA.get(thread_id, {}).get("filename"),
    }

tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# ------------------- 4. Chat State -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ------------------- 5. Chat Node -------------------
def chat_node(state: ChatState, config=None):
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")
    system_message = SystemMessage(
        content=(
            "You are a helpful assistant.\n"
            "If the user asks about the uploaded PDF, you MUST call `rag_tool`.\n"
            f"When calling `rag_tool`, you MUST pass query and thread_id='{thread_id}'\n"
            "Do NOT omit arguments.\n"
            "You can also use web search, stock price, and calculator tools if helpful.\n"
            "If no PDF is uploaded, ask the user to upload one."
        )
    )
    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# ------------------- 6. Checkpointer -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# ------------------- 7. Graph -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
chatbot = graph.compile(checkpointer=checkpointer)

# ------------------- 8. Helpers -------------------
def retrieve_all_threads():
    return list({c.config["configurable"]["thread_id"] for c in checkpointer.list(None)})

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(thread_id, {})

def thread_has_document(thread_id: str) -> bool:
    return thread_id in _THREAD_RETRIEVERS