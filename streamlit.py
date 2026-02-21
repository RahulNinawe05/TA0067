import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rag_langgraph import chatbot, ingest_pdf, retrieve_all_threads, thread_document_metadata

# ================= Session Utilities =================
def generate_thread_id(): return str(uuid.uuid4())
def add_thread(tid): 
    if tid not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(tid)
def reset_chat():
    tid = generate_thread_id()
    st.session_state["thread_id"] = tid
    add_thread(tid)
    st.session_state["message_history"] = []

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

# ================= Session Init =================
if "message_history" not in st.session_state: st.session_state["message_history"] = []
if "thread_id" not in st.session_state: st.session_state["thread_id"] = generate_thread_id()
if "chat_threads" not in st.session_state: st.session_state["chat_threads"] = retrieve_all_threads()
if "ingested_docs" not in st.session_state: st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])
thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ================= Sidebar =================
st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")
if st.sidebar.button("New Chat"): reset_chat(); st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(f"Using `{latest_doc.get('filename')}` ({latest_doc.get('chunks')} chunks)")

uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name not in thread_docs:
        with st.sidebar.status("Indexing PDF…") as status:
            summary = ingest_pdf(uploaded_pdf.getvalue(), thread_key, uploaded_pdf.name)
            thread_docs[uploaded_pdf.name] = summary
            status.update(label="✅ PDF indexed", state="complete")

# ================= Main Chat =================
st.title("Multi-Utility Chatbot")

# Chat display
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]): st.text(msg["content"])

user_input = st.chat_input("Ask your question or use tools")

if user_input:
    st.session_state["message_history"].append({"role":"user","content":user_input})
    with st.chat_message("user"): st.text(user_input)

    CONFIG = {"configurable":{"thread_id":thread_key},"run_name":"chat_turn"}

    with st.chat_message("assistant"):
        status_holder = {"box": None}
        def ai_only_stream():
            for chunk, _ in chatbot.stream(
                {"messages":[HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(chunk, ToolMessage):
                    tool_name = getattr(chunk, "name","tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(f"🔧 Using `{tool_name}` …")
                if isinstance(chunk, AIMessage) and chunk.content:
                    yield chunk.content

        try:
            ai_message = st.write_stream(ai_only_stream())
        except Exception:
            # fallback
            response = chatbot.invoke({"messages":[HumanMessage(content=user_input)]}, config=CONFIG)
            ai_message = response["messages"][-1].content
            st.write(ai_message)

        if status_holder["box"]: status_holder["box"].update(label="✅ Tool finished", state="complete")

    st.session_state["message_history"].append({"role":"assistant","content":ai_message})

    # document metadata
    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(f"Document indexed: {doc_meta.get('filename')} (chunks:{doc_meta.get('chunks')})")