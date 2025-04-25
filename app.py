import streamlit as st
import os
import dotenv
import uuid

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

from rag_methods import (
    load_doc_to_db, 
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
)

dotenv.load_dotenv()

MODEL = "gpt-4o"

st.set_page_config(
    page_title="Last Minute AI",
    page_icon="📚",
    initial_sidebar_state="expanded"
)

st.html("""<h1 style="text-align: center;"> Last Minute AI </h1>""")

# Variables of Page
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
]
    
# Side Bar
with st.sidebar:
    with st.expander("OpenAI API Key"):
        openai_api_key = st.text_input("Your OpenAI API Key",
            label_visibility="hidden",
            value=os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") else "",
            type="password",
            key="openai_api_key",
            )

missing_openai = openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key
if missing_openai:
    st.warning("⬅️ Please insert an Open AI API Key to continue. You can get one for free at (https://console.anthropic.com/)")
    
else:
    with st.sidebar:
        cols0 = st.columns(2)
        with cols0[0]:
            is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)
            st.toggle(
                "Use RAG", 
                value=is_vector_db_loaded, 
                key="use_rag", 
                disabled=not is_vector_db_loaded,
            )

        with cols0[1]:
            st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")


        st.header("RAG Sources:")
        st.file_uploader(
            "📄 Upload a document", 
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
            on_change=load_doc_to_db,
            key="rag_docs",
        )

        st.text_input(
            "🌐 Introduce a URL", 
            placeholder="https://example.com",
            on_change=load_url_to_db,
            key="rag_url",
        )

        with st.expander(f"📚 Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"):
            st.write([] if not is_vector_db_loaded else [source for source in st.session_state.rag_sources])

    # Main Chat
    llm_stream = ChatOpenAI(
            api_key=openai_api_key,
            model_name=MODEL,
            temperature=0.3,
            streaming=True,
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]

            if not st.session_state.use_rag:
                st.write_stream(stream_llm_response(llm_stream, messages))
            else:
                st.write_stream(stream_llm_rag_response(llm_stream, messages))