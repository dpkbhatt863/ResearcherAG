import streamlit as st
import tempfile
import os
from src.loader import process_all_pdfs, split_documents
from src.vectorstore import build_vectorstore, get_retriever
from src.rag import build_rag_chain, ask

st.set_page_config(page_title="Research Paper RAG", layout="wide")
st.title("Research Paper Assistant")
st.caption("Upload research papers and ask questions across all of them.")

# --- Sidebar: PDF Upload ---
with st.sidebar:
    st.header("Upload Papers")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process Papers", type="primary"):
            with st.spinner("Loading and indexing papers..."):
                with tempfile.TemporaryDirectory() as tmpdir:
                    for file in uploaded_files:
                        path = os.path.join(tmpdir, file.name)
                        with open(path, "wb") as f:
                            f.write(file.read())

                    docs = process_all_pdfs(tmpdir)
                    chunks = split_documents(docs)
                    vectorstore = build_vectorstore(chunks)
                    retriever = get_retriever(vectorstore)
                    chain = build_rag_chain(retriever)

                    st.session_state.chain = chain
                    st.session_state.paper_names = [f.name for f in uploaded_files]

            st.success(f"Indexed {len(uploaded_files)} paper(s) — {len(chunks)} chunks")

    if "paper_names" in st.session_state:
        st.divider()
        st.subheader("Loaded Papers")
        for name in st.session_state.paper_names:
            st.markdown(f"- {name}")

# --- Main: Q&A ---
if "chain" not in st.session_state:
    st.info("Upload and process your PDFs using the sidebar to get started.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                st.caption("Sources: " + ", ".join(msg["sources"]))

    question = st.chat_input("Ask a question about your papers...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask(st.session_state.chain, question)
            st.markdown(result["answer"])
            st.caption("Sources: " + ", ".join(result["sources"]))

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })
