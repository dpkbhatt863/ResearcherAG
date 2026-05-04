import streamlit as st
import tempfile
import os
from collections import defaultdict
from src.loader import process_all_pdfs, split_documents
from src.vectorstore import build_vectorstore, get_retriever
from src.rag import build_rag_chain, ask, summarize_paper

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
            with tempfile.TemporaryDirectory() as tmpdir:
                for file in uploaded_files:
                    path = os.path.join(tmpdir, file.name)
                    with open(path, "wb") as f:
                        f.write(file.read())

                with st.spinner("Loading and chunking PDFs..."):
                    docs = process_all_pdfs(tmpdir)
                    chunks = split_documents(docs)

                with st.spinner("Building vector index..."):
                    vectorstore = build_vectorstore(chunks)
                    retriever = get_retriever(vectorstore)
                    chain = build_rag_chain(retriever)

                # Group chunks by paper for summarization
                chunks_by_paper = defaultdict(list)
                for chunk in chunks:
                    chunks_by_paper[chunk.metadata.get("file_name", "unknown")].append(chunk)

                summaries = {}
                for paper_name, paper_chunks in chunks_by_paper.items():
                    with st.spinner(f"Summarizing {paper_name}..."):
                        summaries[paper_name] = summarize_paper(paper_chunks, paper_name)

                st.session_state.chain = chain
                st.session_state.paper_names = [f.name for f in uploaded_files]
                st.session_state.summaries = summaries
                st.session_state.messages = []

            st.success(f"Indexed {len(uploaded_files)} paper(s) — {len(chunks)} chunks")

    # Show paper summaries in sidebar
    if "summaries" in st.session_state:
        st.divider()
        st.subheader("Paper Summaries")
        for paper_name, summary in st.session_state.summaries.items():
            with st.expander(paper_name):
                st.markdown(summary)

# --- Main: Q&A ---
if "chain" not in st.session_state:
    st.info("Upload and process your PDFs using the sidebar to get started.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['file']}** — Page {s['page']}")
                        st.caption(f"> {s['snippet']}...")

    question = st.chat_input("Ask a question about your papers...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask(
                    st.session_state.chain,
                    question,
                    chat_history=st.session_state.messages
                )
            st.markdown(result["answer"])
            with st.expander("Sources"):
                for s in result["sources"]:
                    st.markdown(f"**{s['file']}** — Page {s['page']}")
                    st.caption(f"> {s['snippet']}...")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })
