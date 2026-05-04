# Research Paper RAG Assistant

An AI-powered research assistant that lets you upload multiple research papers (PDFs) and ask questions across all of them. Built with LangChain, Groq (LLaMA 3.3 70B), ChromaDB, and Streamlit.

---

## Screenshot

<img width="1805" height="830" alt="image" src="https://github.com/user-attachments/assets/e3fdd8ea-14f1-4144-8913-982e521d8931" />

---

## How It Works

```
Upload PDFs → Chunk text → Embed with HuggingFace → Store in ChromaDB
                                                              ↓
                                          Ask a question → Multi-Query Retrieval
                                                              ↓
                                                    Groq LLaMA answers with sources
```

## Features

- Upload one or multiple research papers at once
- Ask natural language questions across all uploaded papers
- **Multi-query retrieval** — LLM rewrites your question into 3 alternative phrasings before retrieval, so semantically similar questions all find relevant content
- Answers include which paper(s) the information came from, with page numbers and text snippets
- **Conversation memory** — chat history is passed as context so follow-up questions work naturally
- **Paper summaries on upload** — each paper is summarized into 3 bullet points shown in the sidebar
- Completely free to use — no paid APIs

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Groq (LLaMA 3.3 70B) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (in-memory) |
| PDF Loader | PyMuPDF |
| Framework | LangChain |
| Frontend | Streamlit |

## Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/dpkbhatt863/ResearcherAG.git
cd ResearcherAG
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

**3. Add your Groq API key**

Create a `.env` file in the root:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com)

**4. Run the app**
```bash
streamlit run main.py
```

## Project Structure

```
ResearcherRag/
├── main.py              # Streamlit app
├── src/
│   ├── loader.py        # PDF loading and chunking
│   ├── vectorstore.py   # Embeddings and ChromaDB
│   └── rag.py           # Groq RAG chain with multi-query retrieval
├── requirements.txt
└── .env                 # API keys (not committed)
```

## Deployment

Deployed on **Streamlit Community Cloud** — [Live Demo](#) <!-- Replace with your deployed URL -->
