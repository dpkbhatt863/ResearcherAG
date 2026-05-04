# src/rag.py
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

PROMPT_TEMPLATE = """You are a research assistant. Use the following paper excerpts to answer the question.
Be precise and cite which paper the information comes from.
If the answer is not in the context, say "I don't have enough information in the provided papers."

{history}Context:
{context}

Question: {question}

Answer:"""


def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )


def build_rag_chain(retriever):
    return {"llm": get_llm(), "retriever": retriever}


def ask(chain: dict, question: str, chat_history: list = None) -> dict:
    docs = chain["retriever"].invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Format last 3 turns of conversation as context
    history = ""
    if chat_history:
        for msg in chat_history[-6:]:
            role = "Human" if msg["role"] == "user" else "Assistant"
            history += f"{role}: {msg['content']}\n"
        history = f"Previous conversation:\n{history}\n"

    prompt = PROMPT_TEMPLATE.format(context=context, question=question, history=history)
    response = chain["llm"].invoke(prompt)

    # Deduplicated sources with page numbers and snippet
    seen = set()
    sources = []
    for doc in docs:
        key = (doc.metadata.get("file_name", "unknown"), doc.metadata.get("page", 0))
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": doc.metadata.get("file_name", "unknown"),
                "page": int(doc.metadata.get("page", 0)) + 1,
                "snippet": doc.page_content[:200]
            })

    return {"answer": response.content, "sources": sources}


def summarize_paper(chunks: list, paper_name: str) -> str:
    llm = get_llm()
    sample = "\n\n".join(c.page_content for c in chunks[:6])
    prompt = f"""Summarize this research paper in exactly 3 concise bullet points.
Paper: {paper_name}

Content:
{sample}

Summary (3 bullet points):"""
    response = llm.invoke(prompt)
    return response.content
