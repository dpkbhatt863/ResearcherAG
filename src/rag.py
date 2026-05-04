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


def _expand_query(llm, question: str) -> list[str]:
    prompt = f"""You are helping retrieve research paper content. Generate 3 alternative search queries for the question below.
Rules:
- Rephrase using different vocabulary (e.g. "safe" → "health benefits", "harmful effects", "risks")
- Include both positive framing (benefits, healthy) and negative framing (risks, side effects)
- Always include the full subject name (e.g. if user says "whey", write "whey protein")
- Return only 3 lines, no numbering, no extra text

Question: {question}

Alternative queries:"""
    response = llm.invoke(prompt)
    variants = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    return [question] + variants[:3]


def _multi_query_retrieve(retriever, llm, question: str) -> list:
    queries = _expand_query(llm, question)
    seen_ids = set()
    all_docs = []
    for q in queries:
        for doc in retriever.invoke(q):
            key = (doc.metadata.get("file_name", ""), doc.metadata.get("page", 0), doc.page_content[:100])
            if key not in seen_ids:
                seen_ids.add(key)
                all_docs.append(doc)
    return all_docs


def ask(chain: dict, question: str, chat_history: list = None) -> dict:
    docs = _multi_query_retrieve(chain["retriever"], chain["llm"], question)
    context = "\n\n".join(doc.page_content for doc in docs)

    history = ""
    if chat_history:
        for msg in chat_history[-6:]:
            role = "Human" if msg["role"] == "user" else "Assistant"
            history += f"{role}: {msg['content']}\n"
        history = f"Previous conversation:\n{history}\n"

    prompt = PROMPT_TEMPLATE.format(context=context, question=question, history=history)
    response = chain["llm"].invoke(prompt)

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
