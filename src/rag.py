# src/rag.py
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

PROMPT_TEMPLATE = """Use the following research paper excerpts to answer the question.
Be precise and cite which paper the information comes from.
If the answer is not in the context, say "I don't have enough information in the provided papers."

Context:
{context}

Question: {question}

Answer:"""

def build_rag_chain(retriever):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    return {"llm": llm, "retriever": retriever}


def ask(chain: dict, question: str) -> dict:
    docs = chain["retriever"].invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = chain["llm"].invoke(prompt)

    sources = list({doc.metadata.get("file_name", "unknown") for doc in docs})
    return {"answer": response.content, "sources": sources}
