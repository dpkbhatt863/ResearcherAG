# src/vectorstore.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)

    print(f"Vectorstore built with {len(chunks)} chunks")
    return vectorstore


def get_retriever(vectorstore, k=5):
    # This converts the vectorstore into a retriever
    # k=5 means: return the 5 most similar chunks for any query
    return vectorstore.as_retriever(search_kwargs={"k": k})
