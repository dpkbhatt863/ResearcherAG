from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

def process_all_pdfs(directory_path):
    all_documents = []
    pdf_dir = Path(directory_path)

    #Find all pdf files recursively

    pdf_files = list(pdf_dir.glob("**/*.pdf"))

    print(f"Found {len(pdf_files)} files")

    for pdf in pdf_files:
        print(f"File Name:{pdf.name}")

        try:
            loader = PyMuPDFLoader(str(pdf))

            documents = loader.load()

            for doc in documents:
                doc.metadata['file_name']= pdf.name
                doc.metadata['file_type']='pdf'

            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} pages")

        except Exception as e:
            print("Error")

    print(f"\nTotal documents loaded:{len(all_documents)}")
    return all_documents

### Text splitting get into chunks

def split_documents(documents,chunk_size=1000,chunk_overlap=200):
    """Split documents into smaller chunks for better RAG performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    
    # Show example of a chunk
    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")
    
    return split_docs


    

