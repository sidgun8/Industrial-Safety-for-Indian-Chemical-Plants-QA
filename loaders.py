import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

def extract_docs(pdf_folder: str, chunk_size=800, chunk_overlap=100):
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(pdf_folder, filename)
            loader = PyMuPDFLoader(filepath)
            docs = loader.load()
            split_docs = splitter.split_documents(docs)
            
            # Add filename as metadata
            for d in split_docs:
                d.metadata["source"] = filename
            all_docs.extend(split_docs)
            
    return all_docs
