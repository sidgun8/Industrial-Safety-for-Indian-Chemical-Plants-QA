import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.docstore.document import Document
from nltk.tokenize import sent_tokenize

def semantic_chunk_text(text: str, chunk_size: int = 200, chunk_overlap: int = 50):
    """
    Split text into chunks of roughly chunk_size sentences with optional overlap.
    """
    sentences = sent_tokenize(text)
    chunks = []
    start = 0
    while start < len(sentences):
        end = min(start + chunk_size, len(sentences))
        chunk_text = " ".join(sentences[start:end])
        chunks.append(chunk_text)
        start += chunk_size - chunk_overlap  # move window forward with overlap
    return chunks

def extract_docs(pdf_folder: str, chunk_size: int = 200, chunk_overlap: int = 50):
    """
    Load PDFs and split into semantic chunks based on sentences.
    Returns a list of LangChain Document objects with metadata.
    """
    all_docs = []

    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(pdf_folder, filename)
            loader = PyMuPDFLoader(filepath)
            raw_docs = loader.load()  # list of Document objects
            for doc in raw_docs:
                page_text = doc.page_content
                page_chunks = semantic_chunk_text(page_text, chunk_size, chunk_overlap)
                for chunk in page_chunks:
                    new_doc = Document(
                        page_content=chunk,
                        metadata={**doc.metadata, "source": filename}
                    )
                    all_docs.append(new_doc)

    return all_docs
