from retrievers import ChromaRetriever
from llm import answer_with_llama
from loaders import extract_docs
import os

PDF_FOLDER = os.path.join(os.path.dirname(__file__), "Safety_manuals")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# Global retriever variable
chroma_retriever = None

def ingest(force_reindex=False):
    """Load documents and build Chroma retriever."""
    global chroma_retriever
    docs = extract_docs(PDF_FOLDER)
    chroma_retriever = ChromaRetriever(docs, persist_directory=CHROMA_DIR, force_reindex=force_reindex)
    return chroma_retriever

def query_rag(query, show_chunks=False, k=5):
    """Query the RAG pipeline and generate final answer."""
    global chroma_retriever
    if chroma_retriever is None:
        ingest()

    results = chroma_retriever.get_relevant_documents(query, k=k)

    if show_chunks:
        print("\n==== Retrieved Chunks ====\n")
        for i, d in enumerate(results, 1):
            print(f"{i}. [{d.metadata.get('source','unknown')}]")
            print(d.page_content[:500] + ("..." if len(d.page_content) > 500 else ""))
            print()

    context = "\n\n---\n\n".join(
        f"[{d.metadata.get('source','unknown')}] {d.page_content}" for d in results
    )

    prompt = f"""
You are a strict assistant that answers ONLY using the provided documents.

Question: {query}

Context:
{context}

Answer:"""

    answer = answer_with_llama(prompt)
    return answer
