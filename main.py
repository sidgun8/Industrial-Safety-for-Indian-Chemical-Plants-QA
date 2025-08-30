'''
from loaders import extract_docs
from retrievers import HybridRetriever
from llm import answer_with_llama

PDF_FOLDER = "/Users/siddharthsrinivasan/Desktop/RAG_ChemE_Safety/Safety_manuals"

if __name__ == "__main__":
    docs = extract_docs(PDF_FOLDER)
    hybrid_retriever = HybridRetriever(docs)
    
    query = "How to be safe in a chemical plant?"
    answer = answer_with_llama(query, hybrid_retriever, show_chunks=True)
    
    print("\n==== Final Answer ====\n", answer)
'''

# main.py
from loaders import extract_docs
from retrievers import HybridRetriever
from llm import answer_with_llama
import os

PDF_FOLDER = os.path.join(os.path.dirname(__file__), "Safety_manuals")

# Global retriever (to avoid rebuilding each query)
hybrid_retriever = None

def ingest():
    """Load documents and build retriever."""
    global hybrid_retriever
    docs = extract_docs(PDF_FOLDER)
    hybrid_retriever = HybridRetriever(docs)
    return hybrid_retriever

def query_rag(query: str, show_chunks: bool = False):
    """Ask a question with RAG pipeline."""
    global hybrid_retriever
    if hybrid_retriever is None:
        ingest()  # lazy load if not already ingested
    return answer_with_llama(query, hybrid_retriever, show_chunks=show_chunks)

if __name__ == "__main__":
    q = "How to be safe in a chemical plant?"
    ans = query_rag(q, show_chunks=True)
    print("\n==== Final Answer ====\n", ans)
