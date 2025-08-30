from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
import os, pickle
from rrf import reciprocal_rank_fusion

CHROMA_DIR = "chroma_db"
BM25_FILE = "bm25.pkl"

class HybridRetriever:
    """Combines BM25 + Dense Vector retrievers using RRF with caching."""

    def __init__(self, docs, k_dense: int = 5, bge_model: str = "BAAI/bge-large-en"):
        self.k_dense = k_dense

        # ---------- Embedding function ----------
        self.embeddings = HuggingFaceEmbeddings(model_name=bge_model, model_kwargs={"device": "cpu"})

        # ---------- Chroma vector store ----------
        if os.path.exists(CHROMA_DIR):
            print("[HybridRetriever] Loading persisted Chroma DB...")
            self.vectordb = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=self.embeddings  # ✅ provide embedding
            )
        else:
            print("[HybridRetriever] Creating new Chroma DB...")
            self.vectordb = Chroma.from_documents(
                docs, self.embeddings, persist_directory=CHROMA_DIR
            )
            self.vectordb.persist()

        # ---------- BM25 Retriever ----------
        if os.path.exists(BM25_FILE):
            print("[HybridRetriever] Loading persisted BM25 index...")
            with open(BM25_FILE, "rb") as f:
                self.bm25 = pickle.load(f)
        else:
            print("[HybridRetriever] Creating new BM25 index...")
            self.bm25 = BM25Retriever.from_documents(docs, k=5)
            with open(BM25_FILE, "wb") as f:
                pickle.dump(self.bm25, f)

    def get_relevant_documents(self, query: str):
        bm25_docs = self.bm25.get_relevant_documents(query)
        vector_docs = self.vectordb.similarity_search(query, k=self.k_dense)
        fused = reciprocal_rank_fusion([bm25_docs, vector_docs], k=60, top_n=5)
        return [doc for doc, _ in fused]