import os
import torch
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHROMA_DIR = "chroma_db"

class BGEEmbeddingWrapper(Embeddings):
    """Wrapper for BGE embeddings compatible with LangChain/Chroma."""

    def __init__(self, model_name="BAAI/bge-small-en-v1.5", device="cpu"):
        print(f"Loading embedding model {model_name}...")
        self.device = device
        self.model = SentenceTransformer(model_name, device="cpu")

        # Optional MPS fallback
        if device == "mps" and torch.backends.mps.is_available():
            try:
                self.model = self.model.to("mps")
                print("✅ Using MPS acceleration")
            except Exception as e:
                print("⚠️ MPS failed, falling back to CPU:", e)
                self.model = self.model.to("cpu")

    def embed_documents(self, texts):
        return [emb.tolist() for emb in self.model.encode(texts, convert_to_numpy=True)]

    def embed_query(self, query):
        return self.model.encode([query], convert_to_numpy=True)[0].tolist()


class ChromaRetriever:
    """Chroma vector store retriever."""

    def __init__(self, docs, persist_directory=CHROMA_DIR, device="cpu", force_reindex=False):
        self.embedding_fn = BGEEmbeddingWrapper(device=device)

        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)

        # If forcing reindex or DB is missing, clear old DB
        if force_reindex and os.path.exists(persist_directory):
            import shutil
            shutil.rmtree(persist_directory)
            os.makedirs(persist_directory, exist_ok=True)
            print("[ChromaRetriever] Cleared existing Chroma DB")

        # Create Chroma vector store
        self.vectordb = Chroma.from_documents(
            docs, self.embedding_fn, persist_directory=persist_directory
        )

    def get_relevant_documents(self, query, k=5):
        return self.vectordb.similarity_search(query, k=k)
