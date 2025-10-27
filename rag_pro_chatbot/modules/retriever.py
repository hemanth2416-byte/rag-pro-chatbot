import chromadb
from chromadb.config import Settings
import numpy as np

class Retriever:
    def __init__(self, collection_name="rag_collection", persist_dir="./vector_db"):
        """Initialize a persistent ChromaDB collection."""
        self.client = chromadb.Client(Settings(persist_directory=persist_dir))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        print(f"✅ Connected to ChromaDB at {persist_dir}")

    def add_documents(self, chunks, embeddings, metadatas):
        """Add new chunks with embeddings and metadata to the vector database."""
        if not self.collection:
            self.collection = self.client.get_or_create_collection(name="rag_collection")

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=np.array(embeddings).tolist(),
            metadatas=metadatas
        )
        print(f"✅ Added {len(chunks)} documents to ChromaDB.")

    def retrieve(self, query, embedder, top_k=5):
        """Retrieve top K most relevant documents based on semantic similarity."""
        if not self.collection or self.collection.count() == 0:
            print("⚠️ No documents found in ChromaDB collection.")
            return []

        query_embedding = embedder.embed([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas"]
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        scored_docs = [
            f"{meta.get('source', 'unknown')}: {doc}"
            for doc, meta in zip(documents, metadatas)
        ]
        return scored_docs




