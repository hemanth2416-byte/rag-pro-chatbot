from modules.query_classifier import QueryClassifier
from modules.chunker import chunk_text
from modules.embedder import Embedder
from modules.retriever import Retriever  # now uses ChromaDB
from modules.reranker import Reranker
from modules.repacker import repack
from modules.summarizer import Summarizer
from modules.llm_connector import OllamaLLM

# -------------------------------------------------------------
# Connect to your local Ollama model (supports llama3, mistral, etc.)
# -------------------------------------------------------------
ollama = OllamaLLM(model="llama3")

def simple_llm(prompt):
    """Send prompt to Ollama and return generated text."""
    return ollama.generate(prompt)


def rag_pipeline(query, corpus):
    """End-to-end Retrieval-Augmented Generation pipeline with ChromaDB."""

    # ------------------------------
    # 1. Check if retrieval is needed
    # ------------------------------
    qc = QueryClassifier()
    if not qc.needs_retrieval(query):
        return simple_llm(query)

    # ------------------------------
    # 2. Chunk input corpus
    # ------------------------------
    chunks = [c for doc in corpus for c in chunk_text(doc)]
    embedder = Embedder()
    embeddings = embedder.embed(chunks)

    # ------------------------------
    # 3. Store embeddings in persistent ChromaDB
    # ------------------------------
    retriever = Retriever(collection_name="rag_collection", persist_dir="./vector_db")
    retriever.add_documents(chunks, embeddings)

    # ------------------------------
    # 4. Retrieve top relevant chunks
    # ------------------------------
    top_docs = retriever.retrieve(query, embedder, top_k=5)

    # ------------------------------
    # 5. Rerank documents
    # ------------------------------
    reranker = Reranker()
    reranked = reranker.rerank(query, top_docs)

    # ------------------------------
    # 6. Repack and summarize context
    # ------------------------------
    packed = repack(reranked)
    summarizer = Summarizer()
    summarized = summarizer.summarize(packed)

    # ------------------------------
    # 7. Generate final answer using Ollama
    # ------------------------------
    final_prompt = f"Question: {query}\n\nContext:\n{summarized}\n\nAnswer:"
    return simple_llm(final_prompt)


# -------------------------------------------------------------
# Run standalone CLI test
# -------------------------------------------------------------
if __name__ == "__main__":
    corpus = [
        "Artificial intelligence enables machines to learn from data.",
        "Retrieval Augmented Generation (RAG) combines retrieval and generation to enhance factual accuracy of LLMs."
    ]
    print(rag_pipeline("What is RAG?", corpus))





