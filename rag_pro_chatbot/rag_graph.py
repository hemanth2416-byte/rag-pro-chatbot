# rag_pro_chatbot/rag_graph.py
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from modules.chunker import chunk_text
from modules.embedder import Embedder
from modules.retriever import Retriever
from modules.reranker import Reranker
from modules.summarizer import Summarizer
from modules.llm_connector import OllamaLLM

# --- 1. Define shared state ---
class RAGState(TypedDict):
    query: str
    chunks: list[str]
    embeddings: list
    retrieved_docs: list[str]
    reranked_docs: list[str]
    summary: str
    answer: str

# --- 2. Create Node Functions ---
def chunk_node(state: RAGState):
    chunks = chunk_text(state["query"])
    return {"chunks": chunks}

def embed_node(state: RAGState):
    embedder = Embedder(model="nomic-embed-text")
    embeddings = embedder.embed(state["chunks"])
    return {"embeddings": embeddings}

def retrieve_node(state: RAGState):
    retriever = Retriever()
    docs = retriever.retrieve(state["query"])
    return {"retrieved_docs": docs}

def rerank_node(state: RAGState):
    reranker = Reranker()
    ranked = reranker.rerank(state["retrieved_docs"])
    return {"reranked_docs": ranked}

def summarize_node(state: RAGState):
    summarizer = Summarizer()
    summary = summarizer.summarize(state["reranked_docs"])
    return {"summary": summary}

def llm_node(state: RAGState):
    llm = OllamaLLM(model="llama3.2")
    prompt = f"Use this context:\n{state['summary']}\n\nQ: {state['query']}"
    answer = llm.invoke(prompt)
    return {"answer": answer}

# --- 3. Build Graph ---
def build_rag_graph():
    builder = StateGraph(RAGState)
    builder.add_node("chunk", chunk_node)
    builder.add_node("embed", embed_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("llm", llm_node)

    builder.add_edge(START, "chunk")
    builder.add_edge("chunk", "embed")
    builder.add_edge("embed", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "summarize")
    builder.add_edge("summarize", "llm")
    builder.add_edge("llm", END)

    graph = builder.compile()
    return graph
