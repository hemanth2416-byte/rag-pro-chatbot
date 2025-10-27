import streamlit as st
from PyPDF2 import PdfReader
from modules.query_classifier import QueryClassifier
from modules.chunker import chunk_text
from modules.embedder import Embedder
from modules.retriever import Retriever
from modules.reranker import Reranker
from modules.repacker import repack
from modules.summarizer import Summarizer
from modules.llm_connector import OllamaLLM
import os, shutil, numpy as np, pandas as pd, time, csv
from sklearn.decomposition import PCA
import plotly.express as px
from datetime import datetime

# ---------------------- NEW IMPORTS ----------------------
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langsmith import Client
from langsmith.run_helpers import traceable
from dotenv import load_dotenv

# ---------------------- LOAD ENV ----------------------
load_dotenv()
# Make sure you set your LangSmith key:
# export LANGCHAIN_TRACING_V2="true"
# export LANGCHAIN_PROJECT="RAG-Pro"
# export LANGCHAIN_API_KEY="your_langsmith_api_key"

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="RAG-Pro Chatbot", page_icon="💬", layout="wide")
st.title("💬 RAG-Pro Chatbot (LangChain + LangGraph + LangSmith)")
st.caption("Enhanced with LangGraph orchestration and LangSmith tracing")

# ---------------------- BENCHMARK HELPERS ----------------------
BENCH_DIR = "benchmarks"
os.makedirs(BENCH_DIR, exist_ok=True)
BENCH_FILE = os.path.join(BENCH_DIR, "cpu_results.csv")

def count_tokens(text: str) -> int:
    return len(text.split())

def log_benchmark(query, response, elapsed):
    tokens = count_tokens(response)
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "elapsed_sec": round(elapsed, 3),
        "resp_tokens": tokens,
        "tokens_per_sec": round(tokens / elapsed, 2) if elapsed > 0 else 0,
    }
    write_header = not os.path.exists(BENCH_FILE)
    with open(BENCH_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# ---------------------- SIDEBAR SETTINGS ----------------------
st.sidebar.header("⚙️ Settings")
model_name = st.sidebar.text_input("Ollama model", "llama3.2")
top_k = st.sidebar.slider("Top-K retrieved docs", 1, 10, 5)

# ---------------------- INITIALIZE MODULES ----------------------
if "init_done" not in st.session_state:
    st.session_state.ollama = OllamaLLM(model=model_name)
    st.session_state.classifier = QueryClassifier()
    st.session_state.embedder = Embedder()
    st.session_state.reranker = Reranker()
    st.session_state.summarizer = Summarizer()
    st.session_state.retriever = Retriever(collection_name="rag_collection", persist_dir="./vector_db")
    st.session_state.history = []
    st.session_state.last_query_docs = []
    st.session_state.init_done = True

# ---------------------- LANGGRAPH: Define State ----------------------
class RAGState(TypedDict):
    query: str
    chunks: list[str]
    retrieved_docs: list[str]
    reranked: list[str]
    summarized: str
    answer: str

# ---------------------- LANGGRAPH: Define Nodes ----------------------
@traceable(name="chunk_text_node")
def chunk_node(state: RAGState):
    chunks = chunk_text(state["query"])
    return {"chunks": chunks}

@traceable(name="embed_and_retrieve")
def retrieve_node(state: RAGState):
    retriever = st.session_state.retriever
    embedder = st.session_state.embedder
    docs = retriever.retrieve(state["query"], embedder, top_k=top_k)
    return {"retrieved_docs": docs}

@traceable(name="rerank_docs")
def rerank_node(state: RAGState):
    reranker = st.session_state.reranker
    ranked = reranker.rerank(state["query"], state["retrieved_docs"])
    return {"reranked": ranked}

@traceable(name="summarize_docs")
def summarize_node(state: RAGState):
    summarizer = st.session_state.summarizer
    summarized = summarizer.summarize(repack(state["reranked"]))
    return {"summarized": summarized}

@traceable(name="generate_answer")
def llm_node(state: RAGState):
    llm = st.session_state.ollama
    prompt = f"Use this context:\n{state['summarized']}\n\nQ: {state['query']}"
    answer = llm.generate(prompt)
    return {"answer": answer}

# ---------------------- LANGGRAPH: Build Graph ----------------------
def build_rag_graph():
    builder = StateGraph(RAGState)
    builder.add_node("chunk", chunk_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("llm", llm_node)

    builder.add_edge(START, "chunk")
    builder.add_edge("chunk", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "summarize")
    builder.add_edge("summarize", "llm")
    builder.add_edge("llm", END)

    graph = builder.compile()
    return graph

rag_graph = build_rag_graph()

# ---------------------- FILE UPLOAD ----------------------
uploaded_files = st.file_uploader("📁 Upload PDFs or Text Files", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        all_text = ""
        ext = os.path.splitext(file.name)[1].lower()
        if ext == ".pdf":
            reader = PdfReader(file)
            for page in reader.pages:
                all_text += page.extract_text() or ""
        else:
            all_text += file.read().decode("utf-8")

        corpus = [all_text]
        with st.spinner(f"🔍 Processing {file.name} ..."):
            chunks = [c for doc in corpus for c in chunk_text(doc)]
            embeddings = st.session_state.embedder.embed(chunks)
            metadatas = [{"source": file.name} for _ in chunks]
            st.session_state.retriever.add_documents(chunks, embeddings, metadatas)
        st.success(f"✅ Added {len(chunks)} chunks from {file.name} to ChromaDB.")

# ---------------------- CHAT INTERFACE ----------------------
for role, content in st.session_state.history:
    st.chat_message(role).write(content)

query = st.chat_input("Ask a question...")

if query:
    st.chat_message("user").write(query)
    st.session_state.history.append(("user", query))

    with st.spinner("⚡ Running RAG pipeline (LangGraph + LangSmith tracing)..."):
        start = time.time()

        qc = st.session_state.classifier
        if not qc.needs_retrieval(query):
            answer = st.session_state.ollama.generate(query)
        else:
            result = rag_graph.invoke({"query": query})
            answer = result["answer"]

        elapsed = time.time() - start
        log_benchmark(query, answer, elapsed)

    st.chat_message("assistant").write(answer)
    st.session_state.history.append(("assistant", answer))

# ---------------------- SIDEBAR UTILITIES ----------------------
st.sidebar.header("🧠 Utilities")
if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.history.clear()
    st.sidebar.success("Chat history cleared!")

if st.sidebar.button("🗑️ Clear Vector Database"):
    if os.path.exists("./vector_db"):
        shutil.rmtree("./vector_db")
        st.sidebar.success("Vector database cleared successfully!")
    else:
        st.sidebar.warning("No vector database found to clear.")

if st.sidebar.button("🔄 Refresh Visualization"):
    st.rerun()

# ---------------------- VISUALIZATION SECTION ----------------------
st.divider()
st.subheader("📊 Embedding Visualization (Clustered by Source Document)")

try:
    collection = st.session_state.retriever.collection
    if collection and collection.count() > 0:
        data = collection.get(include=["embeddings", "documents", "metadatas"])
        all_embeddings = np.array(data["embeddings"])
        docs = data["documents"]
        metadatas = data["metadatas"]

        sources = [meta.get("source", "unknown") for meta in metadatas]
        docs = [d[:100] + "..." if len(d) > 100 else d for d in docs]

        reduced = np.zeros((all_embeddings.shape[0], 2)) if all_embeddings.shape[0] < 2 else PCA(n_components=2).fit_transform(all_embeddings)

        df = pd.DataFrame({
            "x": reduced[:, 0],
            "y": reduced[:, 1],
            "source": sources,
            "text": docs,
            "retrieved": [("✅" if any(doc in r for r in st.session_state.last_query_docs) else "🔹") for doc in docs]
        })

        fig = px.scatter(df, x="x", y="y", color="source", symbol="retrieved",
                         hover_data=["text"], title="📈 Embeddings Visualization")
        fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color="DarkSlateGrey")))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📭 No embeddings found yet. Upload documents to visualize.")
except Exception as e:
    st.warning(f"Visualization error: {e}")

# ---------------------- BENCHMARK DASHBOARD ----------------------
st.sidebar.divider()
st.sidebar.subheader("📊 Benchmark Dashboard")
bench_path = os.path.join(BENCH_DIR, "cpu_results.csv")

if os.path.exists(bench_path):
    try:
        df_bench = pd.read_csv(bench_path)
        if not df_bench.empty:
            avg_latency = df_bench["elapsed_sec"].mean()
            avg_throughput = df_bench["tokens_per_sec"].mean()
            total_queries = len(df_bench)

            c1, c2, c3 = st.sidebar.columns(3)
            c1.metric("🕒 Avg Latency (s)", f"{avg_latency:.2f}")
            c2.metric("⚡ Tokens/sec", f"{avg_throughput:.2f}")
            c3.metric("💬 Queries", total_queries)

            fig_bench = px.line(df_bench, x="timestamp", y="tokens_per_sec", title="Tokens/sec Trend", markers=True)
            fig_bench.update_layout(margin=dict(l=10, r=10, t=25, b=10), height=250, xaxis_title=None, yaxis_title="Tokens/sec")
            st.sidebar.plotly_chart(fig_bench, use_container_width=True)
        else:
            st.sidebar.info("No benchmark data yet. Run a few queries first.")
    except Exception as e:
        st.sidebar.warning(f"Benchmark dashboard error: {e}")
else:
    st.sidebar.info("Run some queries to generate benchmark data.")








