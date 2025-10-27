# 💬 RAG-Pro Chatbot  
### _Streamlit + LangChain + LangGraph + LangSmith + Ollama + ChromaDB_
# ===========================
# RAG-Pro Chatbot Requirements
# ===========================

# Core Dependencies
python-dotenv==1.0.1
streamlit==1.39.0
PyPDF2==3.0.1
numpy==1.26.4
pandas==2.2.3
plotly==5.24.1
scikit-learn==1.5.2

# LangChain Ecosystem
langchain==0.3.2
langchain-core==0.3.6
langchain-community==0.3.1
langchain-ollama==0.2.1
langchainhub==0.1.18
langchain-experimental==0.3.2
langchain-openai==0.3.2

# LangGraph & LangSmith
langgraph==0.2.19
langsmith==0.1.93

# Embeddings + Vector DB
chromadb==0.5.13
faiss-cpu==1.8.0.post1
sentence-transformers==3.1.1
nomic==3.1.1

# LLM + Ollama Connector
requests==2.32.3
httpx==0.27.2

# Utility + Logging
tqdm==4.66.5
coloredlogs==15.0.1

# Optional (Visualization & PCA)
matplotlib==3.9.2
seaborn==0.13.2

# Optional (Benchmarking & Serialization)
csvkit==1.3.0
jsonschema==4.23.0


RAG-Pro Chatbot is an advanced Retrieval-Augmented Generation (RAG) system built for intelligent document-aware conversations. It integrates **Ollama** (for local LLM inference), **LangChain** (for modular LLM orchestration), **LangGraph** (for graph-based agentic workflows), and **LangSmith** (for observability and tracing).  
Users can upload PDFs, visualize embeddings, benchmark response quality, and chat seamlessly with a personalized knowledge base.

---

## 🚀 Features

- 📂 **Document Upload & Parsing** – Upload PDF files and automatically chunk them into context-aware text segments.  
- 🧠 **Embeddings + Vector Search** – Uses ChromaDB for semantic retrieval via FAISS-style dense embeddings.  
- 🔁 **LangChain + LangGraph Integration** – Manage multi-step RAG workflows and modular chains with clear state graphs.  
- 🔍 **LangSmith Integration** – Log and visualize all LangChain traces and LLM calls for debugging and performance insights.  
- 🧩 **Ollama-Powered LLM** – Run local LLMs (e.g., Llama 3.1, Llama 3.2) via `Ollama` API for offline inference.  
- 📊 **Interactive Visualization** – Embedding plots and similarity clustering via PCA + Plotly.  
- 🧪 **Benchmark Mode** – Measure latency and quality metrics across local CPU/GPU runs.  
- 🖥️ **Streamlit Front-End** – Simple and elegant UI for document interaction and chat sessions.

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| **Frontend** | Streamlit |
| **Backend LLM** | Ollama (Llama 3.x) |
| **Vector DB** | ChromaDB |
| **RAG Frameworks** | LangChain, LangGraph |
| **Tracing & Analytics** | LangSmith |
| **Visualization** | Plotly, PCA |
| **Environment** | Python 3.12 +, Virtual Env |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/hemanth2416-byte/rag-pro-chatbot.git
cd rag-pro-chatbot
2️⃣ Create Virtual Environment
bash
Copy code
python -m venv venv
venv\Scripts\activate      # (Windows)
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Set Environment Variables
Create a .env file in the project root:

ini
Copy code
LANGCHAIN_API_KEY=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
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT="RAG-Pro Chatbot"
5️⃣ Run Ollama Server
bash
Copy code
ollama serve
ollama run llama3.2
6️⃣ Launch Streamlit UI
bash
Copy code
streamlit run rag_pro_chatbot/rag_ui.py
Then open your browser at → http://localhost:8501

🧭 Folder Structure
bash
Copy code
rag-pro-chatbot/
│
├── rag_pro_chatbot/
│   ├── rag_ui.py               # Streamlit main UI
│   ├── modules/
│   │   ├── query_classifier.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── retriever.py
│   │   ├── reranker.py
│   │   ├── repacker.py
│   │   ├── summarizer.py
│   │   └── llm_connector.py
│   ├── __init__.py
│
├── benchmarks/                 # Benchmark CSV logs
├── .env                        # Environment variables (ignored)
├── .gitignore
├── README.md
└── requirements.txt
🧩 Example Query Flow
mermaid
Copy code
graph TD
A[User Uploads PDF] --> B[Chunk Text & Embed via Embedder]
B --> C[Store in ChromaDB Vector Store]
C --> D[User Asks Question]
D --> E[Retriever Fetches Relevant Context]
E --> F[Reranker Improves Context Ranking]
F --> G[LLM (Ollama) Generates Response]
G --> H[LangSmith Logs & Visualization]
🧠 Benchmarks (CPU / GPU)
Metric	CPU (Intel i7)	GPU (RTX Titan)
Embedding Time (1000 docs)	6.2 s	0.9 s
Response Latency	2.5 s	0.6 s
Throughput (req/sec)	4.1	13.7

🧾 License
This project is open-source under the MIT License.

🤝 Author
Hemanth Kumar Gajagiri
💼  DevOps & AI Engineer
🌐 GitHub @hemanth2416-byte

“RAG-Pro isn’t just another chatbot — it’s a complete, modular, and benchmark-ready RAG system designed for extensibility and real-world AI applications.”
