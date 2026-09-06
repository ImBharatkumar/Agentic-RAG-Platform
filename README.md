# 🚀 Enterprise Hybrid RAG Platform

> A production-grade, locally-deployable Hybrid Retrieval Augmented Generation (RAG) system featuring **Modular Provider Architecture**, **Contextual Retrieval**, **Agentic Orchestration**, and a modern **React UI**.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-18.3+-61DAFB.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.4+-646CFF.svg)](https://vitejs.dev/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0+-green.svg)](https://github.com/langchain-ai/langgraph)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-blue.svg)](https://github.com/pgvector/pgvector)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

---

## 📚 Documentation

- 📄 **[System Architecture & Technical Deep-Dive](docs/architecture.md)**: End-to-end component analysis, data flow diagrams, architectural decision records (ADRs), and interview defense.
- 🛠️ **[Operational Runbook & Troubleshooting Guide](docs/troubleshooting.md)**: Common failure modes, recursion limits, latency optimizations, and database fixes.

---

## 🎯 Overview

The **Enterprise Hybrid RAG Platform** is a sophisticated AI system designed for high-precision document retrieval and reasoning. It features a fully **modular architecture** that allows you to swap LLMs, Embedding models, and Vector Stores with zero changes to the core agent logic.

### Core Capabilities

- **🧩 Modular Provider Design**: Decoupled interfaces for LLMs (Ollama), Embeddings (Gemini), and Storage (Postgres) using the Provider Pattern.
- **🔍 Hybrid Search**: Merges dense semantic retrieval (pgvector) with sparse keyword retrieval (tsvector) using Reciprocal Rank Fusion (RRF).
- **🧠 Contextual Retrieval**: Automatically generates situational context for document chunks, enriching embeddings with document-level awareness.
- **🤖 Agentic Orchestration**: Built on **LangGraph**, featuring self-reflective loops, query rewriting, checkpointing, and iterative refinement.
- **📄 Advanced Ingestion**: High-fidelity OCR and layout analysis for complex documents via **Docling**.
- **⚡ Modern Streaming UI**: Responsive **React + Vite** single-page application with real-time token streaming, session history, and interactive document ingestion.

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────┐
│              Frontend Layer (React 18 + Vite)           │
│        Chat & History  •  Document Ingest Panel         │
└────────────────────────────┬────────────────────────────┘
                             │ HTTP / SSE Stream
                             ▼
┌─────────────────────────────────────────────────────────┐
│              API Gateway (FastAPI + Uvicorn)            │
│       /chat (Streaming)  •  /ingest  •  /sessions       │
└──────────────┬───────────────────────────┬──────────────┘
               │                           │
  [Ingestion Path]                [Query / Reasoning Path]
               ▼                           ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐
│  Docling OCR & Ingestion     │ │  LangGraph Agentic Workflow  │
│  Contextual Sentence Chunking│ │  Query Analyzer • Classifier │
│  Dense Embedder (Jina/Gemini)│ │  Self-Reflection • Generator │
└──────────────┬───────────────┘ └──────────────┬───────────────┘
               │                                │
               ▼                                ▼
┌─────────────────────────────────────────────────────────┐
│        PostgreSQL 14+ Storage Engine (pgvector)         │
│   Dense Vector (<=>)  •  Sparse TSVector (BM25)  •  RRF │
│        PostgresSaver (Episodic Thread Checkpoints)      │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```text
enterprice_rag/
├── docs/                   # Engineering & Architecture documentation
│   ├── architecture.md     # In-depth system architecture & interview guide
│   └── troubleshooting.md  # Operational runbook & failure recovery
├── frontend/               # React + Vite Frontend
│   ├── src/
│   │   ├── components/     # UI Components (ChatWindow, InputBar, MessageBubble, UploadPanel)
│   │   ├── styles/         # CSS styles
│   │   ├── App.jsx         # Main React application component
│   │   └── main.jsx        # React DOM entry point
│   ├── package.json        # Frontend dependencies & scripts
│   └── vite.config.js      # Vite configuration & proxy settings
├── src/
│   └── enterprice_rag/
│       ├── agents/
│       │   └── rag_agent/  # Modular LangGraph (state, nodes, graph)
│       ├── api/            # FastAPI REST endpoints & streaming routes
│       ├── config/         # System settings & model configurations
│       ├── core/           # Interfaces (Protocols) & Provider Factory
│       ├── evaluation/     # RAGAS evaluation runner & golden dataset
│       ├── ingestion/      # Document parsing & OCR logic (Docling)
│       ├── processing/     # Modular chunking & retrieval wrappers
│       ├── providers/      # Pluggable implementations (LLM, Embeddings, Storage)
│       ├── storage/        # DB models, postgres client & pgvector store
│       └── utils/          # Shared utilities (logging, text utils)
├── main.py                 # API Launcher (Uvicorn / FastAPI)
├── architecture.md         # Quick-start Architecture overview
├── pyproject.toml          # UV configuration & Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.12+**
- **Node.js 18+** & **npm**
- **uv** package manager
- **PostgreSQL 14+** with `pgvector` extension
- **Ollama** (for local LLMs)

### Installation

1. **Clone and Sync Python Environment**
   ```bash
   git clone https://github.com/yourusername/enterprice_rag.git
   cd enterprice_rag
   uv sync
   ```

2. **Install Frontend Dependencies**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

3. **Database Setup**
   ```bash
   createdb rag
   psql rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```

### Configuration
Edit `src/enterprice_rag/config/settings.py` or provide a `.env` file to configure your database URL, model selections, and API keys.

---

## 💻 Usage

### Launching the System

#### 1. Start the Backend API (FastAPI)
```bash
uv run python main.py
```
*API will run at `http://localhost:8000` with Swagger docs at `http://localhost:8000/docs`.*

#### 2. Start the Frontend (React + Vite)
```bash
cd frontend
npm run dev
```
*Frontend will run at `http://localhost:5173`.*

### Programmatic Access
```python
from enterprice_rag.agents.rag_agent.graph import run_agent

# Stream responses from the modular agentic loop
for chunk in run_agent("What are the key instructions for tenderers?", thread_id="session_1"):
    print(chunk, end="", flush=True)
```

---

## 🛠️ Technology Stack

- **UI / Frontend**: [React](https://react.dev/) + [Vite](https://vitejs.dev/)
- **Backend API**: [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph)
- **Package Manager**: [uv](https://github.com/astral-sh/uv)
- **Document Parsing**: [Docling](https://github.com/DS4SD/docling)
- **Vector Search**: [PostgreSQL](https://www.postgresql.org/) + [pgvector](https://github.com/pgvector/pgvector)
- **Inference**: [Ollama](https://ollama.ai/) / [Gemini](https://ai.google.dev/)
- **Evaluation**: [RAGAS](https://github.com/explodinggradients/ragas)

---

<div align="center">
Built with ❤️ for Modular Enterprise RAG
</div>
