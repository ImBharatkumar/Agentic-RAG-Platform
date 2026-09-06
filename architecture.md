# 🏛️ Enterprise Hybrid RAG Platform — System Architecture & Engineering Deep-Dive

> Complete documentation is maintained in the [`docs/`](file:///home/barry/enterprice_rag/docs/) folder:
> - 🌐 **[Interactive Architecture & Data Flow Visualizer](file:///home/barry/enterprice_rag/docs/system_overview.html)**
> - 📄 **[Detailed Architecture Specification](file:///home/barry/enterprice_rag/docs/architecture.md)**
> - 🛠️ **[Operational Runbook & Troubleshooting Guide](file:///home/barry/enterprice_rag/docs/troubleshooting.md)**

---

## 🏛️ High-Level System Architecture

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

## 📋 Core Architectural Highlights

1. **Contextual Retrieval**: Solves chunk myopia by prepending document-level context before embedding generation.
2. **PostgreSQL Dual-Stage Hybrid Search**: Combines `pgvector` dense cosine similarity with `tsvector` sparse BM25 via Reciprocal Rank Fusion ($k=60$).
3. **LangGraph Agentic Orchestration**: Implements self-reflective loops, automated query expansion, and multi-turn reference resolution.
4. **Episodic PostgreSQL Memory**: Uses `PostgresSaver` connection pooling to checkpoint state across conversational turns.
5. **Real-Time Token Streaming**: Streams generated tokens directly from LangGraph callbacks through FastAPI to the React UI.

👉 **For interactive visual flowcharts, open [`docs/system_overview.html`](file:///home/barry/enterprice_rag/docs/system_overview.html). For in-depth analysis, design trade-offs, and interview defense questions, see [`docs/architecture.md`](file:///home/barry/enterprice_rag/docs/architecture.md).**
