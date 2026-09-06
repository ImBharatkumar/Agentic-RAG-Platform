# 🏛️ Enterprise Hybrid RAG Platform — System Architecture & Engineering Deep-Dive

> Complete documentation is maintained in the [`docs/`](file:///home/barry/enterprice_rag/docs/) folder:
> - 📄 **[Detailed Architecture Specification](file:///home/barry/enterprice_rag/docs/architecture.md)**
> - 🛠️ **[Operational Runbook & Troubleshooting Guide](file:///home/barry/enterprice_rag/docs/troubleshooting.md)**

---

## 🏛️ High-Level System Architecture

```mermaid
graph TB
    subgraph "Frontend Client (React 18 + Vite)"
        UI["Modern UI / SPA<br/>(Tailwind CSS + EventStream)"]
        Chat["Chat & History Window"]
        Upload["Document Ingest Panel"]
    end

    subgraph "API Gateway (FastAPI + Uvicorn)"
        API["FastAPI REST & Streaming Server"]
        CORS["CORS Middleware"]
        SSE["Token-by-Token StreamingResponse"]
    end

    subgraph "Ingestion & Context Generation"
        Docling["Docling Document OCR / Layout Parser"]
        Chunker["Sentence Chunker (8 sentences/chunk)"]
        ContextGen["Document-Aware Context Enricher"]
        Embedder["Dense Embedder (Jina / Gemini)"]
    end

    subgraph "PostgreSQL 14+ Storage Engine"
        PGVector["pgvector (Dense Cosine Similarity <=> )"]
        TSVector["to_tsvector & plainto_tsquery (Sparse BM25)"]
        RRF["Reciprocal Rank Fusion (k=60)"]
        Checkpointer["PostgresSaver (Episodic Thread Checkpoints)"]
    end

    subgraph "Agentic Reasoning Loop (LangGraph)"
        direction TB
        QA["Query Analyzer & Classifier"]
        Retriever["Hybrid Retriever Node"]
        Reflector["Self-Reflection / Grader Node"]
        Generator["Grounded Answer Generator"]
    end

    subgraph "Inference Providers (Ollama / Gemini)"
        LLM["Granite 4.1:3b / Qwen2.5:3b (Task-Optimized)"]
    end

    UI --> API
    API --> Docling
    Docling --> Chunker --> ContextGen --> Embedder --> PGVector & TSVector
    API --> SSE --> QA
    QA -->|Retrieve| Retriever
    QA -->|Generate Direct| Generator
    Retriever --> PGVector & TSVector --> RRF --> Reflector
    Reflector -->|Relevant ('yes')| Generator
    Reflector -->|Inadequate ('no')| QA
    Generator --> LLM
    Checkpointer -.-> QA & Generator
    Generator --> SSE
```

---

## 📋 Core Architectural Highlights

1. **Contextual Retrieval**: Solves chunk myopia by prepending document-level context before embedding generation.
2. **PostgreSQL Dual-Stage Hybrid Search**: Combines `pgvector` dense cosine similarity with `tsvector` sparse BM25 via Reciprocal Rank Fusion ($k=60$).
3. **LangGraph Agentic Orchestration**: Implements self-reflective loops, automated query expansion, and multi-turn reference resolution.
4. **Episodic PostgreSQL Memory**: Uses `PostgresSaver` connection pooling to checkpoint state across conversational turns.
5. **Real-Time Token Streaming**: Streams generated tokens directly from LangGraph callbacks through FastAPI to the React UI.

👉 **For in-depth analysis, design trade-offs, and interview defense questions, see [`docs/architecture.md`](file:///home/barry/enterprice_rag/docs/architecture.md).**
