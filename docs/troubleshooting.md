# 🛠️ Operational Troubleshooting & Runbook

This document details diagnostic procedures, error resolutions, performance tuning, and operational best practices for the **Enterprise Hybrid RAG Platform**.

---

## 📑 Common Issues and Runbook Solutions

### 1. LangGraph Recursion Limit Reached
**Symptom:**
```text
Error: Recursion limit of 25 reached without hitting a stop condition.
```
**Root Cause:**
- The graph cycled between `query_analyzer` $\leftrightarrow$ `retriever` $\leftrightarrow$ `reflection` repeatedly because the reflection node answered `"no"` on every pass, and the iteration counter wasn't stopping the loop.

**Resolution Steps:**
1. Check [`src/enterprice_rag/agents/rag_agent/graph.py`](file:///home/barry/enterprice_rag/src/enterprice_rag/agents/rag_agent/graph.py):
   ```python
   def should_continue(state):
       if state.get("reflection") == "yes":
           return "generate"
       elif state.get("iterations", 0) >= MAX_ITERATIONS:
           return "generate"
       else:
           return "query_analyzer"
   ```
2. Verify `MAX_ITERATIONS` in `.env` (default is `2`).
3. Ensure `recursion_limit` in `run_agent()` is set to at least `20`–`25`.

---

### 2. High Latency During Query Execution (>10s)
**Symptom:**
Chat queries take a long time to start streaming.

**Root Cause:**
- **Model swapping in Ollama**: If different models are set for `MODEL_QUERY_REWRITE`, `MODEL_REFLECTION`, and `MODEL_GENERATION`, Ollama spends 2–4 seconds unloading and loading weights into memory on each step.
- **Reasoning models used on intermediate steps**: Reasoning models emit hundreds of `<think>` tokens for simple yes/no checks.

**Resolution Steps:**
1. Run `ollama ps` to see if models are constantly being swapped.
2. Unify all tasks to a single fast model in `.env`:
   ```env
   MODEL_QUERY_REWRITE=granite4.1:3b
   MODEL_REFLECTION=granite4.1:3b
   MODEL_GENERATION=granite4.1:3b
   MODEL_CONTEXT_GENERATION=granite4.1:3b
   MODEL_DEFAULT=granite4.1:3b
   ```
3. Ensure strict `num_predict` limits are active in [`src/enterprice_rag/providers/llm/ollama.py`](file:///home/barry/enterprice_rag/src/enterprice_rag/providers/llm/ollama.py) (`num_predict=10` for reflection).

---

### 3. Database Connection Failure
**Symptom:**
```text
psycopg.OperationalError: connection to server at "localhost", port 5432 failed: Connection refused
```

**Resolution Steps:**
1. Check PostgreSQL service:
   ```bash
   sudo systemctl status postgresql
   ```
2. Restart PostgreSQL:
   ```bash
   sudo systemctl restart postgresql
   ```
3. Test connection string and pgvector extension:
   ```bash
   psql -U postgres -d rag -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
   ```

---

### 4. Docling OCR Extraction Issues on Large PDFs
**Symptom:**
Ingestion times out or fails with memory warnings on large scanned documents.

**Resolution Steps:**
1. Verify document is within readable limits (PDF is not password-protected or corrupted).
2. Check temporary files in `/tmp` are being cleaned up properly.
3. For large PDFs (>100 pages), process in batches or use asynchronous queue workers.

---

### 5. Frontend Streaming Doesn't Render Tokens
**Symptom:**
Frontend receives the complete response at once rather than typewriter streaming.

**Resolution Steps:**
1. Verify FastAPI is using `StreamingResponse` in `src/enterprice_rag/api/main.py`.
2. Verify the React client uses `fetch` with a `ReadableStreamDefaultReader` rather than awaiting full `res.json()`.
3. Check proxy settings in `frontend/vite.config.js` to ensure no buffering middleware is intercepting HTTP chunks.
