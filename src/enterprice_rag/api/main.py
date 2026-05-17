from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from enterprice_rag.agents.rag_agent.graph import run_agent
from enterprice_rag.storage.vector_store import process_document
from enterprice_rag.storage.postgres_client import SessionLocal
from enterprice_rag.ingestion.pdf_ingest import docling_ocr
from pathlib import Path
import tempfile
import os

app = FastAPI(title="Enterprise RAG API")

class Query(BaseModel):
    text: str
    top_k: Optional[int] = 5

@app.get("/")
async def root():
    return {"message": "Enterprise RAG API is running"}

from fastapi.responses import StreamingResponse

@app.post("/chat")
async def chat(query: Query):
    try:
        return StreamingResponse(run_agent(query.text), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            markdown_text = docling_ocr(tmp_path)
            if not markdown_text:
                results.append({"filename": file.filename, "status": "Error", "message": "Failed to extract text"})
                continue

            session = SessionLocal()
            try:
                doc_id = Path(file.filename).stem
                num_chunks = process_document(doc_id, markdown_text, session)
                results.append({"filename": file.filename, "status": "Success", "chunks": num_chunks})
            finally:
                session.close()
                os.unlink(tmp_path)
        except Exception as e:
            results.append({"filename": file.filename, "status": "Error", "message": str(e)})
    
    return {"results": results}
