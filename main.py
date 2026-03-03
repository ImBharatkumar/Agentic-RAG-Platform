import uvicorn

if __name__ == "__main__":
    uvicorn.run("enterprice_rag.api.main:app", host="0.0.0.0", port=8000, reload=True)
