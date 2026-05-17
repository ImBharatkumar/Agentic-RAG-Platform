import uvicorn
import sys
import os

# Add src to PYTHONPATH programmatically
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

if __name__ == "__main__":
    uvicorn.run("enterprice_rag.api.main:app", host="0.0.0.0", port=8000, reload=True)
