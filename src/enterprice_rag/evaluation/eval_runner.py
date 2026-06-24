import os
import sys

# Ensure the src/ directory is on sys.path so `enterprice_rag` can be found
# regardless of how this script is invoked (uv run, python, etc.)
_src_path = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, os.path.abspath(_src_path))

import json
import asyncio
import pandas as pd
import nest_asyncio
from datasets import Dataset


# LangChain wrappers
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings

# Ragas evaluation imports
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

# Enterprise RAG imports
from enterprice_rag.config.settings import GEMINI_API_KEY
from enterprice_rag.core.factory import get_embedding
from enterprice_rag.agents.rag_agent.graph import app as rag_app

# Apply nest_asyncio to support nested event loops (required for Ragas async execution in some environments)
nest_asyncio.apply()

class CustomLangChainEmbeddings(Embeddings):
    """
    Adapter to bridge our factory's EmbeddingProvider to LangChain's Embeddings interface.
    This ensures Ragas uses the exact same embedding model used in our RAG retrieval.
    """
    def __init__(self):
        self._provider = get_embedding()

    def embed_query(self, text: str) -> list[float]:
        return self._provider.embed_text(text, task_type="retrieval_query")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._provider.embed_batch(texts, task_type="retrieval_document")


def run_rag_pipeline(query: str) -> tuple[list[str], str]:
    """
    Executes the active LangGraph RAG workflow and collects the 
    retrieved contexts and the final generated answer.
    """
    inputs = {"original_query": query, "iterations": 0}
    # Invoke the compiled LangGraph workflow synchronously
    state = rag_app.invoke(inputs, {"recursion_limit": 5})
    
    # Extract retrieved chunks
    retrieved_chunks = [chunk["content"] for chunk in state.get("context", [])]
    
    # Resolve the streamed generator answer to a full string
    generated_answer = state.get("answer", "")
    if hasattr(generated_answer, "__iter__") and not isinstance(generated_answer, str):
        generated_answer = "".join(list(generated_answer))
        
    return retrieved_chunks, generated_answer


def main():
    # 1. Load Golden Dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    if not os.path.exists(dataset_path):
        print(f"Error: Golden dataset not found at {dataset_path}")
        return

    with open(dataset_path, "r") as f:
        golden_data = json.load(f)

    print(f"Loaded {len(golden_data)} test cases from golden dataset.")

    # 2. Run test cases through our RAG system
    eval_samples = []
    for idx, item in enumerate(golden_data, 1):
        question = item["question"]
        print(f"\n[{idx}/{len(golden_data)}] Running RAG pipeline for query: '{question}'...")
        
        contexts, answer = run_rag_pipeline(question)
        
        print(f"-> Retrieved {len(contexts)} context chunks.")
        print(f"-> Generated Answer: {answer[:100]}...")

        eval_samples.append({
            "question": question,
            "contexts": contexts,
            "answer": answer,
            "ground_truth": item["ground_truth"]
        })

    # 3. Create Hugging Face Dataset from samples
    hf_dataset = Dataset.from_list(eval_samples)

    # 4. Initialize LLM-as-a-Judge and custom embeddings
    # Using Gemini's OpenAI-compatible endpoint for robust evaluation
    evaluator_llm = ChatOpenAI(
        model="gemini-2.5-flash-lite",  # Lighter model with higher free-tier quota
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    evaluator_embeddings = CustomLangChainEmbeddings()

    # 5. Define metrics to evaluate
    metrics = [
        faithfulness,         # Grounding (Is the answer derived from retrieved context?)
        answer_relevancy,     # Generation Relevance (Does the answer address the question?)
        context_recall,       # Retrieval Recall (Did we retrieve what was in the ground truth?)
        context_precision     # Retrieval Precision (Are retrieved items relevant to the query?)
    ]

    print("\nStarting Ragas evaluation with Gemini LLM-as-a-Judge...")
    
    # 6. Execute Ragas Evaluation
    results = evaluate(
        dataset=hf_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    # 7. Print and Save Results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(results)
    print("=" * 60)

    # Export detailed results to a pandas dataframe and CSV
    df = results.to_pandas()
    output_csv = "rag_evaluation_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"Detailed evaluation metrics saved to '{output_csv}'")


if __name__ == "__main__":
    main()
