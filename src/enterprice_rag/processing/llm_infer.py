import requests
import json
from enterprice_rag.config.settings import OLLAMA_BASE_URL, MODELS


def query_ollama(prompt: str, task: str = "generation", temperature: float = 0.3):
    """
    Query Ollama with task-specific model selection.

    Args:
        prompt: The prompt to send to the model
        task: Task type - one of ["query_rewrite", "reflection", "generation"]
        temperature: Sampling temperature (lower = more focused)

    Returns:
        str: Model response text
    """
    # Select model based on task
    model = MODELS.get(task, MODELS.get("llm", "nemotron-mini:latest"))

    # Task-specific temperature overrides
    temp_map = {
        "query_rewrite": 0.5,  # Slightly creative
        "reflection": 0.1,  # Very deterministic
        "generation": 0.3,  # Balanced
    }
    temperature = temp_map.get(task, temperature)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_predict": 1024 if task in ["generation", "context_generation"] else 128,
        },
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=300
        )
        response.raise_for_status()
        result = response.json()
        answer = result.get("response", "").strip()
        print(f"DEBUG: Ollama response length: {len(answer)}")

        # Always attempt to extract final answer if reasoning tags exist
        answer = extract_reasoning_answer(answer)
        if not answer:
            print(f"DEBUG: Extract reasoning returned empty string for: '{result.get('response', '')[:100]}...'")

        return answer

    except requests.exceptions.RequestException as e:
        print(f"Error querying Ollama ({model}): {e}")
        return "Error: Could not connect to LLM service."
    except json.JSONDecodeError as e:
        print(f"Error parsing response: {e}")
        return "Error: Invalid response from LLM."


def extract_reasoning_answer(text: str) -> str:
    """
    Extract final answer from models that output reasoning (e.g. DeepSeek-R1, Phi-4).
    Commonly wraps reasoning in <think>...</think> tags.
    """
    import re
    # Remove <think>...</think> blocks if present
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = cleaned.strip()

    if not cleaned and "</think>" in text:
        # Fallback: take content after last </think>
        cleaned = text.split("</think>")[-1].strip()
    
    return cleaned if cleaned else text


def query_ollama_stream(prompt: str, task: str = "generation"):
    """
    Stream response from Ollama (for interactive use).

    Yields:
        str: Response chunks
    """
    model = MODELS.get(task, MODELS.get("llm", "nemotron-mini:latest"))

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.15,
            "top_p": 0.9,
            "num_predict": 1024 if task == "generation" else 256,
        },
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate", json=payload, stream=True, timeout=300
        )
        response.raise_for_status()

        if "deepseek-r1" in model.lower():
            buffer = ""
            in_think_block = False
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        buffer += data["response"]
                        while True:
                            if in_think_block:
                                end_tag = "</think>"
                                end_index = buffer.find(end_tag)
                                if end_index != -1:
                                    buffer = buffer[end_index + len(end_tag) :]
                                    in_think_block = False
                                else:
                                    break
                            else:
                                start_tag = "<think>"
                                start_index = buffer.find(start_tag)
                                if start_index != -1:
                                    yield buffer[:start_index]
                                    buffer = buffer[start_index:]
                                    in_think_block = True
                                else:
                                    yield buffer
                                    buffer = ""
                                    break
            if buffer and not in_think_block:
                yield buffer
        else:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]

    except Exception as e:
        print(f"Error streaming from Ollama: {e}")
        yield "Error: Could not stream from LLM."


# Backward compatibility
def generate_response(prompt: str, model: str = None):
    """Legacy function - redirects to query_ollama."""
    if model:
        # Override with specific model
        return query_ollama(prompt, task="generation")
    return query_ollama(prompt, task="generation")
