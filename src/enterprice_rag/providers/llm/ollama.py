import requests
import json
import re
from typing import Generator, Optional
from enterprice_rag.core.interfaces import LLMProvider
from enterprice_rag.config.settings import OLLAMA_BASE_URL, MODELS

class OllamaLLM(LLMProvider):
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url

    def generate(self, prompt: str, task: str = "generation", temperature: Optional[float] = None, **kwargs) -> str:
        model = MODELS.get(task, MODELS.get("llm", "nemotron-mini:latest"))
        
        # Task-specific temperature overrides
        temp_map = {
            "query_rewrite": 0.5,
            "reflection": 0.1,
            "generation": 0.3,
        }
        if temperature is None:
            temperature = temp_map.get(task, 0.3)

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
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            answer = result.get("response", "").strip()
            return self._extract_reasoning_answer(answer)
        except Exception as e:
            print(f"Error querying Ollama ({model}): {e}")
            return f"Error: {e}"

    def stream(self, prompt: str, task: str = "generation", **kwargs) -> Generator[str, None, None]:
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
            response = requests.post(f"{self.base_url}/api/generate", json=payload, stream=True, timeout=300)
            response.raise_for_status()

            if "deepseek-r1" in model.lower():
                # Handle deepseek reasoning blocks
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
            yield f"Error: {e}"

    def _extract_reasoning_answer(self, text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if not cleaned and "</think>" in text:
            cleaned = text.split("</think>")[-1].strip()
        return cleaned if cleaned else text
