import re
from typing import Generator, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from enterprice_rag.config.settings import OLLAMA_BASE_URL, MODELS


class OllamaLLM:
    """Thin wrapper around langchain-ollama's ChatOllama.

    - generate()  → blocking call, strips <think> blocks from reasoning models.
    - stream()    → token-level generator (used by graph.py callback handler).
    - get_chat_model() → returns a raw ChatOllama for use with LangChain callbacks.
    """

    # Temperature per task
    _TEMPS = {"query_rewrite": 0.5, "reflection": 0.1, "generation": 0.3}

    def _build(self, task: str, streaming: bool = False) -> ChatOllama:
        model = MODELS.get(task, MODELS.get("llm", "granite4.1:3b"))
        temp = self._TEMPS.get(task, 0.3)
        # Give reasoning models a higher token budget so they can finish thinking
        is_reasoning = "reasoning" in model.lower() or "r1" in model.lower()
        num_predict = 1024 if (task in ["generation", "context_generation"] or is_reasoning) else 128
        return ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=model,
            temperature=temp,
            num_predict=num_predict,
            streaming=streaming,
        )

    # ── Public API ──────────────────────────────────────────────────────────────

    def generate(self, prompt: str, task: str = "generation", **kwargs) -> str:
        """Blocking call — returns the full answer as a plain string."""
        llm = self._build(task, streaming=False)
        response = llm.invoke([HumanMessage(content=prompt)])
        return self._clean(response.content)

    def stream(self, prompt: str, task: str = "generation", **kwargs) -> Generator[str, None, None]:
        """Token-level generator — yields raw text chunks."""
        llm = self._build(task, streaming=True)
        for chunk in llm.stream([HumanMessage(content=prompt)]):
            if chunk.content:
                yield chunk.content

    def get_chat_model(self, task: str = "generation") -> ChatOllama:
        """Return a raw ChatOllama for use inside LangGraph nodes with callbacks."""
        return self._build(task, streaming=True)

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        """Strip <think>…</think> blocks emitted by reasoning models."""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if not cleaned and "</think>" in text:
            cleaned = text.split("</think>")[-1].strip()
        return cleaned if cleaned else text
