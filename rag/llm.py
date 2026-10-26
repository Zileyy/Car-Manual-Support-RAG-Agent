# Imports
from __future__ import annotations
from typing import Protocol
import httpx

# Embedder interface and Ollama-backed implementation
class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...

# Ollama-backed LLM class
class OllamaLLM:
    DEFAULT_MODEL = "deepseek-r1"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL, timeout: float = 120.0,):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass

    # Complete a chat interaction with system and user messages
    def complete(self, system: str, user: str) -> str:
        response = self._client.post(
            "/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]