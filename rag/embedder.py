# Imports
from __future__ import annotations
from typing import Protocol
import httpx

# Embedder interface and Ollama-backed implementation
class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimension(self) -> int: ...

# Ollama-backed embedder class
class OllamaEmbedder:
    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_BASE_URL = "http://localhost:11434"
    KNOWN_DIMENSIONS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
    }

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL, timeout: float = 60.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass

    def dimension(self) -> int:
        return self.KNOWN_DIMENSIONS.get(self.model, 768)

    # Embed a single text string into a vector
    def embed(self, text: str) -> list[float]:
        response = self._client.post(
            "/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]

    # Embed a batch of text strings into vectors
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]