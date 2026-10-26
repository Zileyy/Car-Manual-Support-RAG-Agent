# Imports
from __future__ import annotations
from pathlib import Path
import re
from typing import Iterable

from .chunker import chunk_text
from .embedder import Embedder, OllamaEmbedder
from .llm import LLM, OllamaLLM
from .loader import load_pdf
from .metadata import add_metadata
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .store import SearchHit, VectorStore

# Coordinate document ingestion, retrieval, and answer generation
class RAGPipeline:
    def __init__(self, store: VectorStore | None = None, embedder: Embedder | None = None, llm: LLM | None = None, system_prompt: str = SYSTEM_PROMPT, chunk_size: int = 800, chunk_overlap: int = 100):
        self.store = store or VectorStore()
        self.embedder = embedder or OllamaEmbedder()
        self.llm = llm or OllamaLLM()
        self.system_prompt = system_prompt
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # Load, chunk, embed, and store one PDF
    def ingest_file(self, source: str | Path) -> int:
        text = load_pdf(source)
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            raise ValueError(f"No text found in PDF: {source}")

        metadata = add_metadata(chunks, source)
        embeddings = self.embedder.embed_batch([chunk.text for chunk in chunks])
        self.store.add(chunks, embeddings, metadata)
        return len(chunks)

    # Ingest several PDF files and return the total number of chunks stored
    def ingest_files(self, sources: Iterable[str | Path]) -> int:
        return sum(self.ingest_file(source) for source in sources)

    # Retrieve relevant context and generate an answer
    def ask(self, question: str, top_k: int = 5) -> str:
        if not question.strip():
            raise ValueError("question must not be empty")
        if self.store.count() == 0:
            raise ValueError("The vector store is empty; ingest a PDF before asking a question")

        document_id = self._find_document_id(question)
        query_embedding = self.embedder.embed(question)
        hits = self.store.search(query_embedding, top_k, document_id=document_id)
        context = self._format_context(hits)
        user_prompt = build_user_prompt(question, context)
        return self.llm.complete(self.system_prompt, user_prompt)

    # Find the manual whose metadata is named in the user's question
    def _find_document_id(self, question: str) -> str:
        question_words = set(re.findall(r"[a-z0-9]+", question.lower()))
        matches: list[tuple[str, str]] = []
        available_models: list[str] = []

        for metadata in self.store.list_sources():
            document_id = str(metadata.get("document_id", ""))
            model = str(metadata.get("vehicle_model", metadata.get("file_name", "")))
            model_words = set(re.findall(r"[a-z0-9]+", model.lower()))
            if model:
                available_models.append(model)
            if document_id and model_words & question_words:
                matches.append((document_id, model))

        if not matches:
            models = ", ".join(sorted(set(available_models)))
            raise ValueError(f"Please specify a vehicle model, for example: 'How do I check the oil on a Seat Toledo?'. Available models: {models}")
        if len(matches) > 1:
            models = ", ".join(sorted(model for _, model in matches))
            raise ValueError(f"Please specify one vehicle model. Available models: {models}")

        return matches[0][0]

    # Format retrieved chunks with their source metadata
    def _format_context(self, hits: list[SearchHit]) -> str:
        if not hits:
            return "No relevant manual context was found."

        return "\n\n".join(
            f"Source: {hit.metadata.get('file_name', 'unknown')}\n{hit.text}"
            for hit in hits
        )
