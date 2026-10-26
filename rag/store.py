# Imports
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import chromadb
from chromadb.config import Settings

from .chunker import Chunk
from .metadata import Metadata

# One result returned from a vector search
@dataclass(frozen=True)
class SearchHit:
    text: str
    score: float
    metadata: Metadata
    chunk_index: int

# Persistent ChromaDB-backed vector store
class VectorStore:
    def __init__(self, persist_dir: str | Path = "./chroma_db", collection_name: str = "rag",):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # Return the number of stored chunks
    def count(self) -> int:
        return self._collection.count()

    # Check whether a source file has already been stored
    def contains_source(self, source: str | Path) -> bool:
        result = self._collection.get(where={"source": str(Path(source))}, limit=1)
        return bool(result["ids"])

    # Return one metadata record for each stored vehicle manual
    def list_sources(self) -> list[Metadata]:
        result = self._collection.get(include=["metadatas"])
        sources: dict[str, Metadata] = {}
        for item in result["metadatas"] or []:
            metadata = {key: value for key, value in item.items()}
            document_id = str(metadata.get("document_id", ""))
            if document_id:
                sources[document_id] = metadata
        return list(sources.values())

    # Add chunks, embeddings, and metadata to the store
    def add(self, chunks: Iterable[Chunk], embeddings: list[list[float]], metadata: list[Metadata]) -> None:
        chunks = list(chunks)
        if len(chunks) != len(embeddings):
            raise ValueError(f"Got {len(chunks)} chunks but {len(embeddings)} embeddings")
        if len(chunks) != len(metadata):
            raise ValueError(f"Got {len(chunks)} chunks but {len(metadata)} metadata records")
        if not chunks:
            return

        ids = [f"{item['document_id']}::{chunk.index}" for chunk, item in zip(chunks, metadata)]
        documents = [chunk.text for chunk in chunks]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadata,
        )

    # Find the most similar chunks, optionally filtered by document metadata
    def search(self, query_embedding: list[float], top_k: int = 5, document_id: str | None = None, file_name: str | None = None,) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        if document_id is not None and file_name is not None:
            raise ValueError("document_id and file_name cannot both be provided")

        query: dict[str, object] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if document_id is not None:
            query["where"] = {"document_id": document_id}
        elif file_name is not None:
            query["where"] = {"file_name": file_name}

        result = self._collection.query(**query)
        hits: list[SearchHit] = []

        for document, distance, item in zip(result["documents"][0], result["distances"][0], result["metadatas"][0]):
            metadata = {key: value for key, value in item.items()}
            similarity = max(0.0, 1.0 - float(distance))
            hits.append(
                SearchHit(
                    text=document,
                    score=similarity,
                    metadata=metadata,
                    chunk_index=int(metadata.get("chunk_index", 0)),
                )
            )

        return hits
