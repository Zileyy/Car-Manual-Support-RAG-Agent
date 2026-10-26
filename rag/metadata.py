# Imports
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
import hashlib

if TYPE_CHECKING:
    from .chunker import Chunk

# Metadata fields attached to each chunk before storage
Metadata = dict[str, str | int]

# Create a stable identifier for a source file
def _document_id(source: str | Path) -> str:
    source_path = Path(source)
    return hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()

# Create the vehicle model label used during retrieval
def _vehicle_model(source: str | Path) -> str:
    return Path(source).stem.replace("_", " ").replace("-", " ")

# Add source and location metadata to chunks
def add_metadata(chunks: list[Chunk], source: str | Path) -> list[Metadata]:
    source_path = Path(source)
    document_id = _document_id(source_path)
    file_name = source_path.name
    file_type = source_path.suffix.lower().lstrip(".")
    vehicle_model = _vehicle_model(source_path)

    return [
        {
            "document_id": document_id,
            "source": str(source_path),
            "file_name": file_name,
            "file_type": file_type,
            "vehicle_model": vehicle_model,
            "chunk_index": chunk.index,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
        }
        for chunk in chunks
    ]
