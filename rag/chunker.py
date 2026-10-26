# Imports
from __future__ import annotations
import re
from dataclasses import dataclass

# Chunk data structure
@dataclass
class Chunk:
    text: str
    index: int            # ordinal position in the chunk stream
    char_start: int       # offset in the source document
    char_end: int         # offset in the source document

# Normalize text by collapsing whitespace and trimming
def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

# Split text into overlapping chunks
def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100,) -> list[Chunk]:

    # Validate parameters
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})")

    # Normalize the text and handle empty cases
    normalized = _normalize(text)
    if not normalized:
        return []

    # Calculate the step size and initialize variables for chunking
    step = chunk_size - chunk_overlap
    chunks: list[Chunk] = []
    i = 0
    idx = 0
    n = len(normalized)

    # Create chunks while respecting the overlap
    while i < n:
        piece = normalized[i : i + chunk_size]
        # Find the original-text char range for this normalized slice
        char_start = _normalized_to_original_offset(text, i)
        char_end = _normalized_to_original_offset(text, min(i + chunk_size, n))
        chunks.append(Chunk(text=piece, index=idx, char_start=char_start, char_end=char_end))
        idx += 1
        i += step

    return chunks

# Map an offset in the normalized text back to an offset in the original text
def _normalized_to_original_offset(text: str, normalized_offset: int) -> int:

    # Handle edge cases for offsets
    if normalized_offset <= 0:
        return 0
    
    if normalized_offset >= len(text):
        return len(text)
    
    out_i = 0
    norm_i = 0
    in_ws = False

    # Iterate through the original text to find the corresponding offset
    for ch in text:
        if ch.isspace():
            if not in_ws:
                # We just emitted a single space into the normalized stream
                if norm_i >= normalized_offset:
                    return out_i
                norm_i += 1
                in_ws = True
        else:
            if in_ws:
                in_ws = False
            if norm_i >= normalized_offset:
                return out_i
            norm_i += 1
        out_i += 1
    return out_i