# Imports
from __future__ import annotations
from pathlib import Path
from pypdf import PdfReader

# Load text from a PDF file, returning the concatenated text of all pages
def load_pdf(path: str | Path) -> str:
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    chunks: list[str] = []

    # Extract text from each page, skipping empty pages
    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            chunks.append(text)

    return "\n\n".join(chunks)