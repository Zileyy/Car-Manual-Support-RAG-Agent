# Imports
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Sequence

from .pipeline import RAGPipeline
from .store import VectorStore

# Build the command-line argument parser
def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask questions about vehicle manuals.")
    parser.add_argument("--persist-dir", default="./chroma_db", help="Directory used to persist the vector store.")
    parser.add_argument("--collection-name", default="rag", help="ChromaDB collection name.")
    parser.add_argument("--data-dir", default="./data", type=Path, help="Directory containing vehicle manuals.")

    commands = parser.add_subparsers(dest="command", required=True)

    ingest_parser = commands.add_parser("ingest", help="Load one or more PDF manuals.")
    ingest_parser.add_argument("sources", nargs="+", type=Path, help="PDF files to ingest.")

    ask_parser = commands.add_parser("ask", help="Ask a question using the stored manuals.")
    ask_parser.add_argument("question", help="Question to ask.")
    ask_parser.add_argument("--top-k", default=5, type=positive_integer, help="Number of chunks to retrieve.")

    return parser

# Parse a positive integer command-line option
def positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number

# Create a pipeline using the shared CLI configuration
def create_pipeline(arguments: argparse.Namespace) -> RAGPipeline:
    store = VectorStore(persist_dir=arguments.persist_dir, collection_name=arguments.collection_name)
    return RAGPipeline(store=store)

# Ingest PDF manuals that are not already present in the store
def ensure_documents_ingested(pipeline: RAGPipeline, data_dir: Path) -> int:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {data_dir}")

    sources = sorted(data_dir.glob("*.pdf"))
    if not sources:
        raise FileNotFoundError(f"No PDF manuals found in data directory: {data_dir}")

    missing_sources = [source for source in sources if not pipeline.store.contains_source(source)]
    return pipeline.ingest_files(missing_sources) if missing_sources else 0

# Ingest the PDF files supplied by the user
def run_ingest(arguments: argparse.Namespace) -> int:
    pipeline = create_pipeline(arguments)
    chunk_count = pipeline.ingest_files(arguments.sources)
    print(f"Ingested {len(arguments.sources)} file(s) and stored {chunk_count} chunk(s).")
    return 0

# Ask a question using the stored vehicle manuals
def run_ask(arguments: argparse.Namespace) -> int:
    pipeline = create_pipeline(arguments)
    indexed_chunks = ensure_documents_ingested(pipeline, arguments.data_dir)
    if indexed_chunks:
        print(f"Prepared {indexed_chunks} manual chunk(s).")

    answer = pipeline.ask(arguments.question, top_k=arguments.top_k)
    print(answer)
    return 0

# Run the selected command and report expected application errors
def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    arguments = parser.parse_args(argv)

    try:
        if arguments.command == "ingest":
            return run_ingest(arguments)
        return run_ask(arguments)
    except (FileNotFoundError, OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

# Allow execution with python -m rag.cli
if __name__ == "__main__":
    raise SystemExit(main())
