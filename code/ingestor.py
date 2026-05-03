"""
ingestor.py - Document Loader and Chunker

Responsibility:
    1. Walk through all .md files in data/
    2. Read each file's content
    3. Split content into overlapping chunks
    4. Attach metadata to each chunk (company, category, source file)
    5. Return a list of chunks ready for embedding

Why this file exists:
    The vector database stores chunks, not whole documents.
    This file is the first step of the RAG pipeline - preparing
    raw documents into a format suitable for embedding and retrieval.

Principle: Single Responsibility - this file ONLY handles
           loading and chunking. It does NOT embed or store.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Generator
from tqdm import tqdm

from config import (
    HACKERRANK_DIR,
    CLAUDE_DIR,
    VISA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# Data Structure: DocumentChunk
@dataclass
class DocumentChunk:
    """
    Represents one piece of a support document.

    Why a dataclass?
        A plain dict works, but a dataclass gives us:
        - Type hints (self-documenting)
        - Auto-generated __repr__ for easy debugging
        - Structured, predictable shape

    Fields:
        text     : The actual content of this chunk
        company  : "hackerrank", "claude", or "visa"
        category : The subfolder name (e.g. "screen", "privacy")
        source   : The original filename (for traceability)
        chunk_id : A unique identifier for this chunk
    """
    text: str
    company: str
    category: str
    source: str
    chunk_id: str = ""

    def __post_init__(self):
        """
        __post_init__ runs after __init__ automatically.
        We use it to build the chunk_id from other fields
        so it's always consistent.
        """
        if not self.chunk_id:
            # chunk_id format: "hackerrank::screen::filename::0"
            # The last number is the chunk index within the file.
            # We'll set this properly in the chunking step.
            self.chunk_id = f"{self.company}::{self.category}::{self.source}"

    def to_metadata(self) -> dict:
        """
        Convert to a flat dict for storing in Qdrant.

        Why? Qdrant stores metadata as a flat key-value payload.
        This method gives us a clean way to serialize.
        """
        return {
            "company": self.company,
            "category": self.category,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "text": self.text,  # store text in metadata for retrieval
        }


# Core Function: Chunk Text
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split a long text into overlapping chunks.

    Args:
        text       : The full document text
        chunk_size : Max characters per chunk
        overlap    : How many characters consecutive chunks share

    Returns:
        List of text chunks

    Example with chunk_size=20, overlap=5:
        text = "Hello world this is a test of chunking"
        chunks = [
            "Hello world this is ",   # chars 0-19
            "s is a test of chunk",   # chars 15-34  (overlaps by 5)
            "f chunking",             # chars 29-end
        ]

    Why character-based and not word/sentence-based?
        Simpler and predictable. Word-based chunking is better
        (we could use nltk/spacy) but character-based is sufficient
        for this project and easier to understand.
    """
    # Clean up the text first
    # Strip leading/trailing whitespace
    # Collapse multiple consecutive newlines into two
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If we're not at the end, try to break at a natural boundary
        # (newline or space) rather than mid-word
        if end < len(text):
            # Look backwards from 'end' for a newline or space
            break_point = text.rfind("\n", start, end)
            if break_point == -1:
                break_point = text.rfind(" ", start, end)
            if break_point != -1 and break_point > start:
                end = break_point

        chunk = text[start:end].strip()
        if chunk:  # only add non-empty chunks
            chunks.append(chunk)

        # Move start forward by (chunk_size - overlap)
        # This creates the overlap between consecutive chunks
        start += chunk_size - overlap

    return chunks


# Core Function: Infer Category from Path
def infer_category(file_path: Path, company_dir: Path) -> str:
    """
    Extract the category (subfolder name) from a file path.

    Args:
        file_path   : Full path to the .md file
        company_dir : Root directory of this company's docs

    Returns:
        Category string, e.g. "screen", "privacy", "billing"

    Example:
        file_path   = data/hackerrank/screen/managing-tests/test.md
        company_dir = data/hackerrank
        relative    = screen/managing-tests/test.md
        category    = "screen"  (first subfolder)

    Why extract category?
        Category acts as a "topic label" stored in metadata.
        It helps with filtering: "only search billing-related docs"
    """
    try:
        relative = file_path.relative_to(company_dir)
        parts = relative.parts
        # parts[0] is the first subfolder = the category
        # parts[-1] is the filename
        if len(parts) > 1:
            return parts[0]
        else:
            return "general"
    except ValueError:
        return "general"


# Core Function: Load Documents from One Company
def load_company_docs(company_name: str, company_dir: Path) -> Generator[DocumentChunk, None, None]:
    """
    Walk a company's directory, read all .md files,
    chunk them, and yield DocumentChunk objects.

    Args:
        company_name : "hackerrank", "claude", or "visa"
        company_dir  : Path to that company's data folder

    Yields:
        DocumentChunk objects one at a time

    Why a Generator (yield) instead of returning a list?
        With hundreds of files, building a full list in memory
        before processing wastes RAM. A generator yields one
        chunk at a time - memory efficient.

        Generator principle: "lazy evaluation" - compute only
        what's needed, when it's needed.
    """
    if not company_dir.exists():
        print(f" Directory not found, skipping: {company_dir}")
        return

    # rglob("*.md") recursively finds ALL .md files in all subfolders
    md_files = list(company_dir.rglob("*.md"))

    if not md_files:
        print(f"No .md files found in: {company_dir}")
        return

    for file_path in md_files:
        try:
            # Read the file content
            # encoding="utf-8" handles special characters (like em dashes, accents)
            # errors="ignore" skips unreadable bytes instead of crashing
            text = file_path.read_text(encoding="utf-8", errors="ignore")

            if not text.strip():
                continue  # skip empty files

            # Infer the category from the folder structure
            category = infer_category(file_path, company_dir)

            # Get just the filename (without full path) for the source field
            source = file_path.name

            # Split this document into chunks
            chunks = chunk_text(text)

            # Yield each chunk as a DocumentChunk
            for i, chunk_text_content in enumerate(chunks):
                chunk = DocumentChunk(
                    text=chunk_text_content,
                    company=company_name,
                    category=category,
                    source=source,
                    # chunk_id uniquely identifies this specific chunk
                    # format: "company::category::filename::chunk_index"
                    chunk_id=f"{company_name}::{category}::{source}::{i}",
                )
                yield chunk

        except Exception as e:
            # Don't crash the whole ingestion if one file fails
            # Log the error and continue with the next file
            print(f"Failed to process {file_path}: {e}")
            continue


# Main Function: Load All Documents
def load_all_documents() -> list[DocumentChunk]:
    """
    Load and chunk documents from all three companies.

    Returns:
        List of all DocumentChunk objects across all companies.

    This is the main entry point called by the ingestion pipeline.
    """
    print("Loading documents from corpus...")

    # Map company name -> directory
    companies = {
        "hackerrank": HACKERRANK_DIR,
        "claude": CLAUDE_DIR,
        "visa": VISA_DIR,
    }

    all_chunks: list[DocumentChunk] = []

    for company_name, company_dir in companies.items():
        print(f"\n Processing {company_name}...")

        # Collect all chunks for this company
        # tqdm wraps any iterable and shows a progress bar
        company_chunks = list(
            tqdm(
                load_company_docs(company_name, company_dir),
                desc=f"    Chunking {company_name}",
                unit="chunk",
            )
        )

        print(f" {len(company_chunks)} chunks created")
        all_chunks.extend(company_chunks)

    print(f"\n Total chunks across all companies: {len(all_chunks)}")
    return all_chunks


# Stats Helper
def print_corpus_stats(chunks: list[DocumentChunk]) -> None:
    """
    Print a summary of what was loaded.
    Useful for verifying ingestion before embedding.
    """
    from collections import Counter

    company_counts = Counter(c.company for c in chunks)
    category_counts = Counter(f"{c.company}/{c.category}" for c in chunks)

    print("\n Corpus Statistics:")
    print("  By company:")
    for company, count in sorted(company_counts.items()):
        print(f"    {company:<15} {count:>5} chunks")

    print("\n  Top 10 categories:")
    for cat, count in category_counts.most_common(10):
        print(f"    {cat:<40} {count:>4} chunks")


# Quick Test
if __name__ == "__main__":
    """
    Run directly to test: python ingestor.py
    This loads all documents and prints statistics.
    Does NOT embed or store - just chunking.
    """
    chunks = load_all_documents()
    print_corpus_stats(chunks)

    # Show a sample chunk so you can verify the output
    if chunks:
        print("\n Sample chunk:")
        sample = chunks[0]
        print(f"  company  : {sample.company}")
        print(f"  category : {sample.category}")
        print(f"  source   : {sample.source}")
        print(f"  chunk_id : {sample.chunk_id}")
        print(f"  text     :\n    {sample.text[:200]}...")