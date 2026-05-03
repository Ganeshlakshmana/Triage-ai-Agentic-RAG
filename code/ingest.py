"""
ingest.py - Ingestion Pipeline

Now much simpler because:
- Local embeddings = no rate limits = no complex batching logic
- embed_chunks() handles everything internally
- store_chunks() handles Qdrant upload

Usage:
    python ingest.py           # Skip if already populated
    python ingest.py --force   # Rebuild from scratch
"""

import sys
import time

from config import validate_config, COLLECTION_NAME
from ingestor import load_all_documents, print_corpus_stats
from embedder import embed_chunks
from vector_store import (
    create_collection,
    store_chunks,
    get_collection_info,
    is_collection_populated,
    client,
)


def run_ingestion(force_recreate: bool = False) -> None:
    print("=" * 60)
    print("  HackerRank Orchestrate - Ingestion Pipeline")
    print("=" * 60)

    #  Step 0: Validate config 
    print("\nValidating configuration...")
    validate_config()
    print("   [OK] Config OK")

    #  Step 1: Check if already done 
    if not force_recreate and is_collection_populated():
        print("\n[OK] Already populated. Use --force to rebuild.")
        info = get_collection_info()
        print(f"   Points: {info.get('points_count')}")
        return

    #  Step 2: Load & chunk documents 
    print("\nStep 1: Loading and chunking documents...")
    start = time.time()
    chunks = load_all_documents()
    print_corpus_stats(chunks)
    print(f"   Loaded in {time.time() - start:.1f}s")

    #  Step 3: Create Qdrant collection 
    print("\nStep 2: Setting up Qdrant collection...")
    create_collection(force_recreate=force_recreate)

    #  Step 4: Embed all chunks locally 
    # No API calls - runs on your machine
    # 13,244 chunks takes ~30-60 seconds on CPU
    print("\nStep 3: Embedding chunks locally (no rate limits!)...")
    start = time.time()
    embeddings = embed_chunks(chunks)
    print(f"   Embedded in {time.time() - start:.1f}s")

    #  Step 5: Store in Qdrant 
    print("\nStep 4: Storing vectors in Qdrant...")
    start = time.time()
    store_chunks(chunks, embeddings)
    print(f"   Stored in {time.time() - start:.1f}s")

    #  Step 6: Verify 
    print("\nVerifying...")
    info = get_collection_info()
    print(f"   Points in Qdrant : {info.get('points_count')}")
    print(f"   Status           : {info.get('status')}")

    if info.get("points_count", 0) == len(chunks):
        print(f"\n[OK] Ingestion complete! {len(chunks)} chunks in Qdrant.")
    else:
        print(f"\n[WARNING] Mismatch: expected {len(chunks)}, got {info.get('points_count')}")

    print("\nReady! Run: python main.py")


if __name__ == "__main__":
    force = "--force" in sys.argv
    if force:
        confirm = input("[WARNING] --force deletes all vectors. Continue? (y/n): ")
        if confirm.lower() != "y":
            print("Cancelled.")
            sys.exit(0)
    run_ingestion(force_recreate=force)