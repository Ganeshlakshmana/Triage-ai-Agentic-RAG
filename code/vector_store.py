"""
vector_store.py - Qdrant Vector Database Interface

Responsibility:
    1. Create and manage the Qdrant collection
    2. Store embedded document chunks (insert vectors + metadata)
    3. Search for similar chunks given a query vector
    4. Provide utility functions (collection info, delete, etc.)

Why this file exists:
    Qdrant is our "smart database" that understands meaning, not
    just exact keywords. This file is the single interface between
    our Python code and the Qdrant server running in Docker.

    All other files talk to Qdrant ONLY through this module.
    This is the "Repository Pattern" - a clean abstraction over
    the database layer.

Principle: Abstraction - if we ever switch from Qdrant to Pinecone
           or Weaviate, we only change THIS file, nothing else.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from tqdm import tqdm

from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    EMBEDDING_DIMENSION,
)


# Initialize Qdrant Client
# QdrantClient connects to our Docker-running Qdrant instance.
# host + port matches what we set up in Docker:
#   docker run -p 6333:6333 qdrant/qdrant
#
# In production this would be:
#   client = QdrantClient(url="https://your-cluster.qdrant.io", api_key="...")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# Function: Create Collection
def create_collection(force_recreate: bool = False) -> None:
    """
    Create the Qdrant collection if it doesn't exist.

    Args:
        force_recreate : If True, delete existing collection and
                         recreate it. Use when re-ingesting from scratch.

    What is a collection?
        Like a table in SQL. Holds all vectors of the same dimension.
        You define it once with:
        - Vector size: must match embedding dimensions (3072)
        - Distance metric: how to measure similarity between vectors

    Distance metrics explained:
        COSINE   : Measures angle between vectors. Best for text.
                   Score range: 0 (opposite) to 1 (identical)
        DOT      : Raw dot product. Faster but less normalized.
        EUCLID   : Euclidean distance. Better for images/audio.

        We use COSINE because it's the standard for text embeddings.
        It's scale-invariant - a short sentence and a long sentence
        about the same topic will still score similarly.
    """
    # Check if collection already exists
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        if force_recreate:
            print(f"Deleting existing collection '{COLLECTION_NAME}'...")
            client.delete_collection(COLLECTION_NAME)
            print("   Deleted.")
        else:
            print(f"[OK] Collection '{COLLECTION_NAME}' already exists. Skipping creation.")
            return

    print(f"Creating collection '{COLLECTION_NAME}'...")
    print(f"   Vector size : {EMBEDDING_DIMENSION}")
    print(f"   Distance    : Cosine")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIMENSION,   # must match embedding model output
            distance=Distance.COSINE,   # similarity metric
        ),
    )
    print(f"[OK] Collection '{COLLECTION_NAME}' created successfully.")


# Function: Store Chunks
def store_chunks(chunks: list, embeddings: list[list[float]]) -> None:
    """
    Store document chunks and their vectors into Qdrant.

    Args:
        chunks     : List of DocumentChunk objects (from ingestor.py)
        embeddings : List of vectors (from embedder.py)
                     Must be same length and order as chunks.

    What is a PointStruct?
        Qdrant calls each stored item a "point". A point has:
        - id      : unique integer identifier
        - vector  : the embedding (list of 3072 floats)
        - payload : metadata dict (company, category, text, etc.)

    Why store text in the payload?
        When Qdrant returns search results, we need the actual text
        to send to the LLM. Storing it in payload means one round trip -
        search returns both the vector match AND the text we need.

    Batching:
        Uploading 13,244 points one-by-one would be slow.
        We batch them in groups of 100 for efficient uploading.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings. "
            "They must be the same length and in the same order."
        )

    UPLOAD_BATCH_SIZE = 100
    total_batches = len(chunks) // UPLOAD_BATCH_SIZE + 1

    print(f"\nUploading {len(chunks)} points to Qdrant...")
    print(f"   Upload batch size : {UPLOAD_BATCH_SIZE}")
    print(f"   Total batches     : {total_batches}")

    # Split into upload batches
    for batch_start in tqdm(
        range(0, len(chunks), UPLOAD_BATCH_SIZE),
        desc="  Uploading to Qdrant",
        unit="batch",
    ):
        batch_end = min(batch_start + UPLOAD_BATCH_SIZE, len(chunks))
        batch_chunks = chunks[batch_start:batch_end]
        batch_embeddings = embeddings[batch_start:batch_end]

        # Build PointStruct list for this batch
        points = []
        for i, (chunk, vector) in enumerate(zip(batch_chunks, batch_embeddings)):
            point = PointStruct(
                # id must be a unique integer
                # We use global index: batch_start + i
                id=batch_start + i,

                # The actual vector (3072 floats)
                vector=vector,

                # Payload = metadata stored alongside the vector
                # This is what we get back when searching
                payload=chunk.to_metadata(),
            )
            points.append(point)

        # Upload this batch to Qdrant
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )
        # upsert = "insert or update"
        # If a point with the same id exists, it updates it.
        # If not, it inserts. Idempotent - safe to run multiple times.

    print(f"[OK] Successfully uploaded {len(chunks)} points to Qdrant.")


# Function: Search
def search(
    query_vector: list[float],
    top_k: int = 5,
    company_filter: str | None = None,
) -> list[dict]:
    """
    Find the most semantically similar chunks to a query vector.

    Args:
        query_vector   : Embedded support ticket (from embedder.embed_query)
        top_k          : Number of results to return (default 5)
        company_filter : If set, only search within this company's docs.
                         E.g. "hackerrank" -> only HackerRank chunks.
                         If None -> search across all companies.

    Returns:
        List of dicts, each containing:
        - text     : The chunk content
        - company  : Which company this chunk belongs to
        - category : The document category
        - source   : Original filename
        - score    : Similarity score (0-1, higher = more similar)

    Why filter by company?
        If the ticket says company="HackerRank", searching only
        HackerRank docs gives more precise results and avoids
        irrelevant Visa/Claude content contaminating the response.

        If company="None" or unknown, we search everything.

    How Qdrant filtering works:
        Qdrant can filter by payload fields before doing vector search.
        It's like SQL WHERE but applied to the metadata.
        "Find the 5 most similar vectors WHERE company = 'hackerrank'"
    """
    # Build optional company filter
    search_filter = None
    if company_filter:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="company",           # payload field name
                    match=MatchValue(value=company_filter.lower()),
                )
            ]
        )

    # Execute the search
    try:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            query_filter=search_filter,
            with_payload=True,
        ).points
    except Exception as e:
        import sys
        print(f"\n[CRITICAL ERROR] Failed to query Qdrant: {e}", file=sys.stderr)
        print("[HELP] Make sure Qdrant is running in Docker: docker run -d -p 6333:6333 qdrant/qdrant", file=sys.stderr)
        raise RuntimeError(f"Qdrant connection failed: {e}") from e

    # Format results into clean dicts
    formatted = []
    for hit in results:
        formatted.append({
            "text": hit.payload.get("text", ""),
            "company": hit.payload.get("company", ""),
            "category": hit.payload.get("category", ""),
            "source": hit.payload.get("source", ""),
            "chunk_id": hit.payload.get("chunk_id", ""),
            "score": round(hit.score, 4),
            # score is cosine similarity: 1.0 = identical, 0.0 = unrelated
        })

    return formatted


# Utility: Collection Info
def get_collection_info() -> dict:
    """
    Get stats about the current collection.
    Useful for verifying ingestion completed correctly.
    """
    try:
        info = client.get_collection(COLLECTION_NAME)
        return {
            "name": COLLECTION_NAME,
            "points_count": info.points_count,
            "status": str(info.status),
        }
    except Exception:
        return {"error": f"Collection '{COLLECTION_NAME}' not found."}


def collection_exists() -> bool:
    """Check if our collection exists in Qdrant."""
    existing = [c.name for c in client.get_collections().collections]
    return COLLECTION_NAME in existing


def is_collection_populated() -> bool:
    """
    Check if collection has vectors stored.
    Used in main.py to skip re-ingestion if already done.
    """
    if not collection_exists():
        return False
    info = client.get_collection(COLLECTION_NAME)
    return (info.points_count or 0) > 0


# Quick Test
if __name__ == "__main__":
    """
    Run directly to test: python vector_store.py
    Tests connection to Qdrant and collection operations.
    Does NOT ingest real data - just verifies connectivity.
    """
    print("Testing vector_store.py...")

    # Test 1: Connection
    print("\n[1] Testing Qdrant connection...")
    try:
        collections = client.get_collections()
        print(f"   [OK] Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        print(f"   Existing collections: {[c.name for c in collections.collections]}")
    except Exception as e:
        print(f"   [ERROR] Connection failed: {e}")
        exit(1)

    # Test 2: Collection info
    print("\n[2] Collection info...")
    info = get_collection_info()
    if "error" in info:
        print(f"   [INFO] {info['error']} (Run ingest.py first to populate it)")
    else:
        print(f"   [OK] Collection '{info['name']}'")
        print(f"      Points  : {info['points_count']}")
        print(f"      Status  : {info['status']}")

    # Test 3: Is populated?
    print("\n[3] Is collection populated?")
    populated = is_collection_populated()
    print(f"   {'[OK] Yes' if populated else '[WARNING] No - run ingest.py first'}")

    print("\n[OK] vector_store.py is ready!")