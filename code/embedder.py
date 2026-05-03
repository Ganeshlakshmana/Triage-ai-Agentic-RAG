"""
embedder.py - Local Text Embedding using sentence-transformers

Why local embeddings instead of API?
    - No rate limits (runs on your machine)
    - No API cost
    - No network dependency
    - Fast - GPU if available, CPU otherwise
    - Perfect for ingestion of large document sets

Model: all-MiniLM-L6-v2
    - Downloads once (~90MB), cached forever after
    - Produces 384-dimensional vectors
    - Great quality for semantic search
    - Used by many production RAG systems

Principle: Right tool for the job.
    Embeddings = compute-heavy, benefits from local execution.
    Generation = reasoning-heavy, benefits from large cloud LLM.
"""

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import EMBEDDING_MODEL, EMBEDDING_DIMENSION

# Load the model - happens ONCE at import time
# First run: downloads ~90MB model from HuggingFace, caches it.
# Subsequent runs: loads from cache instantly (~1 second).
# The model runs entirely on your machine - no API calls.
print(f"Loading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)
print(f"[OK] Embedding model loaded ({EMBEDDING_DIMENSION} dimensions)")

BATCH_SIZE = 64
# sentence-transformers handles batching internally.
# 64 is efficient for CPU. Larger batch = faster but more RAM.


# Core Function: Embed a Single Text
def embed_text(text: str) -> list[float]:
    """
    Convert a single text into a vector.

    Args:
        text : Any text string

    Returns:
        List of 384 floats representing the text's meaning

    Note: No task_type needed for sentence-transformers.
    The model handles documents and queries the same way.
    This simplifies our API vs the Gemini approach.
    """
    vector = model.encode(text, normalize_embeddings=True)
    # normalize_embeddings=True makes vectors unit length (magnitude=1)
    # This is required for cosine similarity to work correctly in Qdrant
    return vector.tolist()


# Core Function: Embed a Batch
def embed_batch_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple texts efficiently in one call.

    sentence-transformers handles batching internally -
    it parallelizes computation across all texts at once.
    Much faster than calling embed_text() in a loop.
    """
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


# Main Function: Embed All Chunks
def embed_chunks(chunks: list) -> list[list[float]]:
    """
    Embed all document chunks for ingestion into Qdrant.

    Args:
        chunks : List of DocumentChunk objects from ingestor.py

    Returns:
        List of 384-dim vectors, same order as input chunks.

    Why so much faster than before?
        No API calls = no network latency, no rate limits.
        sentence-transformers processes all chunks locally
        using optimized matrix operations.
        13,244 chunks takes ~30-60 seconds on CPU.
    """
    print(f"\nEmbedding {len(chunks)} chunks locally...")
    print(f"   Model     : {EMBEDDING_MODEL}")
    print(f"   Dimensions: {EMBEDDING_DIMENSION}")
    print(f"   Batch size: {BATCH_SIZE}")

    texts = [chunk.text for chunk in chunks]

    # encode() with show_progress_bar=True shows a tqdm bar
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = vectors.tolist()
    print(f"[OK] Embedded {len(embeddings)} chunks")
    return embeddings


# Query Embedding (used at search time)
def embed_query(query: str) -> list[float]:
    """
    Embed a support ticket query for similarity search.
    Same model as documents - no task_type distinction needed.
    """
    return embed_text(query)


# Quick Test
if __name__ == "__main__":
    print("\nTesting embedder...")
    print(f"   Model     : {EMBEDDING_MODEL}")
    print(f"   Dimensions: {EMBEDDING_DIMENSION}")

    test_doc = "Tests in HackerRank remain active indefinitely unless a start and end time are set."
    doc_vector = embed_text(test_doc)
    print(f"\n[OK] Document embedding:")
    print(f"   Input  : '{test_doc[:60]}...'")
    print(f"   Output : [{doc_vector[0]:.4f}, {doc_vector[1]:.4f}, ... {doc_vector[-1]:.4f}]")
    print(f"   Length : {len(doc_vector)} dimensions")

    test_query = "how long do tests stay active?"
    query_vector = embed_query(test_query)
    print(f"\n[OK] Query embedding:")
    print(f"   Input  : '{test_query}'")
    print(f"   Output : [{query_vector[0]:.4f}, {query_vector[1]:.4f}, ... {query_vector[-1]:.4f}]")
    print(f"   Length : {len(query_vector)} dimensions")

    assert len(doc_vector) == EMBEDDING_DIMENSION
    assert len(query_vector) == EMBEDDING_DIMENSION

    # Check semantic similarity - similar texts should have high dot product
    dot = sum(a * b for a, b in zip(doc_vector, query_vector))
    print(f"\nSemantic similarity score: {dot:.4f}")
    print(f"   (1.0 = identical, 0.0 = unrelated)")
    print(f"   These should be fairly similar -> score should be > 0.3")

    print(f"\n[OK] All tests passed! Embedder is ready.")