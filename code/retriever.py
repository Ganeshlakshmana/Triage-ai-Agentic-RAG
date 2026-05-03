"""
retriever.py - Semantic Search over the Support Corpus

Responsibility:
    Given a support ticket, find the most relevant document
    chunks from Qdrant using semantic similarity search.

This is the R in RAG - Retrieval Augmented Generation.

Why retrieval matters:
    Without retrieval, the LLM answers from its own training data
    which may be outdated, wrong, or hallucinated.
    With retrieval, the LLM reads REAL support docs before answering.
    Grounded answers = trustworthy answers.

Flow:
    ticket text
        ->
    embed_query()       - convert ticket to vector
        ->
    vector_store.search()  - find similar chunks in Qdrant
        ->
    format_context()    - build clean text block for LLM
        ->
    Claude reads context + generates answer

Principle: Retrieval is the quality gate.
    Better retrieval = better answers, regardless of LLM quality.
"""

from embedder import embed_query
from vector_store import search
from config import TOP_K, VALID_COMPANIES


# Core Function: Retrieve
# 
def retrieve(
    query: str,
    company: str | None = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Find the most relevant support document chunks for a query.

    Args:
        query   : The support ticket text to search with
        company : Company name from the ticket (e.g. "HackerRank")
                  If provided, filters search to that company's docs.
                  If None or unrecognized, searches all companies.
        top_k   : Number of chunks to retrieve (default from config)

    Returns:
        List of dicts, each with:
        - text     : chunk content
        - company  : which company
        - category : document category
        - source   : original filename
        - score    : similarity score (0-1)

    Example:
        chunks = retrieve(
            query="how do I add extra time for a candidate?",
            company="HackerRank",
            top_k=5
        )
        # Returns 5 most relevant HackerRank support doc chunks
    """
    # Normalize company name for filtering
    # "HackerRank" -> "hackerrank", "Claude" -> "claude", etc.
    company_filter = None
    if company and company.strip().lower() != "none":
        normalized = company.strip().lower()
        # Only filter if it's a known company
        known = [c.lower() for c in VALID_COMPANIES]
        if normalized in known:
            company_filter = normalized

    # Step 1: Embed the query into a vector
    # Uses the same model as document embedding (all-MiniLM-L6-v2)
    # This ensures query and document vectors are in the same space
    query_vector = embed_query(query)

    # Step 2: Search Qdrant for similar vectors
    results = search(
        query_vector=query_vector,
        top_k=top_k,
        company_filter=company_filter,
    )

    return results


# Helper: Format Retrieved Chunks for LLM
# 
def format_context(chunks: list[dict]) -> str:
    """
    Convert retrieved chunks into a clean text block
    that the LLM can read and reference.

    Args:
        chunks : List of chunk dicts from retrieve()

    Returns:
        Formatted string to include in the LLM prompt

    Why format matters:
        The LLM needs to clearly understand:
        1. Which company each chunk belongs to
        2. What category/topic it covers
        3. Where it came from (source file)
        4. The actual content

        Clear formatting = LLM can cite and use the right sources.

    Example output:
        [Source 1] Company: hackerrank | Category: screen | Score: 0.8821
        File: 4811403281-adding-extra-time-for-candidates.md
        ---
        To add extra time: Go to Tests > Select test > Candidates tab...
    """
    if not chunks:
        return "No relevant documentation found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        part = (
            f"[Source {i}] "
            f"Company: {chunk['company']} | "
            f"Category: {chunk['category']} | "
            f"Relevance: {chunk['score']:.4f}\n"
            f"File: {chunk['source']}\n"
            f"{'-' * 40}\n"
            f"{chunk['text']}\n"
            f"{'-' * 40}"
        )
        parts.append(part)

    return "\n\n".join(parts)


# Helper: Build Full Query from Ticket Fields
# 
def build_query(issue: str, subject: str = "") -> str:
    """
    Combine ticket fields into a single search query.

    Args:
        issue   : Main ticket body
        subject : Ticket subject line (may be empty/noisy)

    Returns:
        Combined query string for embedding

    Why combine?
        Subject lines sometimes have useful keywords ("Card stolen")
        that don't appear in the issue body.
        Combining gives the embedding model more signal.

        But we put issue first - it's the authoritative content.
        Subject is supplementary.
    """
    subject = subject.strip() if subject else ""
    issue = issue.strip() if issue else ""

    if subject and subject.lower() not in issue.lower():
        # Only add subject if it adds new information
        return f"{issue}\n{subject}"
    return issue


# Quick Test
# 
if __name__ == "__main__":
    print("Testing retriever...\n")

    # Test 1: HackerRank specific query
    print("=" * 50)
    print("Test 1: HackerRank - extra time for candidates")
    print("=" * 50)
    query1 = "How do I add extra time accommodation for a candidate?"
    chunks1 = retrieve(query1, company="HackerRank", top_k=3)

    print(f"Query   : {query1}")
    print(f"Company : HackerRank (filtered)")
    print(f"Results : {len(chunks1)} chunks\n")
    for i, c in enumerate(chunks1, 1):
        print(f"  [{i}] score={c['score']:.4f} | {c['category']} | {c['source']}")
        print(f"       {c['text'][:100]}...")

    # Test 2: Claude specific query
    print("\n" + "=" * 50)
    print("Test 2: Claude - delete conversation")
    print("=" * 50)
    query2 = "How can I delete a conversation in Claude?"
    chunks2 = retrieve(query2, company="Claude", top_k=3)

    print(f"Query   : {query2}")
    print(f"Results : {len(chunks2)} chunks\n")
    for i, c in enumerate(chunks2, 1):
        print(f"  [{i}] score={c['score']:.4f} | {c['category']} | {c['source']}")
        print(f"       {c['text'][:100]}...")

    # Test 3: Visa query
    print("\n" + "=" * 50)
    print("Test 3: Visa - stolen card")
    print("=" * 50)
    query3 = "My Visa card was stolen, what should I do?"
    chunks3 = retrieve(query3, company="Visa", top_k=3)

    print(f"Query   : {query3}")
    print(f"Results : {len(chunks3)} chunks\n")
    for i, c in enumerate(chunks3, 1):
        print(f"  [{i}] score={c['score']:.4f} | {c['category']} | {c['source']}")
        print(f"       {c['text'][:100]}...")

    # Test 4: Show formatted context
    print("\n" + "=" * 50)
    print("Test 4: Formatted context (what LLM sees)")
    print("=" * 50)
    print(format_context(chunks1[:2]))

    print("\n[OK] Retriever is working!")