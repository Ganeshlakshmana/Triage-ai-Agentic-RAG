"""
config.py - Single Source of Truth for all settings.

Why this file exists:
    Instead of scattering constants across files, we centralize them here.
    If a model name changes or a path moves, you change it in ONE place.

Principle: Single Source of Truth
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file so secrets stay out of code
load_dotenv()


# API Keys
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
# Used for Claude generation (claude-sonnet-4-5)


# Embedding Model - runs LOCALLY, no API needed
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
# sentence-transformers model that runs on your machine.
# No API calls, no rate limits, completely free.
# Downloads once (~90MB), then cached locally forever.
# Produces 384-dimensional vectors.

EMBEDDING_DIMENSION: int = 384
# all-MiniLM-L6-v2 always produces 384-dim vectors.
# Must match what we tell Qdrant when creating the collection.


# Generation Model - Claude API
GENERATION_MODEL: str = "claude-sonnet-4-5"
# Used to read retrieved chunks + ticket and generate response.
# Fast, capable, cost-efficient for support ticket use case.


# Qdrant Vector Database Connection
QDRANT_HOST: str = "localhost"
QDRANT_PORT: int = 6333

COLLECTION_NAME: str = "support_docs"
# A "collection" in Qdrant is like a table in SQL.
# All embedded support documents live here.


# Document Chunking Settings
CHUNK_SIZE: int = 500
# Each chunk will be ~500 characters long (~100 words).

CHUNK_OVERLAP: int = 50
# Consecutive chunks share 50 characters to preserve context
# at chunk boundaries.


# Retrieval Settings
TOP_K: int = 7
# How many chunks to retrieve from Qdrant per ticket.
# Increased from 5 to 7 - gives Claude more context to work with,
# especially for small corpora like Visa where relevant info
# may be spread across multiple chunks.


# File System Paths
PROJECT_ROOT: Path = Path(__file__).parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
HACKERRANK_DIR: Path = DATA_DIR / "hackerrank"
CLAUDE_DIR: Path = DATA_DIR / "claude"
VISA_DIR: Path = DATA_DIR / "visa"

SUPPORT_TICKETS_DIR: Path = PROJECT_ROOT / "support_tickets"
INPUT_CSV: Path = SUPPORT_TICKETS_DIR / "support_tickets.csv"
OUTPUT_CSV: Path = SUPPORT_TICKETS_DIR / "output.csv"
SAMPLE_CSV: Path = SUPPORT_TICKETS_DIR / "sample_support_tickets.csv"


# Company -> Directory Mapping
COMPANY_DIRS: dict = {
    "hackerrank": HACKERRANK_DIR,
    "claude": CLAUDE_DIR,
    "visa": VISA_DIR,
}

VALID_COMPANIES: list = ["HackerRank", "Claude", "Visa"]


# Output Schema - Allowed Values
VALID_STATUSES: list = ["replied", "escalated"]
VALID_REQUEST_TYPES: list = ["product_issue", "feature_request", "bug", "invalid"]


# Escalation Keywords - Hard Rules
# These ALWAYS trigger escalation regardless of LLM decision.
# Principle: For high-risk situations, rules beat AI.
ESCALATION_KEYWORDS: list = [
    "fraud",
    "hacked",
    "unauthorized",
    "site is down",
    "breach",
    "legal",
    "lawsuit",
    "data leak",
    "locked out",
    "identity theft",
    # Note: "stolen" removed - too broad.
    # "stolen card" has documented Visa guidance -> should reply.
    # "stolen credentials/password" caught by HIGH_RISK_PATTERNS.
]


# Startup Validation
def validate_config() -> None:
    """
    Call at startup to catch misconfigurations early.
    Principle: Fail fast - surface errors at startup, not mid-run.
    """
    errors = []

    if not ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY is not set. Add it to your .env file.")

    if not DATA_DIR.exists():
        errors.append(f"Data directory not found: {DATA_DIR}")

    if not INPUT_CSV.exists():
        errors.append(f"Input CSV not found: {INPUT_CSV}")

    if errors:
        raise EnvironmentError(
            "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
        )


if __name__ == "__main__":
    validate_config()
    print("[OK] Config loaded successfully!")
    print(f"   Project root     : {PROJECT_ROOT}")
    print(f"   Embedding model  : {EMBEDDING_MODEL} (local)")
    print(f"   Embedding dims   : {EMBEDDING_DIMENSION}")
    print(f"   Generation model : {GENERATION_MODEL}")
    print(f"   Qdrant           : {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"   Collection       : {COLLECTION_NAME}")