# HackerRank Orchestrate — Support Triage Agent

A production-style AI agent that triages support tickets across three product ecosystems — **HackerRank**, **Claude**, and **Visa** — using RAG (Retrieval Augmented Generation).

---

## Architecture

```
support_tickets.csv
        ↓
    main.py              ← orchestrates everything
        ↓
    agent.py             ← per-ticket pipeline
      ├── guardrails.py  ← input/output safety scanning
      ├── classifier.py  ← rules-based routing + validation
      ├── retriever.py   ← semantic search over corpus
      │     ├── embedder.py      ← local sentence-transformers
      │     └── vector_store.py  ← Qdrant vector DB
      └── Claude API     ← grounded response generation
        ↓
    output.csv
```

### Stack
| Component | Technology | Why |
|---|---|---|
| Vector DB | Qdrant (Docker) | Production-grade, open source |
| Embeddings | `all-MiniLM-L6-v2` | Local, free, no rate limits |
| Generation | Claude (`claude-sonnet-4-5`) | Structured JSON output |
| Safety | Custom guardrails | Injection, PII, hallucination |

---

## Project Structure

```
code/
├── main.py          # Entry point — reads CSV, writes output
├── agent.py         # Full pipeline per ticket
├── guardrails.py    # Input/output safety scanning
├── classifier.py    # Rules-based escalation + classification
├── retriever.py     # Semantic search + context formatting
├── vector_store.py  # Qdrant client wrapper
├── embedder.py      # Local sentence-transformer embeddings
├── ingestor.py      # Document loader + chunker
├── ingest.py        # One-time ingestion pipeline
├── config.py        # Centralized settings
└── README.md        # You are here
```

---

## Setup

### Prerequisites
- Python 3.10+
- Docker (for Qdrant)
- Anthropic API key

### 1. Start Qdrant
```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v /path/to/qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 2. Install dependencies
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 3. Configure environment
Create `.env` in the project root:
```
ANTHROPIC_API_KEY=your_key_here
```

### 4. Ingest the corpus (one-time)
```bash
cd code
python ingest.py
```
This loads all `.md` files from `data/`, embeds them locally,
and stores 13,244 vectors in Qdrant. Takes ~4 minutes on CPU.

### 5. Run the agent
```bash
python main.py
```
Processes all tickets in `support_tickets/support_tickets.csv`
and writes results to `support_tickets/output.csv`.

---

## How It Works

### RAG Pipeline
1. **Ingest** — 450+ support docs chunked into 13,244 pieces, embedded locally, stored in Qdrant
2. **Retrieve** — each ticket is embedded and searched against Qdrant using cosine similarity
3. **Generate** — Claude reads the top-5 relevant chunks and generates a structured JSON response

### Safety Layers
The agent runs a "safety sandwich" around every LLM call:

```
Input → Guardrails → Classifier → [LLM] → Guardrails → Classifier → Output
```

| Layer | What it catches |
|---|---|
| Input guardrails | Prompt injection, PII, malicious code requests |
| Classifier pre-check | High-risk keywords, out-of-scope tickets |
| LLM | Nuanced reasoning, corpus-grounded responses |
| Output guardrails | Hallucinated policies, overconfident claims |
| Classifier post-check | Schema validation, safe defaults |

### Escalation Logic
The agent escalates when:
- Hard rules trigger (fraud, stolen, site down, security breach)
- Guardrails detect injection or malicious intent
- Retrieved context doesn't sufficiently cover the issue
- LLM response shows signs of policy hallucination

---

## Output Schema

| Column | Values | Description |
|---|---|---|
| `status` | `replied`, `escalated` | Agent decision |
| `product_area` | category string | Most relevant support domain |
| `response` | text | User-facing answer |
| `justification` | text | Routing decision explanation |
| `request_type` | `product_issue`, `feature_request`, `bug`, `invalid` | Request classification |

---

## Key Design Decisions

**Local embeddings over API embeddings**
`sentence-transformers` runs on-device with no rate limits, no cost,
and no network dependency. For a one-time ingestion of 13K chunks,
this is significantly faster and more reliable than API-based embeddings.

**Rules + LLM, not LLM alone**
For safety-critical decisions (fraud, security, site outages),
deterministic rules always win over probabilistic LLM output.
The classifier handles non-negotiables; Claude handles nuance.

**Single collection, metadata filtering**
All three companies share one Qdrant collection.
Company filtering happens at query time via payload filters.
This is simpler to maintain than three separate collections.

**Structured JSON output**
Claude is prompted to return only valid JSON.
This makes parsing reliable and avoids brittle text extraction.
Invalid JSON is cleaned and re-parsed before crashing.