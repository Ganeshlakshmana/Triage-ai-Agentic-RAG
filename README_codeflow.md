# TriageAI — Intelligent Support Triage Application

> **Evolved from the HackerRank Orchestrate Hackathon (May 2026)**
> Originally a batch CSV-processing agent, TriageAI has been upgraded into a
> full-stack, chat-first support intelligence platform — think ChatGPT or Claude,
> but purpose-built for resolving support tickets across **HackerRank**, **Claude**,
> and **Visa** product ecosystems.

---

## What Is TriageAI?

TriageAI is an AI-powered support agent you interact with like a chat app.
You describe your support issue in plain language, and TriageAI:

1. **Understands** the context — product, issue type, urgency
2. **Searches** a 450+ document knowledge base using semantic similarity
3. **Reasons** through the results using Claude (claude-sonnet-4-5) in an agentic loop
4. **Responds** with a grounded, accurate answer — or escalates to a human if it can't be sure
5. **Streams** the response token-by-token, just like ChatGPT

All of this happens in real time through a React chat interface backed by a FastAPI streaming server.

---

## From Hackathon Script to Full Application

| Hackathon Version | TriageAI (Upgraded) |
|---|---|
| Batch CSV input (`support_tickets.csv`) | Real-time chat interface |
| CLI-only (`python main.py`) | React web app + REST API |
| Single-pass processing | Live streaming with SSE tokens |
| No UI | ChatGPT-style sidebar + message thread |
| Manual company selection | Auto-detect from message content |
| Output to `output.csv` | Structured metadata badges in UI |

---

## Full System Architecture

```
USER (Browser)
    |
    | types a support message
    v
+---------------------------+
|   React Frontend (Vite)   |  <- ChatGPT-style chat UI
|   frontend/src/App.jsx    |  <- Sidebar, message bubbles, tool call badges
+---------------------------+
    |
    | POST /api/chat  (SSE stream)
    v
+---------------------------+
|   FastAPI Backend         |  <- api/main.py
|   Streaming SSE server    |  <- thread-based async bridge
+---------------------------+
    |
    v
+---------------------------+
|   Safety Layer (Input)    |  <- code/guardrails.py
|   - Prompt injection      |  <- code/classifier.py
|   - PII detection         |
|   - Out-of-scope check    |
|   - Hard escalation rules |
+---------------------------+
    |
    v
+---------------------------+
|   Agentic Loop (max 3x)   |  <- code/agent.py
|   Claude claude-sonnet-4-5 reasoning  |
|   Tool use: search docs   |
+---------------------------+
    |         |
    |         v
    |   +---------------------+
    |   |  Retriever          |  <- code/retriever.py
    |   |  - embed query      |  <- code/embedder.py (local, no API)
    |   |  - cosine search    |  <- code/vector_store.py (Qdrant)
    |   |  - format context   |
    |   +---------------------+
    |
    v
+---------------------------+
|   Safety Layer (Output)   |  <- code/guardrails.py
|   - Hallucination scan    |
|   - Schema validation     |  <- code/classifier.py
+---------------------------+
    |
    | SSE tokens streamed back
    v
USER sees response word-by-word with status badges
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **UI** | React + Vite + Tailwind CSS | Chat interface, streaming display |
| **API** | FastAPI + Server-Sent Events | Real-time token streaming to browser |
| **LLM** | Claude `claude-sonnet-4-5` (Anthropic) | Reasoning, grounded response generation |
| **Embeddings** | `all-MiniLM-L6-v2` (sentence-transformers) | Local, free, no API calls or rate limits |
| **Vector DB** | Qdrant (Docker) | 13,244 document chunk vectors, cosine search |
| **Safety** | Custom guardrails module | PII, injection, hallucination, schema validation |
| **Classification** | Rule-based classifier | Escalation triggers, request type, company inference |

---

## Project Structure

```
.
+-- api/
|   +-- main.py              # FastAPI server — /api/chat (SSE) + /api/health
|
+-- frontend/
|   +-- src/
|   |   +-- App.jsx          # Full chat UI — sidebar, messages, streaming, badges
|   |   +-- App.css          # Global styles
|   |   +-- main.jsx         # React entry point
|   +-- package.json         # Node dependencies (Vite, React, Tailwind)
|   +-- vite.config.js       # Dev server config
|   +-- index.html
|
+-- code/
|   +-- config.py            # All settings: models, paths, thresholds, keywords
|   +-- agent.py             # Agentic loop (max 3 iterations), streaming Claude
|   +-- tools.py             # Tool definitions + executor (search variants)
|   +-- retriever.py         # Semantic search + context formatting
|   +-- embedder.py          # Local sentence-transformer embeddings
|   +-- vector_store.py      # Qdrant client wrapper (create, store, search)
|   +-- ingestor.py          # Document loader + text chunker
|   +-- ingest.py            # One-time ingestion pipeline (CLI)
|   +-- classifier.py        # Escalation rules, request type, company inference
|   +-- guardrails.py        # Input/output safety scanning
|   +-- cost_tracker.py      # Token usage + USD cost tracking
|   +-- evals.py             # Evaluation suite vs sample_support_tickets.csv
|   +-- main.py              # Batch mode: reads CSV, runs agent, writes output.csv
|
+-- data/
|   +-- hackerrank/          # 150+ HackerRank support docs (.md)
|   +-- claude/              # 150+ Claude support docs (.md)
|   +-- visa/                # 150+ Visa support docs (.md)
|
+-- support_tickets/
|   +-- support_tickets.csv          # Full ticket dataset (batch mode input)
|   +-- sample_support_tickets.csv   # Labeled sample for evals
|
+-- .env                     # Your secrets (never committed)
+-- .env.example             # Template showing required keys
+-- requirements.txt         # Python dependencies
```

---

## How the Chat Works (Step by Step)

### 1. User sends a message
The React frontend (`App.jsx`) POSTs to `http://localhost:8000/api/chat` with:
```json
{ "message": "My Visa card was declined abroad", "company": "None", "subject": "" }
```

### 2. FastAPI receives the request (`api/main.py`)
- Spawns a background thread to run the agent (sync code bridged to async SSE)
- Opens a `Server-Sent Events` stream back to the browser
- Sends real-time events: `status`, `iteration`, `tool`, `token`, `result`, `done`

### 3. Safety pre-checks
```
guardrails.full_scan()      -> blocks prompt injection, flags PII
classifier.is_out_of_scope()-> returns early for irrelevant requests
classifier.should_escalate()-> hard rules: fraud, stolen card, security breach
```

### 4. Agentic loop (up to 3 iterations)
Each iteration Claude returns a JSON action:
```json
{"action": "search_company_docs", "query": "card declined international", "company": "Visa"}
```
The agent executes the tool, gets back the top-7 semantically similar chunks from Qdrant, feeds them as context, and asks Claude again.

### 5. Final answer streamed token by token
When Claude reaches a `final_answer` action:
```json
{
  "action": "final_answer",
  "confidence": 8,
  "status": "replied",
  "product_area": "travel_support",
  "response": "Your Visa card may be declined abroad due to...",
  "justification": "Found relevant documentation in Visa travel_support",
  "request_type": "product_issue"
}
```
Each word of `response` is sent as a `token` SSE event — the UI renders them one by one with a blinking cursor.

### 6. UI renders result
The chat bubble shows:
- The full response text
- A `replied` / `escalated` status badge
- A `product_area` tag (e.g. "travel_support")
- A `request_type` tag (e.g. "product issue")
- A justification note in muted text below

---

## Safety Architecture

The agent runs a "safety sandwich" around every request:

```
Request -> [Input Guardrails] -> [Agentic Loop] -> [Output Guardrails] -> Response
```

| Check | What it catches |
|---|---|
| Prompt injection detection | "Ignore previous instructions...", role-playing attacks |
| PII scanner | Credit card numbers, passwords, SSNs, emails |
| Hard escalation rules | Fraud, stolen credentials, site-wide outages, security breach |
| Out-of-scope filter | Requests unrelated to HackerRank / Claude / Visa |
| Output hallucination scan | LLM claiming policies that don't exist in docs |
| Schema validation | Enforces valid `status`, `request_type`, `product_area` values |

---

## Setup & Running Locally

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker (for Qdrant)
- Anthropic API key

### 1. Start Qdrant (vector database)
```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

### 2. Python environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 3. Environment variables
Copy `.env.example` to `.env` and fill in your key:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Ingest the knowledge base (one-time, ~4 minutes)
```bash
cd code
python ingest.py
```
Loads 450+ `.md` files, chunks them into 13,244 pieces, embeds locally using
`all-MiniLM-L6-v2`, and stores vectors in Qdrant. No API calls — runs entirely on-device.

### 5. Start the FastAPI backend
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
API available at: `http://localhost:8000`
Health check: `http://localhost:8000/api/health`

### 6. Start the React frontend
```bash
cd frontend
npm install
npm run dev
```
App available at: `http://localhost:5173`

---

## Batch Mode (Original Hackathon Flow)

The original CLI mode still works for bulk processing:
```bash
cd code
python main.py
```
Reads `support_tickets/support_tickets.csv`, runs the full agent per ticket,
writes results to `support_tickets/output.csv`.

### Run evaluations
```bash
python evals.py
```
Tests the agent against `sample_support_tickets.csv` (labeled ground truth)
and prints accuracy scores for status, request type, and product area.

---

## Output Schema

Every agent response — whether via chat or batch — produces:

| Field | Values | Description |
|---|---|---|
| `status` | `replied`, `escalated` | Agent's routing decision |
| `product_area` | e.g. `billing`, `travel_support`, `interviews` | Most relevant support domain |
| `response` | Free text | The user-facing support answer |
| `justification` | Free text | Why this routing decision was made |
| `request_type` | `product_issue`, `feature_request`, `bug`, `invalid` | Issue classification |
| `confidence` | 1-10 | Agent's self-assessed certainty (internal) |

---

## Key Design Decisions

**Local embeddings over API embeddings**
`sentence-transformers` runs on-device — no rate limits, no cost, no network dependency.
For 13K chunks this is significantly faster and more reliable than any embedding API.

**Rules + LLM, not LLM alone**
For safety-critical decisions (fraud, stolen credentials, site outages),
deterministic rules always win over probabilistic LLM output.
The classifier handles non-negotiables; Claude handles nuance.

**Thread-based async bridge in the API**
The Anthropic streaming SDK is synchronous. FastAPI needs async generators for SSE.
`api/main.py` runs the agent in a daemon thread and drains events via a `queue.Queue`,
bridging the two worlds cleanly without blocking the event loop.

**Single Qdrant collection with metadata filtering**
All three companies share one collection.
Company filtering happens at query time via payload filters — simpler to
maintain than three separate collections, and enables cross-company search.

**Structured JSON output enforced throughout**
Claude is prompted to return only valid JSON.
A robust parser handles markdown fences and partial JSON before falling back to escalation.
This makes the pipeline deterministic and avoids brittle regex extraction.

---

## Hackathon Origin

This project began as the **HackerRank Orchestrate** hackathon submission (May 1-2, 2026).
The challenge: build an AI agent to resolve real support tickets accurately.

The hackathon deliverable was a batch pipeline (`main.py` + `output.csv`).
**TriageAI** is the upgraded version — taking the same agentic RAG core and
wrapping it in a full-stack application that any user can interact with
through a chat interface, making the technology accessible beyond just CSV files.