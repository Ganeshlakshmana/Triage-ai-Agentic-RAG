# Support Triage Agent - Source Code

This directory contains the Python codebase for the Support Triage Agent (TriageAI).

## Project Architecture

The pipeline uses an **LLM Sandwich** architecture for safety and correctness, combined with an **Agentic RAG** loop:

```
Input -> [Guardrails Scan] -> [Pre-check Intercepts] -> [Agentic Search Loop] -> [Output Guardrails] -> Output
```

1. **Input Guardrails** (`guardrails.py`): Scans input for prompt injection, toxic content, and PII.
2. **Pre-checks** (`classifier.py`): Deterministic overrides for quick hand-offs of high-risk security tickets, invalid empty requests, greetings, or gratitude.
3. **Agentic Loop** (`agent.py` & `tools.py`): Reasoning agent powered by Claude that chooses search tools, evaluates retrieval relevance, refines queries, and reflects before responding.
4. **Semantic Retrieval** (`retriever.py`, `embedder.py`, `vector_store.py`): Local sentence-transformer embeddings indexing 13,244 document chunks stored in Qdrant.
5. **Output Guardrails** (`guardrails.py`): Post-generation check that scans for hallucinated policy phrases or overconfident claims before replying to the user.

---

## Core Scripts & Usage

First, activate your virtual environment:
```bash
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 1. Ingestion Pipeline
To ingest support documentation from `data/` into Qdrant:
```bash
python ingest.py
```
To force a full rebuild:
```bash
python ingest.py --force
```

### 2. Batch Processing Mode
To run the agent in batch mode against `support_tickets/support_tickets.csv` and write outputs to `support_tickets/output.csv`:
```bash
python main.py
```

### 3. Evaluation Suite
To measure status, request type, and product area accuracy against the ground-truth labeled sample set:
```bash
python evals.py
```

### 4. Interactive Development Testing
- Run individual module scripts to perform unit tests:
```bash
python guardrails.py
python classifier.py
python vector_store.py
python test_retrieve.py
```
