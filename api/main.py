"""
api/main.py — FastAPI Backend for TriageAI (Multi-Provider Streaming)
"""

import sys
import json
import asyncio
import threading
import queue
from pathlib import Path

CODE_DIR = Path(__file__).parent.parent / "code"
sys.path.insert(0, str(CODE_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import validate_config, GENERATION_MODEL, DEFAULT_PROVIDER

# ─────────────────────────────────────────────
app = FastAPI(title="TriageAI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

validate_config()

MAX_ITERATIONS = 3
CONFIDENCE_THRESHOLD = 5


class ChatRequest(BaseModel):
    message: str
    company: str = "None"
    subject: str = ""
    provider: str = "auto"
    # provider: "claude" | "openai" | "gemini" | "auto"
    # "auto" = LLMRouter picks the best available provider with fallback chain.


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def run_agent_in_thread(
    message: str,
    company: str,
    subject: str,
    provider: str,
    event_queue: queue.Queue,
):
    """
    Run the full agentic pipeline in a background thread.
    Puts SSE events into the queue as they happen.
    The async generator reads from the queue.
    """
    try:
        from agent import process_ticket_generator
        for event_type, data in process_ticket_generator(
            message, subject, company, provider=provider
        ):
            # Don't forward raw LLM JSON tokens to the client
            if event_type == "raw_token":
                continue
            event_queue.put(sse(event_type, data))
    except Exception as e:
        event_queue.put(sse("error", {"message": str(e)}))
    finally:
        event_queue.put(None)  # Signal done


async def generate_response(message: str, company: str, subject: str, provider: str):
    """
    Async generator that reads from a queue fed by a background thread.
    This cleanly bridges sync agent code with async SSE streaming.
    """
    event_queue = queue.Queue()

    thread = threading.Thread(
        target=run_agent_in_thread,
        args=(message, company, subject, provider, event_queue),
        daemon=True,
    )
    thread.start()

    loop = asyncio.get_event_loop()
    while True:
        try:
            event = await loop.run_in_executor(None, lambda: event_queue.get(timeout=60))
        except queue.Empty:
            yield sse("error", {"message": "Request timed out"})
            break

        if event is None:
            yield sse("done", {})
            break

        yield event
        await asyncio.sleep(0)


@app.post("/api/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        generate_response(
            request.message,
            request.company,
            request.subject,
            request.provider,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/health")
async def health():
    """Health check — also exposes provider availability."""
    from llm_router import _available_providers
    available = _available_providers()
    return {
        "status": "ok",
        "model": GENERATION_MODEL,
        "default_provider": DEFAULT_PROVIDER,
        "available_providers": available,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)