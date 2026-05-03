"""
api/main.py — FastAPI Backend for TriageAI (Fixed async streaming)
"""

import sys
import os
import json
import asyncio
import re
import threading
import queue
from pathlib import Path

CODE_DIR = Path(__file__).parent.parent / "code"
sys.path.insert(0, str(CODE_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import validate_config, ANTHROPIC_API_KEY, GENERATION_MODEL
from classifier import (
    should_escalate, is_out_of_scope, infer_company,
    classify_request_type, validate_and_fix_output,
    get_escalation_request_type
)
from guardrails import full_scan, get_input_block_response
from tools import execute_tool, build_tools_prompt
import anthropic

# ─────────────────────────────────────────────
app = FastAPI(title="TriageAI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

validate_config()
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

MAX_ITERATIONS = 3
CONFIDENCE_THRESHOLD = 5

SYSTEM_PROMPT = """You are TriageAI, an expert support agent for HackerRank, Claude, and Visa.

You work in an agentic loop. Each iteration return a JSON action object.

STRICT RULES:
- Base responses ONLY on retrieved documentation
- If docs don't cover the issue, escalate
- For high-risk issues (fraud, security, outages), always escalate

PRODUCT AREA: Use the category name from retrieved source metadata.
Examples: screen, privacy, community, integrations, interviews, settings,
general-help, billing, support, travel_support, conversation-management,
account-management, features-and-capabilities, safeguards.

TOOLS:
- search_company_docs: known company + clear query
- search_all_docs: unknown company or cross-domain
- search_refined: previous search scored < 0.35
- escalate: can't answer confidently
- reply_out_of_scope: clearly irrelevant

RESPONSE FORMAT — always valid JSON only:

Search iteration:
{"action": "search_company_docs"|"search_all_docs"|"search_refined"|"escalate"|"reply_out_of_scope", "query": "...", "company": "HackerRank"|"Claude"|"Visa"|null, "reason": "...", "confidence": 0}

Final answer:
{"action": "final_answer", "confidence": 7, "status": "replied"|"escalated", "product_area": "from source metadata", "response": "...", "justification": "...", "request_type": "product_issue"|"feature_request"|"bug"|"invalid"}

confidence 1-4: search again or escalate
confidence 5-10: reply (5-6 with caveats, 7-10 confidently)"""


class ChatRequest(BaseModel):
    message: str
    company: str = "None"
    subject: str = ""


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"action": "escalate", "reason": "parse error"}


def run_agent_in_thread(message: str, company: str, subject: str, event_queue: queue.Queue):
    """
    Run the full agentic pipeline in a background thread.
    Puts SSE events into the queue as they happen.
    The async generator reads from the queue.
    """
    try:
        # Pre-checks
        event_queue.put(sse("status", {"message": "Analyzing your request..."}))

        guard_result, warning_prefix = full_scan(message, subject)
        if guard_result.action == "block":
            event_queue.put(sse("result", get_input_block_response(guard_result)))
            return

        oos, oos_reason = is_out_of_scope(message, subject)
        if oos:
            event_queue.put(sse("result", validate_and_fix_output({
                "status": "replied",
                "product_area": "general",
                "response": "I'm sorry, this request is outside the scope of my support capabilities. I can help with HackerRank, Claude, and Visa support topics.",
                "justification": f"Out of scope: {oos_reason}",
                "request_type": "invalid",
            })))
            return

        escalate, escalation_reason = should_escalate(message, subject)
        if escalate:
            event_queue.put(sse("result", validate_and_fix_output({
                "status": "escalated",
                "product_area": "escalation",
                "response": "This issue requires immediate attention from our support team. A human agent will contact you shortly.",
                "justification": f"Auto-escalated: {escalation_reason}",
                "request_type": get_escalation_request_type(message, subject),
            })))
            return

        inferred_company = infer_company(message, subject, company)
        hint_type = classify_request_type(message, subject)

        event_queue.put(sse("status", {"message": f"Detected: {inferred_company} | {hint_type}"}))

        messages = []
        search_history = []

        for iteration in range(1, MAX_ITERATIONS + 1):
            event_queue.put(sse("iteration", {"current": iteration, "max": MAX_ITERATIONS}))
            event_queue.put(sse("status", {"message": f"Thinking... (iteration {iteration}/{MAX_ITERATIONS})"}))

            # Build prompt
            if iteration == 1:
                user_prompt = f"""SUPPORT TICKET
{'='*50}
Company : {inferred_company}
Subject : {subject or '(none)'}
Issue   : {message}

CLASSIFIER HINT: {hint_type}

{build_tools_prompt()}
{'='*50}
Iteration 1: Decide which tool to use. Return JSON."""
            else:
                history_text = ""
                for i, h in enumerate(search_history, 1):
                    history_text += f"Search {i}: {h['tool']} | '{h['query']}' | score={h['best_score']:.3f}\n"
                    history_text += f"Preview: {h['context'][:200]}\n\n"

                user_prompt = f"""SUPPORT TICKET: {message}
Company: {inferred_company}

SEARCH HISTORY:
{history_text}

LATEST CONTEXT:
{search_history[-1]['full_context'][:1500] if search_history else 'None'}

{build_tools_prompt()}
Iteration {iteration}/{MAX_ITERATIONS}: Review results and decide. Return JSON."""

            messages.append({"role": "user", "content": user_prompt})

            # Call Claude synchronously (we're in a thread)
            raw = ""
            with claude_client.messages.stream(
                model=GENERATION_MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                for token in stream.text_stream:
                    raw += token

            messages.append({"role": "assistant", "content": raw})
            action = parse_json(raw)
            action_type = action.get("action", "")

            if action_type == "final_answer":
                confidence = action.get("confidence", 5)
                if confidence < CONFIDENCE_THRESHOLD and iteration == MAX_ITERATIONS:
                    action["status"] = "escalated"
                    action["response"] = "This issue requires human review. A support agent will follow up shortly."

                if warning_prefix:
                    action["response"] = warning_prefix + action.get("response", "")

                # Stream response word by word
                response_text = action.get("response", "")
                event_queue.put(sse("status", {"message": "Generating response..."}))
                words = response_text.split(" ")
                for i, word in enumerate(words):
                    event_queue.put(sse("token", {"text": word + (" " if i < len(words)-1 else "")}))

                event_queue.put(sse("result", validate_and_fix_output(action)))
                return

            elif action_type == "escalate":
                event_queue.put(sse("result", validate_and_fix_output({
                    "status": "escalated",
                    "product_area": "escalation",
                    "response": "This issue requires attention from our support team. A human agent will contact you shortly.",
                    "justification": action.get("reason", "Agent escalation"),
                    "request_type": hint_type,
                })))
                return

            elif action_type == "reply_out_of_scope":
                event_queue.put(sse("result", validate_and_fix_output({
                    "status": "replied",
                    "product_area": "general",
                    "response": "I'm sorry, this request is outside the scope of my support capabilities.",
                    "justification": action.get("reason", "Out of scope"),
                    "request_type": "invalid",
                })))
                return

            elif action_type in ("search_company_docs", "search_all_docs", "search_refined"):
                query = action.get("query", message)
                tool_company = action.get("company", inferred_company)

                event_queue.put(sse("tool", {
                    "name": action_type,
                    "query": query,
                    "company": str(tool_company),
                    "reason": action.get("reason", ""),
                }))
                event_queue.put(sse("status", {"message": f"Searching: {query[:50]}..."}))

                tool_result = execute_tool(action_type, query=query, company=tool_company)

                search_history.append({
                    "tool": action_type,
                    "query": query,
                    "best_score": tool_result.best_score,
                    "is_relevant": tool_result.is_relevant(),
                    "context": tool_result.context[:300] if tool_result.context else "",
                    "full_context": tool_result.context,
                })

                event_queue.put(sse("status", {
                    "message": f"Found {len(tool_result.content)} chunks (relevance: {tool_result.best_score:.2f})"
                }))

        # Max iterations reached
        event_queue.put(sse("result", validate_and_fix_output({
            "status": "escalated",
            "product_area": "general",
            "response": "After thorough review, this issue requires human expertise. A support agent will contact you shortly.",
            "justification": f"Escalated after {MAX_ITERATIONS} iterations.",
            "request_type": hint_type,
        })))

    except Exception as e:
        event_queue.put(sse("error", {"message": str(e)}))
    finally:
        event_queue.put(None)  # Signal done


async def generate_response(message: str, company: str, subject: str):
    """
    Async generator that reads from a queue fed by a background thread.
    This is the clean way to bridge sync code with async SSE streaming.
    """
    event_queue = queue.Queue()

    # Start agent in background thread
    thread = threading.Thread(
        target=run_agent_in_thread,
        args=(message, company, subject, event_queue),
        daemon=True,
    )
    thread.start()

    # Read events from queue and yield them
    loop = asyncio.get_event_loop()
    while True:
        # Non-blocking queue check with asyncio
        try:
            event = await loop.run_in_executor(None, lambda: event_queue.get(timeout=60))
        except queue.Empty:
            yield sse("error", {"message": "Request timed out"})
            break

        if event is None:  # Done signal
            yield sse("done", {})
            break

        yield event
        await asyncio.sleep(0)  # Yield control to event loop


@app.post("/api/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        generate_response(request.message, request.company, request.subject),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": GENERATION_MODEL}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)