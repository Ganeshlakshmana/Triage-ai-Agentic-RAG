"""
langgraph_agent.py — LangGraph StateGraph Triage Pipeline

Responsibility:
    Implement the agentic triage loop as a typed LangGraph StateGraph,
    replacing the hand-written for-loop in agent.py with an explicit
    state machine that has named nodes, edges, and conditional routing.

Why LangGraph?
    The hand-written loop in agent.py works, but has implicit state
    (local variables) and implicit transitions (if/elif chains).
    LangGraph makes state and transitions explicit:
    - State = TypedDict (typed, inspectable, serializable)
    - Nodes = named Python functions
    - Edges = conditional routing functions

    This enables:
    - Visual graph tracing
    - LangSmith observability
    - Easier testing (inject state at any node)
    - Clean integration with LangChain tool calling

Graph topology:
    [START] → guard_check → classify → search → reflect → answer → [END]
                   │                     ↑         │
                   │                     └─ loop ──┘
                   └──── block / escalate ──────────→ [END]

CV-aligned features:
    - LangChain + LangGraph usage (matches CV claim)
    - Integration with LLMRouter (multi-provider routing)
    - All 4 safety layers wired in
"""

from __future__ import annotations
import json
import re
import sys
from typing import TypedDict, Optional, Annotated
from typing_extensions import Literal

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("[LangGraph] langgraph not installed. Install with: pip install langgraph", file=sys.stderr)

from llm_router import LLMRouter, get_router
from guardrails import (
    full_scan, get_input_block_response, apply_output_guardrails,
    pre_generation_confidence_score, get_routed_provider,
)
from classifier import (
    should_escalate, is_out_of_scope, classify_request_type,
    infer_company, validate_and_fix_output, check_greetings_and_gratitude,
    get_escalation_request_type,
)
from tools import execute_tool, build_tools_prompt

# ── Shared system prompt (same as agent.py for consistency) ──────────────────

SYSTEM_PROMPT = """You are an expert support triage agent for HackerRank, Claude, and Visa.

You work in an agentic loop. Each iteration return a JSON action object.

STRICT RULES:
- Base responses ONLY on retrieved documentation - never use outside knowledge
- If documentation doesn't cover the issue, escalate
- For high-risk issues (fraud, security, outages), always escalate
- Be honest about confidence - low confidence means escalate, not guess

PRODUCT AREA RULE:
Use the category name from the retrieved documentation source.
Examples: screen, privacy, community, integrations, interviews, library,
settings, general-help, billing, support, travel_support, general_support,
conversation-management, account-management, features-and-capabilities,
claude-code, team-and-enterprise-plans, pro-and-max-plans, safeguards.

RESPONSE FORMAT — always valid JSON:

Search iteration:
{"action": "search_company_docs"|"search_all_docs"|"search_refined"|"escalate"|"reply_out_of_scope", "query": "...", "company": "HackerRank"|"Claude"|"Visa"|null, "reason": "...", "confidence": 0}

Final answer:
{"action": "final_answer", "confidence": 7, "status": "replied"|"escalated", "product_area": "...", "response": "...", "justification": "...", "request_type": "product_issue"|"feature_request"|"bug"|"invalid"}

confidence 1-4: search again or escalate
confidence 5-6: reply with partial info + suggest contacting support
confidence 7-10: answer confidently"""

MAX_ITERATIONS = 3
CONFIDENCE_THRESHOLD = 5


# ── LangGraph State ───────────────────────────────────────────────────────────

class TriageState(TypedDict):
    """
    The full state of a triage pipeline run.
    LangGraph passes this between nodes, each node reads/writes fields.
    """
    # Input
    issue: str
    subject: str
    company: str
    provider: str                   # Requested LLM provider

    # Enriched fields (set by classify node)
    inferred_company: str
    hint_type: str
    routed_provider: str            # After Layer 4 routing gate

    # Safety results
    guard_action: str               # "allow", "warn", "block", "escalate"
    warning_prefix: str

    # Layer 3 confidence
    confidence_score: dict          # Output of pre_generation_confidence_score()

    # Agentic loop state
    iteration: int
    search_history: list[dict]
    messages: list[dict]            # LLM conversation history
    last_action: dict               # Last JSON action from LLM

    # Final output
    result: Optional[dict]
    done: bool                      # True when the pipeline should stop
    events: list[tuple[str, dict]]  # SSE events accumulated during run


# ── Helper ────────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
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
        return {"action": "escalate", "reason": "JSON parse error"}


def _emit(state: TriageState, event_type: str, data: dict) -> None:
    """Append an SSE event to the state's event list."""
    state["events"].append((event_type, data))


# ── Graph Nodes ───────────────────────────────────────────────────────────────

def node_guard_check(state: TriageState) -> TriageState:
    """
    Node 1: Run all input guardrails (Layers 1 & 2).
    Sets guard_action to control routing out of this node.
    """
    _emit(state, "status", {"message": "Running safety checks..."})

    # Layer 0: Greetings / gratitude fast-path
    special = check_greetings_and_gratitude(state["issue"], state["subject"])
    if special:
        state["result"] = validate_and_fix_output(special)
        state["done"] = True
        state["guard_action"] = "special"
        return state

    # Layers 1 & 2: Prompt injection, PII, malicious code, toxicity
    guard_result, warning_prefix = full_scan(state["issue"], state["subject"])
    state["warning_prefix"] = warning_prefix

    if guard_result.action == "block":
        state["result"] = validate_and_fix_output(get_input_block_response(guard_result))
        state["done"] = True
        state["guard_action"] = "block"
        return state

    if guard_result.action == "escalate":
        state["result"] = validate_and_fix_output({
            "status": "escalated",
            "product_area": "security",
            "response": (
                warning_prefix +
                "Your ticket has been flagged for security review. "
                "A human agent will follow up shortly."
            ),
            "justification": f"Guardrail: {guard_result.violations[0][:80]}",
            "request_type": "product_issue",
        })
        state["done"] = True
        state["guard_action"] = "escalate"
        return state

    # Out of scope check
    oos, oos_reason = is_out_of_scope(state["issue"], state["subject"])
    if oos:
        state["result"] = validate_and_fix_output({
            "status": "replied",
            "product_area": "general",
            "response": "I'm sorry, this request is outside the scope of my support capabilities. I can help with HackerRank, Claude, and Visa support topics.",
            "justification": f"Out of scope: {oos_reason}",
            "request_type": "invalid",
        })
        state["done"] = True
        state["guard_action"] = "out_of_scope"
        return state

    # Hard escalation rules
    esc, esc_reason = should_escalate(state["issue"], state["subject"])
    if esc:
        state["result"] = validate_and_fix_output({
            "status": "escalated",
            "product_area": "escalation",
            "response": "This issue requires immediate attention from our support team. A human agent will contact you shortly.",
            "justification": f"Auto-escalated: {esc_reason}",
            "request_type": get_escalation_request_type(state["issue"], state["subject"]),
        })
        state["done"] = True
        state["guard_action"] = "hard_escalate"
        return state

    state["guard_action"] = "allow"
    return state


def node_classify(state: TriageState) -> TriageState:
    """
    Node 2: Classify & enrich the ticket.
    - Infer company
    - Classify request type (hint for LLM)
    - Run Layer 3 confidence scorer
    - Run Layer 4 provider routing gate
    """
    state["inferred_company"] = infer_company(
        state["issue"], state["subject"], state["company"]
    )
    state["hint_type"] = classify_request_type(state["issue"], state["subject"])

    # Layer 3: Confidence score
    conf = pre_generation_confidence_score(state["issue"], state["subject"])
    state["confidence_score"] = conf

    # Layer 4: Provider routing gate
    routed = get_routed_provider(conf, state.get("provider", "auto"))
    state["routed_provider"] = routed

    _emit(state, "status", {
        "message": f"Detected: {state['inferred_company']} | {state['hint_type']} | provider={routed}"
    })
    _emit(state, "confidence", {
        "score": conf["score"],
        "level": conf["level"],
        "provider_routed_to": routed,
    })

    return state


def node_search(state: TriageState) -> TriageState:
    """
    Node 3: Run one LLM iteration (search or final answer).
    This node is re-entered up to MAX_ITERATIONS times.
    """
    iteration = state["iteration"]
    _emit(state, "iteration", {"current": iteration, "max": MAX_ITERATIONS})
    _emit(state, "status", {"message": f"Thinking... (iteration {iteration}/{MAX_ITERATIONS})"})

    # Build prompt
    if iteration == 1:
        user_prompt = _build_initial_prompt(state)
    else:
        user_prompt = _build_iteration_prompt(state)

    state["messages"].append({"role": "user", "content": user_prompt})

    # Call LLM via router
    router = LLMRouter(provider=state.get("routed_provider", "auto"))
    raw = ""

    try:
        for token, provider in router.stream(state["messages"], system=SYSTEM_PROMPT):
            if token == "__PROVIDER__":
                _emit(state, "provider", {"name": provider})
                continue
            raw += token
            _emit(state, "raw_token", {"text": token})
    except Exception as e:
        raw = json.dumps({
            "action": "escalate",
            "reason": f"LLM call failed: {str(e)[:100]}",
        })

    state["messages"].append({"role": "assistant", "content": raw})
    state["last_action"] = _parse_json(raw)
    return state


def node_reflect(state: TriageState) -> TriageState:
    """
    Node 4: Evaluate the last action and decide next step.
    For search results: record in history and decide whether to loop or answer.
    For final_answer / escalate: finalize the result.
    """
    action = state["last_action"]
    action_type = action.get("action", "")
    iteration = state["iteration"]

    if action_type == "final_answer":
        confidence = action.get("confidence", 5)

        if confidence < CONFIDENCE_THRESHOLD and iteration == MAX_ITERATIONS:
            action["status"] = "escalated"
            action["response"] = (
                "This issue requires human review for an accurate answer. "
                "A support agent will follow up shortly."
            )
            action["justification"] = (
                f"Low confidence ({confidence}/10) after {iteration} iterations. "
                + action.get("justification", "")
            )

        if state["warning_prefix"]:
            action["response"] = state["warning_prefix"] + action.get("response", "")

        action = apply_output_guardrails(action)

        # Stream final response word by word
        response_text = action.get("response", "")
        _emit(state, "status", {"message": "Generating response..."})
        words = response_text.split(" ")
        for i, word in enumerate(words):
            _emit(state, "token", {"text": word + (" " if i < len(words) - 1 else "")})

        state["result"] = validate_and_fix_output(action)
        state["done"] = True

    elif action_type == "escalate":
        state["result"] = validate_and_fix_output({
            "status": "escalated",
            "product_area": "escalation",
            "response": (
                state["warning_prefix"] +
                "This issue requires attention from our support team. "
                "A human agent will contact you shortly."
            ),
            "justification": action.get("reason", "Agent escalation decision"),
            "request_type": state["hint_type"],
        })
        state["done"] = True

    elif action_type == "reply_out_of_scope":
        state["result"] = validate_and_fix_output({
            "status": "replied",
            "product_area": "general",
            "response": "I'm sorry, this request is outside the scope of my support capabilities.",
            "justification": action.get("reason", "Out of scope"),
            "request_type": "invalid",
        })
        state["done"] = True

    elif action_type in ("search_company_docs", "search_all_docs", "search_refined"):
        query = action.get("query", state["issue"])
        tool_company = action.get("company", state["inferred_company"])

        _emit(state, "tool", {
            "name": action_type,
            "query": query,
            "company": str(tool_company),
            "reason": action.get("reason", ""),
        })
        _emit(state, "status", {"message": f"Searching: {query[:50]}..."})

        tool_result = execute_tool(
            tool_name=action_type,
            query=query,
            company=tool_company,
        )

        state["search_history"].append({
            "tool": action_type,
            "query": query,
            "best_score": tool_result.best_score,
            "is_relevant": tool_result.is_relevant(),
            "context": tool_result.context[:300] if tool_result.context else "",
            "full_context": tool_result.context,
        })

        _emit(state, "status", {
            "message": f"Found {len(tool_result.content)} chunks (relevance: {tool_result.best_score:.2f})"
        })

        # Increment iteration for next loop
        state["iteration"] += 1

        # Check if we've hit max iterations
        if state["iteration"] > MAX_ITERATIONS:
            best_score = max(
                (h["best_score"] for h in state["search_history"]), default=0.0
            )
            state["result"] = validate_and_fix_output({
                "status": "escalated",
                "product_area": "general",
                "response": (
                    "After thorough review, this issue requires human expertise. "
                    "A support agent will contact you shortly."
                ),
                "justification": (
                    f"Escalated after {MAX_ITERATIONS} iterations. "
                    f"Best retrieval score: {best_score:.3f}"
                ),
                "request_type": state["hint_type"],
            })
            state["done"] = True
    else:
        # Unknown action
        state["result"] = validate_and_fix_output({
            "status": "escalated",
            "product_area": "general",
            "response": "Unable to process this request. A support agent will follow up.",
            "justification": f"Unknown action '{action_type}' from agent",
            "request_type": state["hint_type"],
        })
        state["done"] = True

    return state


# ── Prompt Builders ───────────────────────────────────────────────────────────

def _build_initial_prompt(state: TriageState) -> str:
    return f"""SUPPORT TICKET
{'=' * 50}
Company : {state['inferred_company']}
Subject : {state['subject'] or '(none)'}
Issue   : {state['issue']}

CLASSIFIER HINT
{'=' * 50}
Suggested request type: {state['hint_type']}

{build_tools_prompt()}
{'=' * 50}
Iteration 1: Analyze this ticket and decide which tool to use first.
Return your action as JSON."""


def _build_iteration_prompt(state: TriageState) -> str:
    history = state["search_history"]
    history_text = ""
    for i, h in enumerate(history, 1):
        history_text += f"\nSearch {i}: {h['tool']} | query='{h['query']}'\n"
        history_text += f"  Best score: {h['best_score']:.4f} | Relevant: {h['is_relevant']}\n"
        history_text += f"  Context preview: {h['context'][:300]}...\n"
        history_text += f"  {'-' * 40}\n"

    iteration = state["iteration"]
    last_warning = ""
    if iteration == MAX_ITERATIONS:
        last_warning = f"""
!!! FINAL ITERATION WARNING !!!
This is the final iteration ({iteration} of {MAX_ITERATIONS}).
You MUST output \"final_answer\" to reply or escalate. Do NOT use search actions.
"""

    return f"""SUPPORT TICKET
{'=' * 50}
Company : {state['inferred_company']}
Subject : {state['subject'] or '(none)'}
Issue   : {state['issue']}

SEARCH HISTORY
{'=' * 50}
{history_text}

LATEST RETRIEVED CONTEXT
{'=' * 50}
{history[-1]['full_context'] if history else 'No results yet.'}

{build_tools_prompt()}
{'=' * 50}
Iteration {iteration} of {MAX_ITERATIONS}:
{last_warning}
- If results are relevant (score > 0.35) and confident -> use \"final_answer\"
- If confidence < {CONFIDENCE_THRESHOLD} -> try a different tool or refined query
- Last iteration -> give best answer or escalate

Return your action as JSON."""


# ── Routing Functions (Edge Conditions) ───────────────────────────────────────

def route_after_guard(state: TriageState) -> Literal["classify", "end"]:
    """After guard check: proceed to classify or end (if blocked/escalated)."""
    return "end" if state["done"] else "classify"


def route_after_reflect(state: TriageState) -> Literal["search", "end"]:
    """After reflect: loop back to search or end."""
    return "end" if state["done"] else "search"


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph():
    """Build and compile the LangGraph StateGraph."""
    if not LANGGRAPH_AVAILABLE:
        return None

    graph = StateGraph(TriageState)

    # Add nodes
    graph.add_node("guard_check", node_guard_check)
    graph.add_node("classify", node_classify)
    graph.add_node("search", node_search)
    graph.add_node("reflect", node_reflect)

    # Add edges
    graph.add_edge(START, "guard_check")
    graph.add_conditional_edges(
        "guard_check",
        route_after_guard,
        {"classify": "classify", "end": END},
    )
    graph.add_edge("classify", "search")
    graph.add_edge("search", "reflect")
    graph.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"search": "search", "end": END},
    )

    return graph.compile()


# Compile once at import time
_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# ── Public API ────────────────────────────────────────────────────────────────

def process_ticket_langgraph(
    issue: str,
    subject: str = "",
    company: str = "None",
    provider: str = "auto",
):
    """
    Generator that runs the LangGraph triage pipeline and yields SSE events.

    Yields:
        Tuples of (event_type, data_dict) identical to agent.py's generator,
        for drop-in compatibility with the API server.

    Args:
        issue    : Ticket body
        subject  : Subject line
        company  : Company hint ("HackerRank", "Claude", "Visa", "None")
        provider : LLM provider ("claude", "openai", "gemini", "auto")
    """
    graph = get_graph()
    if graph is None:
        # LangGraph not available — signal caller to use fallback
        yield "error", {"message": "LangGraph not available. Using fallback agent."}
        return

    initial_state: TriageState = {
        "issue": issue,
        "subject": subject,
        "company": company,
        "provider": provider,
        "inferred_company": "",
        "hint_type": "",
        "routed_provider": provider,
        "guard_action": "",
        "warning_prefix": "",
        "confidence_score": {},
        "iteration": 1,
        "search_history": [],
        "messages": [],
        "last_action": {},
        "result": None,
        "done": False,
        "events": [],
    }

    try:
        final_state = graph.invoke(initial_state)

        # Yield all accumulated events
        for event_type, data in final_state.get("events", []):
            yield event_type, data

        # Yield the final result
        if final_state.get("result"):
            yield "result", final_state["result"]
        else:
            yield "result", validate_and_fix_output({
                "status": "escalated",
                "product_area": "general",
                "response": "Unable to process request. A support agent will follow up.",
                "justification": "Pipeline completed without result.",
                "request_type": "product_issue",
            })

    except Exception as e:
        yield "error", {"message": str(e)}


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not LANGGRAPH_AVAILABLE:
        print("[ERROR] langgraph not installed. Run: pip install langgraph")
        sys.exit(1)

    print("Testing LangGraph agent...\n")

    test_issue = "How do I add extra time for a candidate who needs accessibility accommodations?"

    print(f"Ticket: {test_issue}\n")
    print("=" * 55)

    for event_type, data in process_ticket_langgraph(
        issue=test_issue,
        subject="Extra time for candidate",
        company="HackerRank",
        provider="auto",
    ):
        if event_type == "status":
            print(f"  [STATUS] {data['message']}")
        elif event_type == "tool":
            print(f"  [TOOL]   {data['name']} | query={data['query'][:50]}")
        elif event_type == "provider":
            print(f"  [PROVIDER] Using: {data['name']}")
        elif event_type == "confidence":
            print(f"  [LAYER3] Score={data['score']} level={data['level']} → provider={data['provider_routed_to']}")
        elif event_type == "result":
            print(f"\n  status       : {data.get('status')}")
            print(f"  product_area : {data.get('product_area')}")
            print(f"  request_type : {data.get('request_type')}")
            print(f"  response     : {data.get('response', '')[:120]}")
        elif event_type == "error":
            print(f"  [ERROR] {data['message']}")

    print("\n[OK] LangGraph agent test complete!")
