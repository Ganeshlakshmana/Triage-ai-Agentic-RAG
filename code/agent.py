"""
agent.py - Agentic RAG Pipeline

Upgrade from fixed RAG to Agentic RAG:

BEFORE (fixed pipeline):
    retrieve -> generate -> done
    One retrieval, no decisions, no loops.

AFTER (agentic loop):
    think -> decide tool -> act -> observe -> reflect -> loop if needed
    Claude decides what to do, can retry with better queries,
    scores its own confidence, and loops until satisfied.

The Agentic Loop (max 3 iterations):
    Iteration 1: Claude reads ticket, picks a tool, executes it
    Iteration 2: Claude sees results, scores confidence (1-10)
                 If low -> picks different tool or refines query
    Iteration 3: Final attempt - generate answer or escalate

Why max 3 iterations?
    Prevents infinite loops. In practice, 2-3 is almost always enough.
    Beyond 3 = the issue likely needs human attention anyway.

Principle: The agent should know what it doesn't know.
    A confident wrong answer is worse than an honest escalation.
    Low confidence -> escalate rather than guess.
"""

import json
import re
import anthropic

from config import ANTHROPIC_API_KEY, GENERATION_MODEL, TOP_K
from classifier import (
    should_escalate,
    is_out_of_scope,
    classify_request_type,
    infer_company,
    validate_and_fix_output,
)
from guardrails import full_scan, get_input_block_response, apply_output_guardrails
from tools import execute_tool, build_tools_prompt, ToolResult
from cost_tracker import tracker

# Initialize Claude Client
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Agentic Loop Settings
MAX_ITERATIONS = 3
CONFIDENCE_THRESHOLD = 5  # out of 10 - below this = try again or escalate


# System Prompt
SYSTEM_PROMPT = """You are an expert support triage agent for HackerRank, Claude, and Visa.

You work in an agentic loop. Each iteration you will:
1. Analyze the support ticket and any previous search results
2. Decide which tool to use next
3. Return a JSON action object

STRICT RULES:
- Base responses ONLY on retrieved documentation - never use outside knowledge
- If documentation doesn't cover the issue, escalate
- For high-risk issues (fraud, security, outages), always escalate
- Be honest about confidence - low confidence means escalate, not guess

PRODUCT AREA RULE - IMPORTANT:
Use the category name from the retrieved documentation source.
Examples of valid product areas: screen, privacy, community, integrations,
interviews, library, settings, general-help, billing, support, travel_support,
general_support, conversation-management, account-management, features-and-capabilities,
claude-code, team-and-enterprise-plans, pro-and-max-plans, safeguards.
Do NOT invent new category names. Use what appears in the Source metadata.

TOOL SELECTION GUIDELINES:
- Known company + clear query -> search_company_docs
- Unknown company or cross-domain -> search_all_docs
- Previous search scored < 0.35 -> search_refined with better query
- Issue clearly out of scope -> reply_out_of_scope
- Can't answer confidently from docs -> escalate

RESPONSE FORMAT - you must ALWAYS return valid JSON:

During search iterations:
{
  "action": "search_company_docs" | "search_all_docs" | "search_refined" | "escalate" | "reply_out_of_scope",
  "query": "your search query (for search actions)",
  "company": "HackerRank" | "Claude" | "Visa" | null,
  "reason": "why you chose this action",
  "confidence": 0
}

When ready to give final answer (after seeing search results):
{
  "action": "final_answer",
  "confidence": 7,
  "status": "replied" | "escalated",
  "product_area": "use the category from Source metadata in retrieved docs",
  "response": "user-facing answer grounded in the documentation",
  "justification": "brief explanation of decision",
  "request_type": "product_issue" | "feature_request" | "bug" | "invalid"
}

confidence is 1-10:
  1-4  = not enough info, must search again or escalate
  5-6  = partial info - REPLY with what you know, direct user to contact support for specifics
  7-10 = good info, answer confidently and completely

IMPORTANT: confidence 5-6 should still result in "replied" status with partial guidance.
Only use "escalated" when you have NO relevant documentation at all (confidence 1-3)
or when the issue is genuinely high-risk (fraud, security breach, account takeover).
Having partial documentation = reply with what you know + suggest contacting support.
"""


# Prompt Builders
def build_initial_prompt(
    issue: str,
    subject: str,
    company: str,
    hint_type: str,
) -> str:
    """First iteration prompt - no search results yet."""
    return f"""SUPPORT TICKET
{'=' * 50}
Company : {company}
Subject : {subject or '(none)'}
Issue   : {issue}

CLASSIFIER HINT
{'=' * 50}
Suggested request type: {hint_type}

{build_tools_prompt()}
{'=' * 50}
Iteration 1: Analyze this ticket and decide which tool to use first.
Return your action as JSON.
"""


def build_iteration_prompt(
    issue: str,
    subject: str,
    company: str,
    iteration: int,
    search_history: list[dict],
) -> str:
    """Subsequent iteration prompts - includes previous search results."""

    history_text = ""
    for i, h in enumerate(search_history, 1):
        history_text += f"\nSearch {i}: {h['tool']} | query='{h['query']}'\n"
        history_text += f"  Best score: {h['best_score']:.4f} | "
        history_text += f"Relevant: {h['is_relevant']}\n"
        history_text += f"  Context preview: {h['context'][:300]}...\n"
        history_text += f"  {'-' * 40}\n"

    return f"""SUPPORT TICKET
{'=' * 50}
Company : {company}
Subject : {subject or '(none)'}
Issue   : {issue}

SEARCH HISTORY (what you've tried so far)
{'=' * 50}
{history_text}

LATEST RETRIEVED CONTEXT
{'=' * 50}
{search_history[-1]['full_context'] if search_history else 'No results yet.'}

{build_tools_prompt()}
{'=' * 50}
Iteration {iteration} of {MAX_ITERATIONS}:
Review the search results above.
- If results are relevant (score > 0.35) and you can answer confidently -> use "final_answer"
- If results are poor or confidence < {CONFIDENCE_THRESHOLD} -> try a different tool or refined query
- If this is the last iteration -> either give best answer or escalate

Return your action as JSON.
"""


# JSON Parser
def parse_json_response(text: str) -> dict:
    """Parse JSON from Claude's response, handling markdown fences."""
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
        raise ValueError(f"Could not parse JSON: {text[:200]}")


# Agentic Loop
def run_agentic_loop(
    issue: str,
    subject: str,
    company: str,
    hint_type: str,
    warning_prefix: str = "",
) -> dict:
    """
    The core agentic reasoning loop.

    Each iteration:
    1. Build prompt (with search history if not first)
    2. Call Claude -> get action JSON
    3. Execute the action (tool call)
    4. If final_answer -> return result
    5. If search -> record results, loop again
    6. If max iterations reached -> escalate

    Args:
        issue         : Ticket body
        subject       : Subject line
        company       : Inferred company
        hint_type     : Pre-classified request type
        warning_prefix: PII/security warning to prepend

    Returns:
        Final output dict with all 5 fields
    """
    search_history = []
    messages = []

    for iteration in range(1, MAX_ITERATIONS + 1):

        #  Build prompt for this iteration 
        if iteration == 1:
            user_prompt = build_initial_prompt(issue, subject, company, hint_type)
        else:
            user_prompt = build_iteration_prompt(
                issue, subject, company, iteration, search_history
            )

        #  Call Claude with Streaming 
        # Why stream here?
        # 1. User sees tokens appear immediately (TTFT ~0.3s vs 10s)
        # 2. We can display thinking in real time
        # 3. Still collect full response for JSON parsing
        # 4. Can cancel early if needed in future
        messages.append({"role": "user", "content": user_prompt})

        try:
            raw = ""  # will collect full streamed response

            # Show iteration context
            print(f"\n  Iteration {iteration}/{MAX_ITERATIONS} - streaming response:", flush=True)
            print(f"  ", end="", flush=True)

            # stream=True returns tokens one by one
            with claude.messages.stream(
                model=GENERATION_MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                for token in stream.text_stream:
                    # Print each token as it arrives
                    # end="" prevents newline, flush=True sends immediately
                    print(token, end="", flush=True)
                    raw += token  # collect into full string

                # get_final_message() waits for stream to finish
                # and returns complete message with usage stats
                final_message = stream.get_final_message()

            print()  # newline after streaming completes

            # Log token usage - important for cost tracking
            usage = final_message.usage
            print(
                f"  Tokens: input={usage.input_tokens} "
                f"output={usage.output_tokens} "
                f"total={usage.input_tokens + usage.output_tokens}",
                flush=True,
            )

            # Record in cost tracker
            tracker.record_iteration(
                ticket_idx=id(issue),  # use issue id as ticket key
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )

            messages.append({"role": "assistant", "content": raw})
            action = parse_json_response(raw)

        except Exception as e:
            # Claude API error - safe fallback
            return validate_and_fix_output({
                "status": "escalated",
                "product_area": "general",
                "response": "We encountered an issue processing your request. A support agent will follow up.",
                "justification": f"Agent error on iteration {iteration}: {str(e)[:100]}",
                "request_type": hint_type,
            })

        #  Handle action 
        action_type = action.get("action", "")

        # Final answer - agent is done
        if action_type == "final_answer":
            confidence = action.get("confidence", 5)

            # Low confidence on final iteration -> escalate
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

            # Add warning prefix if PII was detected
            if warning_prefix:
                action["response"] = warning_prefix + action.get("response", "")

            return validate_and_fix_output(action)

        # Escalate - agent decided to hand off
        elif action_type == "escalate":
            return validate_and_fix_output({
                "status": "escalated",
                "product_area": "escalation",
                "response": (
                    warning_prefix +
                    "This issue requires attention from our support team. "
                    "A human agent will contact you shortly."
                ),
                "justification": action.get("reason", "Agent escalation decision"),
                "request_type": hint_type,
            })

        # Out of scope
        elif action_type == "reply_out_of_scope":
            return validate_and_fix_output({
                "status": "replied",
                "product_area": "general",
                "response": (
                    "I'm sorry, this request is outside the scope of my support capabilities. "
                    "I can help with HackerRank, Claude, and Visa support topics."
                ),
                "justification": action.get("reason", "Out of scope"),
                "request_type": "invalid",
            })

        # Search action - execute tool and record results
        elif action_type in ("search_company_docs", "search_all_docs", "search_refined"):
            query = action.get("query", issue)
            tool_company = action.get("company", company)

            tool_result: ToolResult = execute_tool(
                tool_name=action_type,
                query=query,
                company=tool_company,
            )

            # Record this search in history
            search_history.append({
                "tool": action_type,
                "query": query,
                "best_score": tool_result.best_score,
                "is_relevant": tool_result.is_relevant(),
                "context": tool_result.context[:300] if tool_result.context else "",
                "full_context": tool_result.context,
            })

        else:
            # Unknown action - treat as escalation
            return validate_and_fix_output({
                "status": "escalated",
                "product_area": "general",
                "response": "Unable to process this request. A support agent will follow up.",
                "justification": f"Unknown action '{action_type}' from agent",
                "request_type": hint_type,
            })

    #  Max iterations reached without final_answer 
    # This means agent kept searching but never got confident enough
    # Best safe action: escalate
    return validate_and_fix_output({
        "status": "escalated",
        "product_area": search_history[-1]["tool"] if search_history else "general",
        "response": (
            "After thorough review, this issue requires human expertise. "
            "A support agent will contact you shortly."
        ),
        "justification": (
            f"Escalated after {MAX_ITERATIONS} iterations. "
            f"Best retrieval score: {max(h['best_score'] for h in search_history):.3f}"
            if search_history else "No relevant documentation found."
        ),
        "request_type": hint_type,
    })


# Main Entry Point
def process_ticket(
    issue: str,
    subject: str = "",
    company: str = "None",
) -> dict:
    """
    Full agentic pipeline for one support ticket.

    Pre-checks (no LLM):
        1. Guardrails input scan
        2. Out-of-scope check
        3. Hard escalation rules

    Agentic loop (with LLM):
        4. Claude reasons, picks tools, searches, reflects
        5. Max 3 iterations
        6. Output guardrails

    Args:
        issue   : Ticket body
        subject : Subject line
        company : Company from CSV

    Returns:
        Dict with status, product_area, response,
        justification, request_type
    """

    #  Pre-check 1: Guardrails 
    guard_result, warning_prefix = full_scan(issue, subject)

    if guard_result.action == "block":
        return validate_and_fix_output(get_input_block_response(guard_result))

    if guard_result.action == "escalate":
        return validate_and_fix_output({
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

    #  Pre-check 2: Out of scope 
    from classifier import is_out_of_scope, get_escalation_request_type
    out_of_scope, oos_reason = is_out_of_scope(issue, subject)
    if out_of_scope:
        return validate_and_fix_output({
            "status": "replied",
            "product_area": "general",
            "response": "I'm sorry, this request is outside the scope of my support capabilities.",
            "justification": f"Out of scope: {oos_reason}",
            "request_type": "invalid",
        })

    #  Pre-check 3: Hard escalation rules 
    escalate, escalation_reason = should_escalate(issue, subject)
    if escalate:
        return validate_and_fix_output({
            "status": "escalated",
            "product_area": "escalation",
            "response": "This issue requires immediate attention from our support team. "
                       "A human agent will contact you shortly.",
            "justification": f"Auto-escalated: {escalation_reason}",
            "request_type": get_escalation_request_type(issue, subject),
        })

    #  Enrich 
    inferred_company = infer_company(issue, subject, company)
    hint_type = classify_request_type(issue, subject)

    #  Agentic Loop 
    result = run_agentic_loop(
        issue=issue,
        subject=subject,
        company=inferred_company,
        hint_type=hint_type,
        warning_prefix=warning_prefix,
    )

    #  Output guardrails 
    return apply_output_guardrails(result)


# Quick Test
if __name__ == "__main__":
    print("Testing agentic agent...\n")

    test_tickets = [
        {
            "issue": "How do I add extra time for a candidate who needs accessibility accommodations?",
            "subject": "Extra time for candidate",
            "company": "HackerRank",
        },
        {
            "issue": "One of my claude conversations has some private info, can I delete it?",
            "subject": "",
            "company": "Claude",
        },
        {
            "issue": "site is down & none of the pages are accessible",
            "subject": "",
            "company": "None",
        },
        {
            "issue": "i cant get in",
            "subject": "",
            "company": "None",
        },
    ]

    for i, ticket in enumerate(test_tickets, 1):
        print(f"{'=' * 55}")
        print(f"Ticket {i}: {ticket['issue'][:55]}")
        print(f"{'=' * 55}")

        result = process_ticket(**ticket)

        print(f"  status       : {result['status']}")
        print(f"  product_area : {result['product_area']}")
        print(f"  request_type : {result['request_type']}")
        print(f"  justification: {result['justification'][:80]}")
        print(f"  response     : {result['response'][:120]}")
        print()

    print("[OK] Agentic agent test complete!")