"""
tools.py - Tool Definitions for the Agentic Loop

Responsibility:
    Define the set of actions the agent can take during reasoning.
    Each tool has:
    - A name and description (what Claude sees when deciding)
    - An execute() function (what actually runs)
    - A result schema (what gets returned)

Why tools matter in agentic systems:
    In a fixed RAG pipeline, retrieval always happens once with
    the original query. The agent has no choice.

    With tools, Claude DECIDES:
    - Should I search HackerRank docs or all docs?
    - Should I refine my query before searching?
    - Should I escalate immediately without searching?
    - Should I search again with a different angle?

    This decision-making is what makes the system "agentic."

Principle: Tools are the agent's hands.
    The LLM is the brain - it reasons and decides.
    Tools are what it uses to interact with the world.
    Keep tools simple, focused, and well-described.
    The description is what Claude reads to decide whether to use it.
"""

from dataclasses import dataclass, field
from retriever import retrieve, format_context, build_query
from config import TOP_K


# Data Structure: ToolResult
@dataclass
class ToolResult:
    """
    Result from executing a tool.

    Fields:
        tool_name   : Which tool was called
        success     : Whether it executed without error
        content     : The actual result (retrieved chunks, etc.)
        context     : Formatted text ready for LLM consumption
        metadata    : Extra info (scores, chunk count, etc.)
        error       : Error message if success=False
    """
    tool_name: str
    success: bool = True
    content: list = field(default_factory=list)
    context: str = ""
    metadata: dict = field(default_factory=dict)
    error: str = ""

    @property
    def best_score(self) -> float:
        """Highest relevance score from retrieved chunks."""
        if not self.content:
            return 0.0
        return max(c.get("score", 0.0) for c in self.content)

    @property
    def avg_score(self) -> float:
        """Average relevance score from retrieved chunks."""
        if not self.content:
            return 0.0
        return sum(c.get("score", 0.0) for c in self.content) / len(self.content)

    def is_relevant(self, threshold: float = 0.35) -> bool:
        """
        Check if retrieval results are relevant enough.

        Threshold of 0.35 means: top result must be at least
        35% similar to the query. Below this = probably wrong docs.

        Why 0.35?
        - all-MiniLM-L6-v2 scores tend to range 0.2-0.9
        - Below 0.35 = weak signal, likely irrelevant
        - Above 0.35 = reasonable match, worth using
        """
        return self.best_score >= threshold


# Tool Definitions

# This is what Claude reads to decide which tool to use.
# TOOL_DESCRIPTIONS must be clear, specific, and distinguishable.
# Vague descriptions = Claude picks the wrong tool.

TOOL_DESCRIPTIONS = {
    "search_company_docs": (
        "Search support documentation for a specific company. "
        "Use when the company is known (HackerRank, Claude, or Visa) "
        "and the query is clear. This is the most precise search."
    ),
    "search_all_docs": (
        "Search across ALL companies' support documentation. "
        "Use when company is unknown, unclear, or when company-specific "
        "search returned poor results. Broader but less precise."
    ),
    "search_refined": (
        "Search with a refined/rewritten query. "
        "Use when previous searches returned low relevance scores "
        "and you want to try a different angle or simpler phrasing."
    ),
    "escalate": (
        "Escalate this ticket to a human support agent. "
        "Use when: the issue is high-risk, documentation doesn't cover it, "
        "confidence is low after multiple searches, or human judgment is needed."
    ),
    "reply_out_of_scope": (
        "Reply that this request is outside support scope. "
        "Use when the ticket is clearly unrelated to HackerRank, "
        "Claude, or Visa products (e.g. general knowledge questions, "
        "greetings, or requests for harmful content)."
    ),
}


# Tool Executor

def execute_tool(
    tool_name: str,
    query: str = "",
    company: str = None,
    reason: str = "",
    top_k: int = TOP_K,
) -> ToolResult:
    """
    Execute a tool and return the result.

    Args:
        tool_name : Which tool to run
        query     : Search query (for search tools)
        company   : Company filter (for search_company_docs)
        reason    : Explanation (for escalate/reply tools)
        top_k     : Number of chunks to retrieve

    Returns:
        ToolResult with content, context, and metadata

    This is the dispatcher - routes to the right function
    based on tool_name. Adding a new tool = add it here.
    """
    if tool_name == "search_company_docs":
        return _search_company_docs(query, company, top_k)

    elif tool_name == "search_all_docs":
        return _search_all_docs(query, top_k)

    elif tool_name == "search_refined":
        return _search_refined(query, company, top_k)

    elif tool_name == "escalate":
        return _escalate(reason)

    elif tool_name == "reply_out_of_scope":
        return _reply_out_of_scope(reason)

    else:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            error=f"Unknown tool: '{tool_name}'. "
                  f"Available: {list(TOOL_DESCRIPTIONS.keys())}",
        )


# Individual Tool Implementations

def _search_company_docs(
    query: str,
    company: str,
    top_k: int,
) -> ToolResult:
    """
    Search within a specific company's documentation.

    Most precise search - filters Qdrant by company payload field
    before doing vector similarity search.
    """
    try:
        chunks = retrieve(query=query, company=company, top_k=top_k)
        context = format_context(chunks)

        result = ToolResult(
            tool_name="search_company_docs",
            content=chunks,
            context=context,
            metadata={
                "query": query,
                "company_filter": company,
                "chunks_found": len(chunks),
                "best_score": max((c["score"] for c in chunks), default=0.0),
            },
        )
        return result

    except Exception as e:
        return ToolResult(
            tool_name="search_company_docs",
            success=False,
            error=str(e),
        )


def _search_all_docs(query: str, top_k: int) -> ToolResult:
    """
    Search across all companies' documentation.

    Less precise than company-filtered search but broader coverage.
    Used as fallback when company is unknown or company search failed.
    """
    try:
        chunks = retrieve(query=query, company=None, top_k=top_k)
        context = format_context(chunks)

        return ToolResult(
            tool_name="search_all_docs",
            content=chunks,
            context=context,
            metadata={
                "query": query,
                "company_filter": None,
                "chunks_found": len(chunks),
                "best_score": max((c["score"] for c in chunks), default=0.0),
            },
        )
    except Exception as e:
        return ToolResult(
            tool_name="search_all_docs",
            success=False,
            error=str(e),
        )


def _search_refined(query: str, company: str, top_k: int) -> ToolResult:
    """
    Search with a refined query.

    Called when Claude decides the original query wasn't specific enough.
    The query passed here is Claude's rewritten version.

    Why this is a separate tool:
        Having a named "search_refined" tool forces Claude to explicitly
        decide "I need a better query" rather than silently retrying.
        This makes the reasoning loop visible and traceable.
    """
    try:
        # Try company-filtered first, fall back to all if no company
        chunks = retrieve(query=query, company=company, top_k=top_k)

        # If company-filtered gives poor results, try without filter
        if not chunks or max((c["score"] for c in chunks), default=0) < 0.35:
            chunks_all = retrieve(query=query, company=None, top_k=top_k)
            if chunks_all and chunks_all[0]["score"] > (chunks[0]["score"] if chunks else 0):
                chunks = chunks_all

        context = format_context(chunks)

        return ToolResult(
            tool_name="search_refined",
            content=chunks,
            context=context,
            metadata={
                "refined_query": query,
                "chunks_found": len(chunks),
                "best_score": max((c["score"] for c in chunks), default=0.0),
            },
        )
    except Exception as e:
        return ToolResult(
            tool_name="search_refined",
            success=False,
            error=str(e),
        )


def _escalate(reason: str) -> ToolResult:
    """
    Escalate to human agent.
    No retrieval needed - just signals the escalation decision.
    """
    return ToolResult(
        tool_name="escalate",
        content=[],
        context="",
        metadata={"reason": reason},
    )


def _reply_out_of_scope(reason: str) -> ToolResult:
    """
    Reply that ticket is out of scope.
    No retrieval needed - just signals the out-of-scope decision.
    """
    return ToolResult(
        tool_name="reply_out_of_scope",
        content=[],
        context="",
        metadata={"reason": reason},
    )


# Tool Prompt Builder

def build_tools_prompt() -> str:
    """
    Build the tools section for the LLM prompt.
    Shows Claude what tools are available and when to use each.
    """
    lines = ["AVAILABLE TOOLS:", ""]
    for name, description in TOOL_DESCRIPTIONS.items():
        lines.append(f"  {name}:")
        lines.append(f"    {description}")
        lines.append("")
    return "\n".join(lines)


# Quick Test
if __name__ == "__main__":
    print("Testing tools...\n")

    # Test 1: Company search
    print("Test 1: search_company_docs")
    result = execute_tool(
        "search_company_docs",
        query="how to add extra time for candidates",
        company="HackerRank",
    )
    print(f"  success     : {result.success}")
    print(f"  chunks found: {result.metadata.get('chunks_found')}")
    print(f"  best score  : {result.metadata.get('best_score'):.4f}")
    print(f"  is relevant : {result.is_relevant()}")

    # Test 2: All docs search
    print("\nTest 2: search_all_docs")
    result2 = execute_tool(
        "search_all_docs",
        query="delete my account",
    )
    print(f"  chunks found: {result2.metadata.get('chunks_found')}")
    print(f"  best score  : {result2.metadata.get('best_score'):.4f}")
    print(f"  top company : {result2.content[0]['company'] if result2.content else 'N/A'}")

    # Test 3: Escalate
    print("\nTest 3: escalate")
    result3 = execute_tool("escalate", reason="Site is completely down")
    print(f"  tool     : {result3.tool_name}")
    print(f"  reason   : {result3.metadata.get('reason')}")

    # Test 4: Unknown tool
    print("\nTest 4: unknown tool (error handling)")
    result4 = execute_tool("magic_tool")
    print(f"  success  : {result4.success}")
    print(f"  error    : {result4.error}")

    print("\n[OK] Tools ready!")