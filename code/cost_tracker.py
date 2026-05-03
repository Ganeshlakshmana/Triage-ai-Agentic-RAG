"""
cost_tracker.py - Token Usage and Cost Tracking

Responsibility:
    Track token usage and estimated costs across all tickets.
    Report per-ticket and aggregate costs.

Why cost tracking matters:
    The JD says "make agents smarter, more reliable, AND CHEAPER."
    You can't make something cheaper if you don't measure it first.

    In production, LLM costs can surprise you:
    - A prompt that's 200 tokens too long x 10,000 tickets/day
      = 2,000,000 extra tokens/day = $$$
    - Knowing cost per ticket lets you optimize prompts,
      reduce iterations, and justify the agent's ROI.

Claude Sonnet pricing (as of 2025):
    Input  : $3.00 per 1M tokens
    Output : $15.00 per 1M tokens

    These are approximate - check Anthropic's pricing page
    for current rates: https://www.anthropic.com/pricing
"""

from dataclasses import dataclass, field


# Pricing Constants
# Claude claude-sonnet-4-5 pricing per token
INPUT_COST_PER_TOKEN  = 3.00  / 1_000_000   # $3.00 per 1M input tokens
OUTPUT_COST_PER_TOKEN = 15.00 / 1_000_000   # $15.00 per 1M output tokens


# Data Structure: TicketCost
@dataclass
class TicketCost:
    """Cost breakdown for one ticket."""
    ticket_idx: int
    issue_preview: str
    iterations: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def cost_usd(self) -> float:
        return (
            self.total_input_tokens * INPUT_COST_PER_TOKEN +
            self.total_output_tokens * OUTPUT_COST_PER_TOKEN
        )

    def add_iteration(self, input_tokens: int, output_tokens: int) -> None:
        self.iterations += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens


# Cost Tracker - Singleton
class CostTracker:
    """
    Tracks token usage across all tickets in a run.

    Usage:
        tracker = CostTracker()
        tracker.start_ticket(1, "How do I...")
        tracker.record_iteration(1, input_tokens=523, output_tokens=87)
        tracker.record_iteration(1, input_tokens=1205, output_tokens=215)
        tracker.end_ticket(1)
        tracker.print_summary()
    """

    def __init__(self):
        self.tickets: dict[int, TicketCost] = {}
        self._current_ticket: int | None = None

    def start_ticket(self, ticket_idx: int, issue: str) -> None:
        """Start tracking a new ticket."""
        self._current_ticket = ticket_idx
        self.tickets[ticket_idx] = TicketCost(
            ticket_idx=ticket_idx,
            issue_preview=issue[:60],
        )

    def record_iteration(
        self,
        ticket_idx: int,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record token usage for one iteration."""
        if ticket_idx in self.tickets:
            self.tickets[ticket_idx].add_iteration(input_tokens, output_tokens)

    def end_ticket(self, ticket_idx: int) -> TicketCost | None:
        """Finalize a ticket and return its cost."""
        return self.tickets.get(ticket_idx)

    @property
    def total_input_tokens(self) -> int:
        return sum(t.total_input_tokens for t in self.tickets.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(t.total_output_tokens for t in self.tickets.values())

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        return sum(t.cost_usd for t in self.tickets.values())

    @property
    def avg_cost_per_ticket(self) -> float:
        if not self.tickets:
            return 0.0
        return self.total_cost_usd / len(self.tickets)

    @property
    def avg_iterations(self) -> float:
        if not self.tickets:
            return 0.0
        return sum(t.iterations for t in self.tickets.values()) / len(self.tickets)

    def print_summary(self) -> None:
        """Print full cost report."""
        if not self.tickets:
            print("No cost data recorded.")
            return

        print("\n" + "=" * 55)
        print("  COST TRACKING REPORT")
        print("=" * 55)

        # Per-ticket breakdown (top 5 most expensive)
        sorted_tickets = sorted(
            self.tickets.values(),
            key=lambda t: t.cost_usd,
            reverse=True,
        )

        print("\nMost Expensive Tickets:")
        for t in sorted_tickets[:5]:
            print(
                f"   [{t.ticket_idx:02d}] ${t.cost_usd:.4f} | "
                f"{t.iterations} iter | "
                f"{t.total_tokens:,} tokens | "
                f"{t.issue_preview[:40]}..."
            )

        # Aggregate stats
        print(f"\nAggregate Stats:")
        print(f"   Total tickets    : {len(self.tickets)}")
        print(f"   Total tokens     : {self.total_tokens:,}")
        print(f"     Input tokens   : {self.total_input_tokens:,}")
        print(f"     Output tokens  : {self.total_output_tokens:,}")
        print(f"   Avg iterations   : {self.avg_iterations:.1f}")
        print(f"   Avg tokens/ticket: {self.total_tokens // len(self.tickets):,}")

        print(f"\nCost Breakdown:")
        input_cost = self.total_input_tokens * INPUT_COST_PER_TOKEN
        output_cost = self.total_output_tokens * OUTPUT_COST_PER_TOKEN
        print(f"   Input cost       : ${input_cost:.4f}")
        print(f"   Output cost      : ${output_cost:.4f}")
        print(f"   Total cost       : ${self.total_cost_usd:.4f}")
        print(f"   Cost per ticket  : ${self.avg_cost_per_ticket:.4f}")

        # Projections
        print(f"\nCost Projections:")
        print(f"   100 tickets/day  : ${self.avg_cost_per_ticket * 100:.2f}/day")
        print(f"   1,000 tickets/day: ${self.avg_cost_per_ticket * 1000:.2f}/day")
        print(f"   10,000 tickets/day: ${self.avg_cost_per_ticket * 10000:.2f}/day")

        # Optimization tips
        print(f"\nOptimization Tips:")
        if self.avg_iterations > 1.5:
            print(f"   - Avg {self.avg_iterations:.1f} iterations - improve retrieval quality")
            print(f"     to reduce iterations and cut costs")
        if self.total_output_tokens / self.total_tokens > 0.3:
            print(f"   - Output is {self.total_output_tokens/self.total_tokens*100:.0f}% of tokens")
            print(f"     Consider limiting max_tokens to reduce output cost")
        print(f"   - Output tokens cost 5x more than input tokens")
        print(f"     Shorter, more precise responses = significant savings")

        print("=" * 55)


# Global tracker instance
# Using a module-level singleton so agent.py and main.py
# can both access the same tracker without passing it around.
# This is the "module singleton" pattern.
tracker = CostTracker()