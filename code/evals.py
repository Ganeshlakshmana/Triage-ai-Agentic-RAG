"""
evals.py - Evaluation Framework

Responsibility:
    Compare agent output against expected results from
    sample_support_tickets.csv to measure accuracy.

Why evals matter:
    Anyone can build an agent that runs without errors.
    Evals prove it produces CORRECT outputs.
    Without evals, you're flying blind - you don't know
    if a code change made things better or worse.

    This is what separates production AI engineering
    from demo-ware.

What we measure:
    - Status accuracy    (replied/escalated correct?)
    - Request type accuracy (product_issue/bug/etc correct?)
    - Product area accuracy (loose match - categories vary)
    - Overall score      (weighted combination)

Principle: Measure everything. Trust nothing.
    If you can't measure it, you can't improve it.
"""

import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass, field
from config import SAMPLE_CSV, validate_config
from agent import process_ticket


# Data Structure: EvalResult
@dataclass
class EvalResult:
    """Result for one ticket evaluation."""
    ticket_idx: int
    issue: str
    company: str

    # Expected values
    expected_status: str = ""
    expected_request_type: str = ""
    expected_product_area: str = ""

    # Agent output
    got_status: str = ""
    got_request_type: str = ""
    got_product_area: str = ""
    got_response: str = ""

    # Scores (1=correct, 0=wrong)
    status_correct: bool = False
    request_type_correct: bool = False
    product_area_correct: bool = False

    error: str = ""

    @property
    def overall_score(self) -> float:
        """Weighted score for this ticket."""
        # Status is most important (40%)
        # Request type is important (35%)
        # Product area is approximate (25%)
        return (
            self.status_correct * 0.40 +
            self.request_type_correct * 0.35 +
            self.product_area_correct * 0.25
        )


# Scoring Functions
def score_status(expected: str, got: str) -> bool:
    """Exact match for status field."""
    return expected.strip().lower() == got.strip().lower()


def score_request_type(expected: str, got: str) -> bool:
    """Exact match for request_type field."""
    return expected.strip().lower() == got.strip().lower()


def score_product_area(expected: str, got: str) -> bool:
    """
    Loose match for product_area.

    Why loose?
        Expected values in sample CSV use categories like "screen",
        "privacy", "community". Our agent produces values like
        "test_management", "conversation_management".
        These mean the same thing but use different words.

        We check if either:
        1. Exact match
        2. One contains the other
        3. Share a key word (screen, billing, privacy, etc.)
    """
    expected = expected.strip().lower()
    got = got.strip().lower()

    if expected == got:
        return True
    if expected in got or got in expected:
        return True

    # Check for shared key domain words
    domain_words = [
        "screen", "billing", "privacy", "security", "account",
        "interview", "community", "integration", "library",
        "settings", "support", "general", "escalat",
        "conversation", "travel", "payment", "card",
    ]
    for word in domain_words:
        if word in expected and word in got:
            return True

    return False


# Main Eval Runner
def run_evals(verbose: bool = True) -> list[EvalResult]:
    """
    Run evaluation against sample_support_tickets.csv.

    Args:
        verbose : Print per-ticket results if True

    Returns:
        List of EvalResult objects
    """
    # Load sample tickets with expected outputs
    if not SAMPLE_CSV.exists():
        raise FileNotFoundError(
            f"Sample CSV not found: {SAMPLE_CSV}\n"
            "This file contains expected outputs for evaluation."
        )

    df = pd.read_csv(SAMPLE_CSV)
    df.columns = [c.strip() for c in df.columns]
    df = df.fillna("")

    print(f"Loaded {len(df)} sample tickets for evaluation")
    print(f"   Columns: {list(df.columns)}\n")

    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        issue = str(row.get("Issue", row.get("issue", ""))).strip()
        subject = str(row.get("Subject", row.get("subject", ""))).strip()
        company = str(row.get("Company", row.get("company", "None"))).strip()

        # Expected outputs from sample CSV
        expected_status = str(row.get("Status", row.get("status", ""))).strip().lower()
        expected_request_type = str(row.get("Request Type", row.get("request_type", ""))).strip().lower()
        expected_product_area = str(row.get("Product Area", row.get("product_area", ""))).strip().lower()

        result = EvalResult(
            ticket_idx=idx + 1,
            issue=issue[:80],
            company=company,
            expected_status=expected_status,
            expected_request_type=expected_request_type,
            expected_product_area=expected_product_area,
        )

        try:
            # Run agent
            output = process_ticket(issue=issue, subject=subject, company=company)

            result.got_status = output.get("status", "")
            result.got_request_type = output.get("request_type", "")
            result.got_product_area = output.get("product_area", "")
            result.got_response = output.get("response", "")[:100]

            # Score each field
            result.status_correct = score_status(expected_status, result.got_status)
            result.request_type_correct = score_request_type(expected_request_type, result.got_request_type)
            result.product_area_correct = score_product_area(expected_product_area, result.got_product_area)

        except Exception as e:
            result.error = str(e)[:100]

        results.append(result)

        # Print per-ticket result
        if verbose:
            score = result.overall_score
            icon = "[OK]" if score >= 0.75 else "[WARNING]" if score >= 0.40 else "[ERROR]"
            tqdm.write(
                f"  {icon} [{idx+1:02d}] score={score:.2f} | "
                f"status={'[OK]' if result.status_correct else '[ERROR]'} "
                f"type={'[OK]' if result.request_type_correct else '[ERROR]'} "
                f"area={'[OK]' if result.product_area_correct else '[ERROR]'} "
                f"| {issue[:45]}..."
            )
            if not result.status_correct:
                tqdm.write(
                    f"       status: expected='{expected_status}' "
                    f"got='{result.got_status}'"
                )
            if not result.request_type_correct:
                tqdm.write(
                    f"       type  : expected='{expected_request_type}' "
                    f"got='{result.got_request_type}'"
                )

    return results


# Summary Reporter
def print_eval_summary(results: list[EvalResult]) -> dict:
    """
    Print a comprehensive evaluation summary.

    Returns:
        Dict with all scores for programmatic use
    """
    total = len(results)
    if total == 0:
        print("No results to summarize.")
        return {}

    # Calculate scores
    status_correct = sum(1 for r in results if r.status_correct)
    type_correct = sum(1 for r in results if r.request_type_correct)
    area_correct = sum(1 for r in results if r.product_area_correct)
    overall_scores = [r.overall_score for r in results]
    avg_overall = sum(overall_scores) / total

    status_acc = status_correct / total * 100
    type_acc = type_correct / total * 100
    area_acc = area_correct / total * 100
    overall_acc = avg_overall * 100

    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)

    print(f"\nField Accuracy:")
    print(f"   {'Status':<20} {status_correct:>2}/{total} ({status_acc:5.1f}%)  {'[OK]' if status_acc >= 80 else '[WARNING]' if status_acc >= 60 else '[ERROR]'}")
    print(f"   {'Request Type':<20} {type_correct:>2}/{total} ({type_acc:5.1f}%)  {'[OK]' if type_acc >= 80 else '[WARNING]' if type_acc >= 60 else '[ERROR]'}")
    print(f"   {'Product Area':<20} {area_correct:>2}/{total} ({area_acc:5.1f}%)  {'[OK]' if area_acc >= 70 else '[WARNING]' if area_acc >= 50 else '[ERROR]'}")

    print(f"\nOverall Score: {overall_acc:.1f}%")
    bar_filled = int(overall_acc / 5)
    bar = "#" * bar_filled + "-" * (20 - bar_filled)
    print(f"   [{bar}] {overall_acc:.1f}%")

    # Grade
    if overall_acc >= 85:
        grade = "A - Excellent"
    elif overall_acc >= 70:
        grade = "B - Good"
    elif overall_acc >= 55:
        grade = "C - Needs improvement"
    else:
        grade = "D - Significant issues"
    print(f"   Grade: {grade}")

    # Failure analysis
    failures = [r for r in results if r.overall_score < 0.75]
    if failures:
        print(f"\n[WARNING] Tickets needing attention ({len(failures)}):")
        for r in failures[:5]:  # show top 5
            print(f"   [{r.ticket_idx:02d}] score={r.overall_score:.2f} | {r.issue[:50]}")
            if not r.status_correct:
                print(f"        status: expected={r.expected_status} | got={r.got_status}")
            if not r.request_type_correct:
                print(f"        type  : expected={r.expected_request_type} | got={r.got_request_type}")

    print("=" * 55)

    return {
        "status_accuracy": status_acc,
        "request_type_accuracy": type_acc,
        "product_area_accuracy": area_acc,
        "overall_score": overall_acc,
        "total_tickets": total,
        "grade": grade,
    }


# Entry Point
if __name__ == "__main__":
    print("=" * 55)
    print("  HackerRank Orchestrate - Evaluation Suite")
    print("=" * 55)

    validate_config()

    results = run_evals(verbose=True)
    scores = print_eval_summary(results)

    print(f"\nTo improve scores:")
    if scores.get("status_accuracy", 0) < 80:
        print("   - Status accuracy low: review escalation rules in classifier.py")
    if scores.get("request_type_accuracy", 0) < 80:
        print("   - Request type low: add more keywords to REQUEST_TYPE_HINTS")
    if scores.get("product_area_accuracy", 0) < 70:
        print("   - Product area low: improve retrieval or system prompt")