"""
main.py - Entry Point and Pipeline Orchestrator

Responsibility:
    1. Read support_tickets.csv
    2. Process each ticket through the full agent pipeline
    3. Write results to output.csv
    4. Show progress and summary statistics

This is the file you run to process all tickets:
    python main.py

Principle: Orchestration - this file coordinates everything
           but contains no business logic itself.
           It just wires the pieces together.
"""

import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from config import (
    validate_config,
    INPUT_CSV,
    OUTPUT_CSV,
    VALID_STATUSES,
    VALID_REQUEST_TYPES,
)
from agent import process_ticket
from vector_store import is_collection_populated
from cost_tracker import tracker


# Startup Checks
def run_startup_checks() -> None:
    """
    Verify everything is ready before processing tickets.
    Fail fast with clear error messages.
    """
    print("Running startup checks...")

    # Check config (API keys, paths)
    validate_config()
    print("   [OK] Config valid")

    # Check Qdrant is populated
    if not is_collection_populated():
        raise RuntimeError(
            "[ERROR] Qdrant collection is empty!\n"
            "   Run: python ingest.py\n"
            "   Then retry: python main.py"
        )
    print("   [OK] Qdrant collection populated")

    # Check input CSV exists
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"[ERROR] Input CSV not found: {INPUT_CSV}")
    print(f"   [OK] Input CSV found: {INPUT_CSV.name}")


# CSV Processing
def load_tickets() -> pd.DataFrame:
    """
    Load support tickets from CSV.

    Returns:
        DataFrame with columns: Issue, Subject, Company
        All values normalized (NaN -> empty string)
    """
    df = pd.read_csv(INPUT_CSV)

    # Normalize column names - strip whitespace, handle case
    df.columns = [c.strip() for c in df.columns]

    # Fill NaN with empty string so we don't crash on missing values
    df = df.fillna("")

    print(f"\nLoaded {len(df)} tickets from {INPUT_CSV.name}")
    print(f"   Columns: {list(df.columns)}")

    return df


def process_all_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run every ticket through the agent pipeline.

    Args:
        df : Input DataFrame with Issue, Subject, Company columns

    Returns:
        Output DataFrame with all 5 result columns added

    Progress tracking:
        tqdm shows a progress bar with ETA.
        Each row's result is printed for visibility.
    """
    results = []
    errors = 0

    print(f"\nProcessing {len(df)} tickets...\n")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing tickets"):
        # Extract fields - handle different possible column name cases
        issue = str(row.get("Issue", row.get("issue", ""))).strip()
        subject = str(row.get("Subject", row.get("subject", ""))).strip()
        company = str(row.get("Company", row.get("company", "None"))).strip()

        # Skip completely empty rows
        if not issue:
            results.append({
                "status": "escalated",
                "product_area": "general",
                "response": "Empty ticket received.",
                "justification": "No issue content provided.",
                "request_type": "invalid",
            })
            continue

        try:
            # Print ticket header before streaming starts
            tqdm.write(f"\n{'-' * 55}")
            tqdm.write(f"  Ticket {idx+1:03d}: {issue[:55]}...")
            tqdm.write(f"{'-' * 55}")

            # Start cost tracking for this ticket
            tracker.start_ticket(idx + 1, issue)

            result = process_ticket(
                issue=issue,
                subject=subject,
                company=company,
            )
            # Keep only the 5 required output columns
            clean_result = {
                "status": result.get("status", "escalated"),
                "product_area": result.get("product_area", "general"),
                "response": result.get("response", ""),
                "justification": result.get("justification", ""),
                "request_type": result.get("request_type", "product_issue"),
            }
            results.append(clean_result)
            tracker.end_ticket(idx + 1)

            tqdm.write(
                f"\n  [OK] [{idx+1:03d}] {clean_result['status']:<10} "
                f"{clean_result['request_type']:<20}"
            )

        except Exception as e:
            errors += 1
            tqdm.write(f"  [{idx+1:03d}] [ERROR]: {str(e)[:80]}")
            # Safe fallback - never leave a row empty
            results.append({
                "status": "escalated",
                "product_area": "general",
                "response": "An error occurred processing this ticket. A human agent will follow up.",
                "justification": f"Processing error: {str(e)[:100]}",
                "request_type": "product_issue",
            })

    print(f"\n   Total errors: {errors}/{len(df)}")
    return pd.DataFrame(results)


def save_output(input_df: pd.DataFrame, results_df: pd.DataFrame) -> None:
    """
    Combine input + results and save to output.csv.

    Args:
        input_df   : Original ticket data
        results_df : Agent results

    Why combine?
        The output should include both the original ticket fields
        AND the agent's predictions - easier to review and debug.
    """
    # Reset indices to ensure proper alignment
    input_df = input_df.reset_index(drop=True)
    results_df = results_df.reset_index(drop=True)

    # Combine original columns with results
    output_df = pd.concat([input_df, results_df], axis=1)

    # Ensure output directory exists
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to: {OUTPUT_CSV}")
    print(f"   Rows: {len(output_df)}")
    print(f"   Columns: {list(output_df.columns)}")


# Summary Statistics
def print_summary(results_df: pd.DataFrame) -> None:
    """Print a summary of the agent's decisions."""
    print("\n" + "=" * 55)
    print("  PROCESSING SUMMARY")
    print("=" * 55)

    total = len(results_df)

    # Status breakdown
    status_counts = results_df["status"].value_counts()
    print("\nStatus Distribution:")
    for status in VALID_STATUSES:
        count = status_counts.get(status, 0)
        pct = count / total * 100
        bar = "#" * int(pct / 5)
        print(f"   {status:<12} {count:>3} ({pct:5.1f}%) {bar}")

    # Request type breakdown
    type_counts = results_df["request_type"].value_counts()
    print("\nRequest Type Distribution:")
    for rtype in VALID_REQUEST_TYPES:
        count = type_counts.get(rtype, 0)
        pct = count / total * 100
        print(f"   {rtype:<20} {count:>3} ({pct:5.1f}%)")

    # Product area breakdown
    area_counts = results_df["product_area"].value_counts()
    print("\nTop Product Areas:")
    for area, count in area_counts.head(8).items():
        pct = count / total * 100
        print(f"   {area:<25} {count:>3} ({pct:5.1f}%)")

    print(f"\n   Total tickets processed: {total}")
    print("=" * 55)


# Main Entry Point
def main() -> None:
    print("=" * 55)
    print("  HackerRank Orchestrate - Support Triage Agent")
    print("=" * 55)

    start_time = time.time()

    # Step 1: Startup checks
    run_startup_checks()

    # Step 2: Load tickets
    input_df = load_tickets()

    # Step 3: Process all tickets
    results_df = process_all_tickets(input_df)

    # Step 4: Save output
    save_output(input_df, results_df)

    # Step 5: Print summary
    print_summary(results_df)

    # Step 6: Print cost report
    tracker.print_summary()

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Done! Check: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()