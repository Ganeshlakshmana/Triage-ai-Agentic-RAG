"""
classifier.py - Rules-Based Safety and Classification Engine

Responsibility:
    Apply deterministic rules BEFORE and AFTER the LLM to:
    1. Detect high-risk tickets that must always be escalated
    2. Classify request type (product_issue/bug/feature_request/invalid)
    3. Validate and sanitize LLM output fields
    4. Override LLM decisions when safety rules demand it

Why a separate classifier?
    LLMs are probabilistic - they can be wrong, overconfident,
    or manipulated by malicious input in tickets.

    For high-stakes decisions (fraud, security, account breach),
    we NEVER rely on the LLM alone. Hard rules act as a safety net.

Principle: Rules > AI for safety-critical decisions.
    The LLM handles nuance and language.
    The classifier handles non-negotiable rules.

Architecture pattern: "LLM sandwich"
    [Rules pre-check] -> [LLM generates response] -> [Rules post-check]
    This ensures safety at both entry and exit of the LLM.
"""

import re
from config import (
    ESCALATION_KEYWORDS,
    VALID_STATUSES,
    VALID_REQUEST_TYPES,
    VALID_COMPANIES,
)


# Escalation Rules

# High-risk patterns that ALWAYS escalate
# These go beyond simple keywords - they're regex patterns
# that catch variations ("site's down", "sites down", "site is down")
HIGH_RISK_PATTERNS = [
    r"site\s*(is\s*)?(down|not\s*working|inaccessible)",
    r"(cannot|can't|unable to)\s*access\s*(the\s*)?(site|platform|system|account)",
    r"(hacked|compromised|unauthorized\s*access)",
    r"(data\s*breach|data\s*leak|leaked)",
    r"(fraud|fraudulent|scam|phishing)",
    r"(stolen|lost)\s*(card|account|credentials|password)",
    r"(legal|lawsuit|court|attorney|lawyer)",
    r"(all\s*pages?|everything)\s*(is\s*)?(down|broken|not\s*working)",
    r"emergency|urgent\s*security",
    r"account\s*(locked|suspended|banned|terminated)",
]

# Patterns that suggest the ticket is out of scope / invalid
OUT_OF_SCOPE_PATTERNS = [
    r"^(hi|hello|hey|thanks|thank you|ok|okay|yes|no|sure)[\s.!]*$",
    r"^(what is|who is|when was|where is)\s+\w+",  # general knowledge questions
    r"(weather|stock\s*price|sports|movie|recipe|joke)",
    r"(iron\s*man|marvel|disney|netflix|game\s*of\s*thrones)",
]


def should_escalate(issue: str, subject: str = "") -> tuple[bool, str]:
    """
    Check if a ticket should be escalated based on hard rules.
    Returns: Tuple of (should_escalate: bool, reason: str)
    """
    combined = f"{subject} {issue}".lower().strip()

    for keyword in ESCALATION_KEYWORDS:
        if keyword.lower() in combined:
            return True, f"Escalation keyword detected: '{keyword}'"

    for pattern in HIGH_RISK_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return True, f"High-risk pattern matched: '{pattern}'"

    return False, ""


def get_escalation_request_type(issue: str, subject: str = "") -> str:
    """
    When a ticket is escalated by hard rules, determine the
    correct request_type rather than always defaulting to product_issue.

    Site outages + security issues -> bug
    Everything else -> product_issue
    """
    combined = f"{subject} {issue}".lower()
    outage_signals = [
        "site is down", "not working", "inaccessible",
        "stopped working", "all requests", "can't load",
    ]
    for signal in outage_signals:
        if signal in combined:
            return "bug"
    security_signals = ["hacked", "breach", "unauthorized", "data leak", "vulnerability"]
    for signal in security_signals:
        if signal in combined:
            return "bug"
    return "product_issue"


def is_out_of_scope(issue: str, subject: str = "") -> tuple[bool, str]:
    """
    Check if a ticket is clearly out of scope or invalid.

    Args:
        issue   : Ticket body
        subject : Subject line

    Returns:
        Tuple of (is_invalid: bool, reason: str)

    Out-of-scope tickets get status="replied" with a polite
    "this is outside my capabilities" message - NOT escalated.
    Escalation is for real problems that need human attention.
    Invalid requests don't need a human agent.
    """
    issue_clean = issue.strip()

    # Very short tickets (greetings, acknowledgments)
    if len(issue_clean) < 10:
        return True, "Ticket too short to be a valid support request"

    # Check out-of-scope patterns
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, issue_clean, re.IGNORECASE):
            return True, f"Out-of-scope pattern matched: '{pattern}'"

    return False, ""


# Request Type Classification

# Keywords that hint at each request type
REQUEST_TYPE_HINTS = {
    "bug": [
        "bug", "error", "broken", "not working", "doesn't work",
        "crash", "exception", "500", "404", "failed", "issue with",
        "glitch", "problem with", "keeps failing",
        # Site/service outages are bugs
        "site is down", "down", "inaccessible", "not accessible",
        "stopped working", "not loading", "can't load",
    ],
    "feature_request": [
        "feature", "request", "wish", "would be nice", "suggestion",
        "can you add", "please add", "enhancement", "improve",
        "would love", "looking for", "need ability to",
    ],
    "invalid": [
        "thank you", "thanks", "hello", "hi there", "good morning",
        "ok", "noted", "understood", "got it",
    ],
}


def classify_request_type(issue: str, subject: str = "") -> str:
    """
    Classify the type of support request.

    Args:
        issue   : Ticket body
        subject : Subject line

    Returns:
        One of: "product_issue", "feature_request", "bug", "invalid"

    This provides a HINT to the LLM - the LLM may override
    this classification with better reasoning.
    But if the LLM returns an invalid value, we fall back to this.

    Priority: invalid > bug > feature_request > product_issue
    (more specific types take precedence)
    """
    combined = f"{subject} {issue}".lower()

    # Check invalid first (greetings, acknowledgments)
    for hint in REQUEST_TYPE_HINTS["invalid"]:
        if hint in combined and len(issue.strip()) < 30:
            return "invalid"

    # Check bug patterns
    bug_score = sum(1 for h in REQUEST_TYPE_HINTS["bug"] if h in combined)

    # Check feature request patterns
    feature_score = sum(
        1 for h in REQUEST_TYPE_HINTS["feature_request"] if h in combined
    )

    if bug_score > feature_score and bug_score > 0:
        return "bug"
    elif feature_score > bug_score and feature_score > 0:
        return "feature_request"
    else:
        return "product_issue"  # default - most support tickets are product issues


# Infer Company from Content

COMPANY_SIGNALS = {
    "HackerRank": [
        "hackerrank", "hacker rank", "test", "assessment", "candidate",
        "interview", "screening", "recruiter", "proctoring", "coding test",
        "skill test", "invite", "hackerrank for work",
    ],
    "Claude": [
        "claude", "anthropic", "conversation", "chat", "prompt",
        "claude.ai", "claude pro", "claude team", "artifact",
        "claude api", "context window",
    ],
    "Visa": [
        "visa", "card", "payment", "transaction", "credit card",
        "debit card", "visa card", "atm", "merchant", "chargeback",
        "traveller", "foreign transaction",
    ],
}


def infer_company(issue: str, subject: str = "", stated_company: str = "") -> str:
    """
    Infer or confirm the company from ticket content.

    Args:
        issue          : Ticket body
        subject        : Subject line
        stated_company : Company field from CSV (may be "None")

    Returns:
        Company name string or "Unknown"

    Why infer?
        The CSV company field can be "None" for cross-domain tickets.
        We try to infer from content keywords.
        If we can't infer, we return "Unknown" and search all companies.
    """
    # If stated company is valid, trust it
    if stated_company and stated_company.strip().lower() != "none":
        for valid in VALID_COMPANIES:
            if stated_company.strip().lower() == valid.lower():
                return valid

    # Try to infer from content
    combined = f"{subject} {issue}".lower()
    scores = {}

    for company, signals in COMPANY_SIGNALS.items():
        score = sum(1 for signal in signals if signal in combined)
        if score > 0:
            scores[company] = score

    if scores:
        # Return the company with the most signal matches
        return max(scores, key=scores.get)

    return "Unknown"


# Output Validation

def validate_and_fix_output(llm_output: dict) -> dict:
    """
    Validate LLM output and fix any invalid values.

    Args:
        llm_output : Dict with keys: status, product_area,
                     response, justification, request_type

    Returns:
        Cleaned dict with all values guaranteed to be valid.

    Why validate?
        LLMs occasionally produce values outside the allowed set.
        E.g. status="Escalated" (wrong case) or
             request_type="complaint" (not in our schema).
        We fix these rather than crashing.

    Principle: Be strict about what you accept,
               but graceful about what you return.
    """
    output = llm_output.copy()

    # Fix status
    status = output.get("status", "").lower().strip()
    if status not in VALID_STATUSES:
        # Default to escalated if unclear - safer than wrong reply
        output["status"] = "escalated"

    # Fix request_type
    request_type = output.get("request_type", "").lower().strip()
    if request_type not in VALID_REQUEST_TYPES:
        output["request_type"] = "product_issue"  # safe default

    # Ensure required fields exist
    if not output.get("response"):
        output["response"] = "Unable to generate a response. Please contact support."

    if not output.get("justification"):
        output["justification"] = "No justification provided."

    if not output.get("product_area"):
        output["product_area"] = "general"

    return output


# Quick Test
if __name__ == "__main__":
    print("Testing classifier...\n")

    # Test escalation detection
    test_cases = [
        ("site is down & none of the pages are accessible", "", True),
        ("My Visa card was stolen last night", "Card stolen", True),
        ("How do I add extra time for a candidate?", "", False),
        ("I want to delete my account", "", False),
        ("Thank you for helping me", "", False),
        ("What is the name of the actor in Iron Man?", "Urgent", False),
    ]

    print("=" * 55)
    print("Escalation Detection Tests")
    print("=" * 55)
    for issue, subject, expected in test_cases:
        escalate, reason = should_escalate(issue, subject)
        status = "[OK]" if escalate == expected else "[ERROR]"
        print(f"{status} escalate={escalate} | '{issue[:45]}...'")
        if reason:
            print(f"   Reason: {reason}")

    # Test request type classification
    print("\n" + "=" * 55)
    print("Request Type Classification")
    print("=" * 55)
    type_tests = [
        ("The test is broken and candidates can't submit", "bug"),
        ("I would love a feature to bulk export results", "feature_request"),
        ("How do I invite candidates to a test?", "product_issue"),
        ("Thank you!", "invalid"),
    ]
    for issue, expected in type_tests:
        result = classify_request_type(issue)
        status = "[OK]" if result == expected else "[WARNING]"
        print(f"{status} {result:<20} | '{issue[:45]}'")

    # Test company inference
    print("\n" + "=" * 55)
    print("Company Inference")
    print("=" * 55)
    company_tests = [
        ("My HackerRank assessment won't load", "", "None"),
        ("I need to delete my Claude conversation", "", "None"),
        ("My Visa card was declined abroad", "", "None"),
        ("The site is down", "", "None"),
    ]
    for issue, subject, stated in company_tests:
        result = infer_company(issue, subject, stated)
        print(f"  '{issue[:45]}' -> {result}")

    print("\n[OK] Classifier ready!")