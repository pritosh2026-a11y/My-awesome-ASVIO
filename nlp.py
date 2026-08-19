"""
ASVIO - Brand Context Intelligence Engine

Input:
    data/normalized/normalized.json

Output:
    data/evidence/evidence.json

Purpose:
    Determine how a brand appears in a search result.

Current NLP layer:
    1. Brand mention detection
    2. Brand ownership detection
    3. Source type classification
    4. Brand role classification
    5. Basic sentiment
    6. Topic/context detection
    7. Claim extraction

This version is intentionally lightweight and deterministic.
An LLM can be added later as an ambiguity-resolution layer.
"""

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BRAND_PROFILES = {
    "HubSpot": {
        "domains": [
            "hubspot.com",
        ],
        "aliases": [
            "hubspot",
            "hubspot crm",
        ],
    },

    "Salesforce": {
        "domains": [
            "salesforce.com",
        ],
        "aliases": [
            "salesforce",
            "salesforce crm",
        ],
    },

    "Zendesk": {
        "domains": [
            "zendesk.com",
        ],
        "aliases": [
            "zendesk",
        ],
    },

    "Zoho": {
        "domains": [
            "zoho.com",
        ],
        "aliases": [
            "zoho",
            "zoho crm",
        ],
    },

    "Monday.com": {
        "domains": [
            "monday.com",
        ],
        "aliases": [
            "monday.com",
            "monday crm",
            "monday",
        ],
    },

    "Quickbase": {
        "domains": [
            "quickbase.com",
        ],
        "aliases": [
            "quickbase",
        ],
    },

    "Software Advice": {
        "domains": [
            "softwareadvice.com",
        ],
        "aliases": [
            "software advice",
        ],
    },
}


# Words suggesting positive representation.
POSITIVE_WORDS = {
    "best",
    "good",
    "great",
    "excellent",
    "easy",
    "easy-to-use",
    "popular",
    "recommended",
    "recommend",
    "powerful",
    "affordable",
    "flexible",
    "reliable",
    "leading",
    "top",
    "efficient",
    "helpful",
    "strong",
    "trusted",
}

# Words suggesting negative representation.
NEGATIVE_WORDS = {
    "bad",
    "worst",
    "expensive",
    "difficult",
    "hard",
    "poor",
    "slow",
    "complicated",
    "problem",
    "problems",
    "issue",
    "issues",
    "flaw",
    "flaws",
    "limited",
    "negative",
    "complaint",
    "complaints",
    "avoid",
    "stopped",
    "disappointed",
}

# Words suggesting recommendation.
RECOMMENDATION_WORDS = {
    "best",
    "top",
    "recommended",
    "recommend",
    "winner",
    "ideal",
    "great choice",
    "good choice",
    "popular choice",
}

# Words suggesting comparison.
COMPARISON_WORDS = {
    "vs",
    "versus",
    "compare",
    "comparison",
    "alternative",
    "alternatives",
    "compared",
}

# Words suggesting review / experience.
REVIEW_WORDS = {
    "review",
    "reviews",
    "experience",
    "used",
    "using",
    "customer",
    "customers",
    "user",
    "users",
}

# Words suggesting pricing.
PRICING_WORDS = {
    "price",
    "pricing",
    "cost",
    "costs",
    "cheap",
    "expensive",
    "free",
    "plan",
    "plans",
}

# Words suggesting features.
FEATURE_WORDS = {
    "feature",
    "features",
    "integration",
    "integrations",
    "automation",
    "analytics",
    "dashboard",
    "workflow",
    "workflows",
    "customization",
    "customisation",
}

# Words suggesting security.
SECURITY_WORDS = {
    "security",
    "secure",
    "compliance",
    "privacy",
    "gdpr",
    "soc",
    "encryption",
    "authentication",
}


# ---------------------------------------------------------
# File helpers
# ---------------------------------------------------------

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ---------------------------------------------------------
# Text helpers
# ---------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize text for matching.
    """

    text = str(text or "").lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def tokenize(text: str) -> set:
    """
    Convert text into a simple word set.
    """

    text = normalize_text(text)

    return set(
        re.findall(
            r"[a-z0-9]+(?:-[a-z0-9]+)?",
            text,
        )
    )


# ---------------------------------------------------------
# Brand detection
# ---------------------------------------------------------

def detect_brand_mentions(
    text: str,
) -> list:
    """
    Find brands explicitly mentioned in title/snippet.
    """

    normalized = normalize_text(text)

    mentions = []

    for brand, profile in BRAND_PROFILES.items():

        matched_alias = None

        for alias in profile["aliases"]:

            alias_normalized = normalize_text(alias)

            if alias_normalized in normalized:
                matched_alias = alias
                break

        if matched_alias:

            mentions.append(
                {
                    "brand": brand,
                    "matched_alias": matched_alias,
                    "confidence": 0.95,
                }
            )

    return mentions


# ---------------------------------------------------------
# Ownership detection
# ---------------------------------------------------------

def domain_matches_brand(
    domain: str,
    brand: str,
) -> bool:
    """
    Determine whether the result's domain belongs to
    the brand.
    """

    domain = normalize_text(domain)

    profile = BRAND_PROFILES.get(
        brand,
        {},
    )

    for brand_domain in profile.get(
        "domains",
        [],
    ):

        brand_domain = normalize_text(
            brand_domain
        )

        if (
            domain == brand_domain
            or domain.endswith(
                "." + brand_domain
            )
        ):
            return True

    return False


def determine_ownership(
    domain: str,
    brand: str,
) -> dict:

    if domain_matches_brand(
        domain,
        brand,
    ):

        return {
            "type": "owned",
            "confidence": 0.99,
        }

    return {
        "type": "third_party",
        "confidence": 0.90,
    }


# ---------------------------------------------------------
# Source classification
# ---------------------------------------------------------

def classify_source_type(
    domain: str,
) -> str:

    domain = normalize_text(domain)

    if "reddit.com" in domain:
        return "community"

    if (
        "quora.com" in domain
        or "forums." in domain
    ):
        return "community"

    if (
        "g2.com" in domain
        or "capterra.com" in domain
        or "softwareadvice.com" in domain
    ):
        return "review_platform"

    if (
        "forbes.com" in domain
        or "techcrunch.com" in domain
        or "wsj.com" in domain
        or "nytimes.com" in domain
    ):
        return "media"

    if (
        "google.com" in domain
        or "bing.com" in domain
    ):
        return "search_engine"

    return "website"


# ---------------------------------------------------------
# Role classification
# ---------------------------------------------------------

def classify_role(
    text: str,
    brand: str,
    ownership: str,
) -> dict:
    """
    Determine how the brand is positioned in the result.
    """

    normalized = normalize_text(text)
    tokens = tokenize(normalized)

    # Owned page.
    if ownership == "owned":

        return {
            "type": "brand_owned",
            "confidence": 0.99,
        }

    # Comparison.
    if any(
        word in normalized
        for word in COMPARISON_WORDS
    ):

        return {
            "type": "competitor_comparison",
            "confidence": 0.90,
        }

    # Recommendation.
    if any(
        word in normalized
        for word in RECOMMENDATION_WORDS
    ):

        return {
            "type": "recommended_option",
            "confidence": 0.82,
        }

    # Review / experience.
    if any(
        word in tokens
        for word in REVIEW_WORDS
    ):

        return {
            "type": "review_or_experience",
            "confidence": 0.78,
        }

    # Otherwise it is simply mentioned.
    return {
        "type": "mentioned",
        "confidence": 0.70,
    }


# ---------------------------------------------------------
# Sentiment
# ---------------------------------------------------------

def classify_sentiment(
    text: str,
    brand: str,
) -> dict:
    """
    Lightweight sentiment based on words appearing
    in the result context.

    This is NOT intended to replace a full sentiment
    model. It provides a fast baseline.
    """

    normalized = normalize_text(text)
    tokens = tokenize(normalized)

    positive_matches = sorted(
        word
        for word in POSITIVE_WORDS
        if word in normalized or word in tokens
    )

    negative_matches = sorted(
        word
        for word in NEGATIVE_WORDS
        if word in normalized or word in tokens
    )

    positive_score = len(
        positive_matches
    )

    negative_score = len(
        negative_matches
    )

    if positive_score > negative_score:

        sentiment = "positive"

    elif negative_score > positive_score:

        sentiment = "negative"

    else:

        sentiment = "neutral"

    total = (
        positive_score
        + negative_score
    )

    if total == 0:

        confidence = 0.50

    else:

        confidence = min(
            0.50
            + (
                abs(
                    positive_score
                    - negative_score
                )
                / total
            )
            * 0.50,
            0.99,
        )

    return {
        "label": sentiment,
        "confidence": round(
            confidence,
            2,
        ),
        "positive_signals": positive_matches,
        "negative_signals": negative_matches,
    }


# ---------------------------------------------------------
# Topic detection
# ---------------------------------------------------------

def detect_topics(
    text: str,
) -> list:

    normalized = normalize_text(text)
    tokens = tokenize(normalized)

    topics = []

    if any(
        word in normalized
        for word in PRICING_WORDS
    ):
        topics.append("pricing")

    if any(
        word in normalized
        for word in FEATURE_WORDS
    ):
        topics.append("features")

    if any(
        word in normalized
        for word in SECURITY_WORDS
    ):
        topics.append("security")

    if any(
        word in normalized
        for word in COMPARISON_WORDS
    ):
        topics.append("comparison")

    if any(
        word in tokens
        for word in REVIEW_WORDS
    ):
        topics.append("user_experience")

    if not topics:
        topics.append("general")

    return topics


# ---------------------------------------------------------
# Recommendation detection
# ---------------------------------------------------------

def detect_recommendation(
    text: str,
) -> dict:

    normalized = normalize_text(text)

    matches = [
        phrase
        for phrase in RECOMMENDATION_WORDS
        if phrase in normalized
    ]

    return {
        "detected": bool(matches),
        "signals": matches,
        "confidence": (
            0.85 if matches else 0.50
        ),
    }


# ---------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------

def extract_claim_signals(
    text: str,
    brand: str,
) -> list:
    """
    Extract simple claim-like sentences containing
    the target brand.

    This is deliberately conservative.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        str(text or ""),
    )

    claims = []

    brand_lower = brand.lower()

    for sentence in sentences:

        sentence_clean = sentence.strip()

        if not sentence_clean:
            continue

        if brand_lower not in sentence_clean.lower():
            continue

        claims.append(
            {
                "text": sentence_clean,
                "type": "brand_context",
            }
        )

    return claims[:5]


# ---------------------------------------------------------
# Analyze one result
# ---------------------------------------------------------

def analyze_result(
    result: dict,
) -> dict:

    source_id = result.get(
        "source_id"
    )

    rank = result.get(
        "rank"
    )

    domain = result.get(
        "domain",
        "",
    )

    title = result.get(
        "title",
        "",
    )

    snippet = result.get(
        "snippet",
        "",
    )

    url = result.get(
        "url",
        "",
    )

    combined_text = (
        f"{title}. {snippet}"
    )

    mentions = detect_brand_mentions(
        combined_text
    )

    source_type = classify_source_type(
        domain
    )

    brand_analyses = []

    for mention in mentions:

        brand = mention["brand"]

        ownership = determine_ownership(
            domain,
            brand,
        )

        role = classify_role(
            combined_text,
            brand,
            ownership["type"],
        )

        sentiment = classify_sentiment(
            combined_text,
            brand,
        )

        topics = detect_topics(
            combined_text
        )

        recommendation = detect_recommendation(
            combined_text
        )

        claims = extract_claim_signals(
            combined_text,
            brand,
        )

        brand_analyses.append(
            {
                "brand": brand,

                "mention": {
                    "detected": True,
                    "matched_alias": mention[
                        "matched_alias"
                    ],
                    "confidence": mention[
                        "confidence"
                    ],
                },

                "ownership": ownership,

                "source": {
                    "domain": domain,
                    "type": source_type,
                },

                "role": role,

                "sentiment": sentiment,

                "topics": topics,

                "recommendation": recommendation,

                "claims": claims,
            }
        )

    return {
        "source_id": source_id,
        "rank": rank,
        "url": url,
        "domain": domain,
        "title": title,
        "brand_analyses": brand_analyses,
    }


# ---------------------------------------------------------
# Process normalized data
# ---------------------------------------------------------

def process_normalized_data(
    data: dict,
) -> dict:

    evidence_records = []

    records = data.get(
        "records",
        [],
    )

    for record in records:

        query = record.get(
            "query",
            "",
        )

        results = record.get(
            "results",
            [],
        )

        analyzed_results = []

        for result in results:

            analyzed = analyze_result(
                result
            )

            analyzed_results.append(
                analyzed
            )

        evidence_records.append(
            {
                "query": query,
                "results": analyzed_results,
            }
        )

    return {
        "project": "ASVIO",
        "version": "0.1.0",
        "stage": "brand_context_intelligence",
        "records": evidence_records,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze normalized ASVIO search data."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "data/normalized/normalized.json"
        ),
        help="Normalized JSON input",
    )

    parser.add_argument(
        "--output",
        default=(
            "data/evidence/evidence.json"
        ),
        help="Evidence JSON output",
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    data = load_json(
        input_path
    )

    evidence = process_normalized_data(
        data
    )

    save_json(
        evidence,
        output_path
    )

    print(
        "ASVIO NLP processing complete."
    )

    print(
        f"Evidence output: {output_path}"
    )


if __name__ == "__main__":
    main()
