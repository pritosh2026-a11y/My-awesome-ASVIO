"""
ASVIO - Brand Context Intelligence Engine

Input:
    data/normalized/normalized.json

Output:
    data/evidence/evidence.json

Purpose:
    Determine how a brand appears in a search result with robust 
    input validation, partial-failure handling, and audit reporting.

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
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

# Configure structured logging for runtime error messaging and audit trails
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("ASVIO_NLP")

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
    "best","good","great","excellent","easy","easy-to-use","popular",
    "recommended","recommend","powerful","affordable","flexible",
    "reliable","leading","top","efficient","helpful","strong","trusted",
}

# Words suggesting negative representation.
NEGATIVE_WORDS = {
    "bad","worst","expensive","difficult","hard","poor","slow","complicated","problem",
    "problems","issue","issues","flaw","flaws","limited","negative","complaint","complaints",
    "avoid","stopped","disappointed",
}

# Words suggesting recommendation.
RECOMMENDATION_WORDS = {
    "best","top","recommended","recommend","winner","ideal","great choice","good choice","popular choice",
}

# Words suggesting comparison.
COMPARISON_WORDS = {
    "vs","versus","compare","comparison","alternative","alternatives","compared",
}

# Words suggesting review / experience.
REVIEW_WORDS = {
    "review","reviews","experience","used","using","customer","customers","user","users",
}

# Words suggesting pricing.
PRICING_WORDS = {
    "price","pricing","cost","costs","cheap","expensive","free","plan","plans",
}

# Words suggesting features.
FEATURE_WORDS = {
    "feature","features","integration","integrations","automation","analytics",
    "dashboard","workflow","workflows","customization","customisation",
}

# Words suggesting security.
SECURITY_WORDS = {
    "security","secure","compliance","privacy","gdpr","soc","encryption","authentication",
}

ACTION_WORDS = {
    "supports", "support", "offers", "offer", "provides", "provide",
    "includes", "include", "integrates", "integrate", "helps", "help",
    "scales", "scale", "uses", "use", "allows", "allow",
}

VERB_WORDS = {
    "is", "are", "has", "have", "can", "offers", "provides",
    "supports", "includes", "integrates", "helps", "scales",
}

# ---------------------------------------------------------
# Updated functions 
# ---------------------------------------------------------
def detect_source_type(url: str, snippet: str) -> str:
    url_lower = url.lower()
    snippet_lower = snippet.lower()
    if any(k in url_lower for k in ["hubspot.com", "salesforce.com", "zendesk.com", "zoho.com", "monday.com", "quickbase.com", "softwareadvice.com"]):
        return "marketing_page"
    if any(k in url_lower for k in ["/pricing", "/plans", "/features", "/about"]):
        return "marketing_page"
    if any(word in snippet_lower for word in ["sign up", "free trial", "get started", "learn more", "product"]):
        return "marketing_page"
    if any(word in snippet_lower for word in ["review", "vs", "versus", "comparison", "alternative", "best vs"]):
        return "competitor_matrix"
    if "softwareadvice.com" in url_lower or "g2.com" in url_lower or "capterra.com" in url_lower:
        return "review_platform"
    return "website"
# ---------------------------------------------------------
# Content Type & AI Detection (Production-Ready)
# ---------------------------------------------------------

def detect_content_type(url: str, snippet: str) -> dict:
    url_lower = url.lower()
    snippet_lower = snippet.lower()

# Sponsored / AI / third-party signals
    is_sponsored = any(w in snippet_lower for w in ["sponsored", "ad", "advertisement", "promoted", "affiliate", "paid partnership"])
    is_ai_generated = any(w in snippet_lower for w in ["ai", "artificial intelligence", "generated by ai"])
    known_domains = {d.lower() for profile in BRAND_PROFILES.values() for d in profile.get("domains", [])}
    is_third_party = not any(d in url_lower for d in known_domains)
    return {
        "sponsored": is_sponsored,
        "ai_generated": is_ai_generated,
        "third_party": is_third_party,
        "confidence": 0.85 if is_sponsored or is_ai_generated else 0.65
    }

# ---------------------------------------------------------
# Brand Values Scoring (Production-Ready)
# ---------------------------------------------------------

def score_brand_values(text: str, brand: str) -> dict:
    normalized = normalize_text(text)
    tokens = tokenize(normalized)

#weighted scoring 
    security_score = 10 if any(w in normalized for w in SECURITY_WORDS) else 0
    easy_to_use_score = 8 if any(w in tokens for w in ["easy", "simple", "intuitive", "user-friendly"]) else 0
    excellence_score = 8 if any(w in tokens for w in ["best", "top", "excellent", "great"]) else 0
    longevity_score = 7 if any(w in tokens for w in ["long", "established", "reliable", "trusted"]) else 0

    after_sales_score = 6 if any(w in tokens for w in ["support", "service", "help", "customer service"]) else 0
    total_score = security_score + easy_to_use_score + excellence_score + longevity_score + after_sales_score
    return {
        "brand": brand,
        "security": security_score,
        "easy_to_use": easy_to_use_score,
        "excellence": excellence_score,
        "longevity": longevity_score,
        "after_sales": after_sales_score,
        "total_score": total_score,
        "strengths": [k for k, v in {
            "security": security_score,
            "easy_to_use": easy_to_use_score,
            "excellence": excellence_score,
            "longevity": longevity_score,
            "after_sales": after_sales_score
        }.items() if v > 3]
    }

# Brand Mention Frequency & Matrix (Production-Ready) (changes
# ---------------------------------------------------------
def build_mention_matrix(analyzed_results: list) -> dict:
    """Build frequency, position, and context matrix for all brands."""
    matrix = {}
    for result in analyzed_results:
        for mention in result.get("brand_mentions", []):
            brand = mention["brand"]
            if brand not in matrix:
                matrix[brand] = {
                    "count": 0,
                    "first_position": 999,
                    "last_position": 0,
                    "score_sum": 0,
                    "third_party_count": 0,
                    "owned_count": 0,
                    "sponsored_count": 0,
                    "ai_count": 0,
                    "total_prominence": 0,
                    "values": {
                        "security": 0,
                        "easy_to_use": 0,
                        "excellence": 0,
                        "longevity": 0,
                        "after_sales": 0
                    }
                }

            data = matrix[brand]
            data["count"] += 1
            data["first_position"] = min(data["first_position"], result.get("rank", 999))
            data["last_position"] = max(data["last_position"], result.get("rank", 0))
            data["score_sum"] += mention.get("confidence", 0.7)
            data["total_prominence"] += 1  # you can weight by claim prominence later

            if result.get("content_type", {}).get("third_party"):
                data["third_party_count"] += 1
            if result.get("content_type", {}).get("sponsored"):
                data["sponsored_count"] += 1
            if result.get("content_type", {}).get("ai_generated"):
                data["ai_count"] += 1

            # Merge brand values
            if "brand_values" in result:
                for k, v in result["brand_values"].items():
                    if v > 0:
                        data["values"][k] = data["values"].get(k, 0) + v

    # Normalize and compute final matrix
    for brand, data in matrix.items():
        data["avg_confidence"] = data["score_sum"] / data["count"] if data["count"] else 0
        data["third_party_ratio"] = round(data["third_party_count"] / data["count"] * 100, 1) if data["count"] else 0
        data["owned_ratio"] = round((data["count"] - data["third_party_count"]) / data["count"] * 100, 1) if data["count"] else 0

    return matrix

#end changes
def detect_competitors(text: str, brands_in_result: list) -> list:
    normalized = normalize_text(text)
    competitors = []
    for brand, profile in BRAND_PROFILES.items():
        if any(d.lower() in normalized for d in profile.get("domains", [])):
            competitors.append({"brand": brand, "confidence": 0.85, "is_main": False})
    return sorted(competitors, key=lambda c: c["confidence"], reverse=True)[:3]

# ---------------------------------------------------------
# TEXT NORMALIZATION & TOKENIZATION
# ---------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize text for matching.

    - Converts input to string safely.
    - Converts text to lowercase.
    - Collapses repeated whitespace.
    - Removes leading/trailing whitespace.
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
    Retained for backward compatibility and fast
    word-level membership checks.
    """
    text = normalize_text(text)
    return set(
        re.findall(r"\b\w+\b", text)
    )

# ---------------------------------------------------------
# KEYWORD / PHRASE REGEX COMPILATION
# ---------------------------------------------------------

def compile_word_regex(words: set) -> re.Pattern:
    """
    Compile a set of words/phrases into a single
    case-insensitive regex.
    Longer phrases are placed first so that expressions
    such as 'great choice' are considered before shorter
    expressions such as 'great'.

    re.escape() ensures configured terms are treated
    literally rather than as regex syntax.
    """

    sorted_words = sorted(
        words,
        key=len,
        reverse=True,
    )

    escaped = [
        re.escape(word)
        for word in sorted_words
    ]

    return re.compile(
        rf"(?<![A-Za-z0-9])({'|'.join(escaped)})(?![A-Za-z0-9])",
        re.IGNORECASE,
    )

# ---------------------------------------------------------
# COMPILED REGEX PATTERNS
# ---------------------------------------------------------

REGEX_POS = compile_word_regex(POSITIVE_WORDS)
REGEX_NEG = compile_word_regex(NEGATIVE_WORDS)
REGEX_REC = compile_word_regex(RECOMMENDATION_WORDS)
REGEX_COMP = compile_word_regex(COMPARISON_WORDS)
REGEX_REV = compile_word_regex(REVIEW_WORDS)
REGEX_PRICE = compile_word_regex(PRICING_WORDS)
REGEX_FEAT = compile_word_regex(FEATURE_WORDS)
REGEX_SEC = compile_word_regex(SECURITY_WORDS)


# ---------------------------------------------------------
# SENTENCE SPLITTING
# ---------------------------------------------------------

SENTENCE_SPLITTER = re.compile(
    r"(?<=[.!?])\s+"
)

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
# Brand detection
# ---------------------------------------------------------

def detect_brand_mentions(text: str) -> list:
    normalized = normalize_text(text)
    mentions = []

    for brand, profile in BRAND_PROFILES.items():
        matched_alias = None
        sorted_aliases = sorted(profile["aliases"],key=len,reverse=True,)

        for alias in sorted_aliases:
            alias_normalized = normalize_text(alias)
            pattern = rf"(?<![A-Za-z0-9]){re.escape(alias_normalized)}(?![A-Za-z0-9])"
            if re.search(pattern, normalized):
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

   # Ownership mapping (backward-compatible)
    is_first_party = ownership in ("first_party", "owned")

    if is_first_party:
        return {"type": "brand_owned", "confidence": 0.99}
    if any(word in normalized for word in COMPARISON_WORDS):
        return {"type": "competitor_comparison", "confidence": 0.90}
    if any(word in normalized for word in RECOMMENDATION_WORDS):
        return {"type": "recommended_option", "confidence": 0.82}
    if any(word in tokens for word in REVIEW_WORDS):
        return {"type": "review_or_experience", "confidence": 0.78}

    return {"type": "mentioned", "confidence": 0.70}

   

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

    total = positive_score + negative_score
    confidence = 0.50 if total == 0 else min(0.50 + (abs(positive_score - negative_score) / total) * 0.50, 0.99)

    return {
        "label": sentiment,
        "confidence": round(confidence, 2),
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
    if any(word in normalized for word in PRICING_WORDS):
        topics.append("pricing")
    if any(word in normalized for word in FEATURE_WORDS):
        topics.append("features")
    if any(word in normalized for word in SECURITY_WORDS):
        topics.append("security")
    if any(word in normalized for word in COMPARISON_WORDS):
        topics.append("comparison")
    if any(word in tokens for word in REVIEW_WORDS):
        topics.append("user_experience")
    if not topics:
        topics.append("general")
    return topics

# ---------------------------------------------------------
# Recommendation detection
# ---------------------------------------------------------

def detect_recommendation(text: str) -> dict:

    normalized = normalize_text(text)
    matches = [phrase for phrase in RECOMMENDATION_WORDS if phrase in normalized]
    return { 
        "detected": bool(matches), 
        "signals": matches,  
        "confidence": (0.85 if matches else 0.50),
    }


# ---------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------

def extract_claim_signals(text: str, brand: str,source_type: str = "website") -> list:
    """
    Extract grounded claim-like sentences containing
    the target brand.
    """
    sentences = SENTENCE_SPLITTER.split(str(text or ""))
    claims = []
    
   brand_profile = BRAND_PROFILES.get(brand, {})
    aliases = brand_profile.get("aliases", [brand.lower()])
    escaped_aliases = [re.escape(normalize_text(a)) for a in aliases]
    brand_pattern = re.compile(
        rf"(?<![A-Za-z0-9])({'|'.join(escaped_aliases)})(?![A-Za-z0-9])",
        re.IGNORECASE,
    )

    # Source config
    SOURCE_CONFIG = {
        "marketing_page": {"reliability": 0.95, "factuality_weight": 0.8},
        "user_review": {"reliability": 0.85, "factuality_weight": 0.6},
        "competitor_matrix": {"reliability": 0.7, "factuality_weight": 0.4},
        "landing_page": {"reliability": 0.9, "factuality_weight": 0.7},
        "review_platform": {"reliability": 0.8, "factuality_weight": 0.65},
        "website": {"reliability": 0.9, "factuality_weight": 0.75},
    }

    cfg = SOURCE_CONFIG.get(source_type, SOURCE_CONFIG["website"])

    for idx, sentence in enumerate(sentences):
        sentence_clean = sentence.strip()
        if not sentence_clean or len(sentence_clean) < 15:
            continue

        normalized_sent = normalize_text(sentence_clean)

        # Check for brand boundary match
        brand_match = brand_pattern.search(normalized_sent)
        if not brand_match:
            continue

        tokens = tokenize(normalized_sent)
        
        # Signal density filter
        has_signal = (
            any(w in normalized_sent for w in PRICING_WORDS) or
            any(w in normalized_sent for w in COMPARISON_WORDS) or
            any(w in normalized_sent for w in FEATURE_WORDS) or
            any(w in tokens for w in REVIEW_WORDS) or
            any(w in tokens for w in ACTION_WORDS) or
            (source_type in ["customer_review", "review_platform", "forum"] and any(w in tokens for w in REVIEW_WORDS))
        )
        if not has_signal:
            continue

        # Sentiments + Attribution
        pos_signals = [w for w in POSITIVE_WORDS if w in normalized_sent or w in tokens]
        neg_signals = [w for w in NEGATIVE_WORDS if w in normalized_sent or w in tokens]

        if len(pos_signals) > len(neg_signals):
            polarity = "positive"
        elif len(neg_signals) > len(pos_signals):
            polarity = "negative"
        else:
            polarity = "neutral"

        action_words = [w for w in tokens if w in ACTION_WORDS or w in VERB_WORDS]
        if any(w in tokens for w in ["is", "are", "feels", "seems", "appears"]):
            attribution = "brand_as_object"
        elif action_words:
            attribution = "brand_as_subject"
        else:
            attribution = "brand_passive"

        # Fact vs Opinion
        factuality = "mixed"
        if polarity == "neutral":
            factuality = "fact"
        elif len(pos_signals) > 3 or len(neg_signals) > 3:
            factuality = "opinion"

        # Categorize claim type
        claim_type = "general_assertion"
        if any(w in normalized_sent for w in PRICING_WORDS):
            claim_type = "pricing_or_cost"
        elif any(w in normalized_sent for w in COMPARISON_WORDS):
            claim_type = "competitor_comparison"
        elif any(w in normalized_sent for w in RECOMMENDATION_WORDS):
            claim_type = "endorsement_or_recommendation"
        elif any(w in tokens for w in REVIEW_WORDS) or source_type in ["customer_review", "review_platform", "forum"]:
            claim_type = "user_experience_or_review"
        elif any(w in normalized_sent for w in FEATURE_WORDS):
            claim_type = "feature_capability"
        elif action_words:
            claim_type = "action_description"

        # Context relevance scoring
        position_score = 100 if idx == 0 else 70 if brand_match.start() < len(normalized_sent) // 3 else 45
        sentence_length_bonus = min(len(sentence_clean) // 3, 20)
        action_density = len(action_words) / max(len(tokens), 1)
        verb_density = len([w for w in tokens if w in VERB_WORDS]) / max(len(tokens), 1)
        action_bonus = action_density * 25
        verb_bonus = verb_density * 15

        relevance_score = position_score + (cfg["reliability"] * 30) + (cfg["factuality_weight"] if factuality == "fact" else 0.3) * 30 + action_bonus + verb_bonus + sentence_length_bonus
        prominence = min(relevance_score, 100)
        confidence = max(0.65, prominence / 120)

        claims.append(
            {
                "text": sentence_clean,
                "claim_type": claim_type,
                "polarity": polarity,
                "attribution": attribution,
                "factuality": factuality,
                "prominence": "core" if is_core_mention else "incidental",
                "signals": {"positive": pos_signals,"negative": neg_signals,},
                "confidence": 0.85 if is_core_mention else 0.65,
                "source_type": source_type,
                "is_core": prominence > 75,
            }
        )

    # Prioritize core and high-confidence claims
    sorted_claims = sorted(
        claims,
        key=lambda c: (c["prominence"] == "core", c["confidence"]),
        reverse=True,
    )

    return sorted_claims[:5]


# ---------------------------------------------------------
# VALIDATION & PIPELINE LOGIC & NORMALIZATION 
# ---------------------------------------------------------
def validate_input_schema(data: Any) -> Tuple[bool, List[str]]:
    """Strict schema validation for input data payload."""
    errors = []
    if not isinstance(data, dict):
        return False, ["Root payload must be a JSON object."]
    if "records" not in data:
        errors.append("Missing mandatory 'records' key in root payload.")
    elif not isinstance(data["records"], list):
        errors.append("Field 'records' must be a list.")
    return len(errors) == 0, errors


#1. Full docstring + rich analysis (highest priority)


def analyze_result(result: dict, audit_stats: dict) -> dict | None:
"""Full validation layer for a single search result entry.

    Runs validate_result() for fast checks + full brand-context intelligence analysis.
    Returns rich validated dict OR None if invalid.
    """
    validated = validate_result(result)
    if validated is None:
        audit_stats["invalid_result_structures"] += 1
        return None

# === BRAND DETECTION & ANALYSIS + DOMAIN VALIDATION ==
   
    brand_mentions = detect_brand_mentions(
        f"{validated['title']}. {validated['snippet']}",
         validated['url']
    )

    brand_analyses = []
    if not brand_mentions:
        audit_stats["missing_source_metadata_warnings"] += 1
        brand_analyses = [{
            "brand": "unknown",
            "matched_alias": "unknown",
            "confidence": 0.0,
            "signals": "No brand mention detected in this result"
        }]
    else:
        for mention in brand_mentions:
            brand_analyses.append({
                "brand": mention["brand"],
                "matched_alias": mention["matched_alias"],
                "confidence": mention["confidence"],
                is_first_party": mention.get("is_first_party", False),
            })
# === CONTENT TYPE & AI DETECTION ===
    content_type = detect_content_type(validated['url'], validated['snippet'])

# === BRAND VALUES SCORING ===
    brand_values = {mention["brand"]: score_brand_values(validated['snippet'], mention["brand"]) for mention in brand_mentions}

                
    role = classify_role(
        validated['snippet'], 
        brand_mentions[0]['brand'] if brand_mentions else 'Unknown',
        'owned' if any(m.get('is_first_party', False) for m in brand_mentions) else 'unknown'
    )

    sentiment = classify_sentiment(validated['snippet'], brand_mentions[0]['brand'] if brand_mentions else 'Unknown')
    topics = detect_topics(validated['snippet'])
    rec = detect_recommendation(validated['snippet'])
    claims = extract_claim_signals(validated['snippet'], brand_mentions[0]['brand'] if brand_mentions else 'Unknown')

# AUTO SOURCE TYPE 

    source_type = detect_source_type(validated['url'], validated['snippet'])
          
    return {
        **validated,
        "brand_analyses": brand_analyses,
        "content_type": content_type,
        "brand_values": brand_values,
        "role_classification": role,
        "sentiment": sentiment,
        "topics": topics,
        "recommendation": rec,
        "competitors": competitors,
        "claims": claims,
        "source_type": source_type,
    }

def validate_result(result: dict) -> dict | None:
    """Fast structural/type validation (kept for reference - now only used by analyze_result)."""
    if not isinstance(result, dict):
        return None

    source_id = result.get("source_id")
    rank = result.get("rank")
    domain = result.get("domain", "")
    title = result.get("title", "")
    snippet = result.get("snippet", "")
    url = result.get("url", "")
    if not url:
        return None

    combined_text = f"{title}. {snippet}".strip()
    if not combined_text or combined_text == ".":
        return None

    # Basic field-type guard (no full analysis)
    if (
        not isinstance(source_id, (str, int))
        or not isinstance(rank, (int, float))
        or not isinstance(title, str)
        or not isinstance(snippet, str)
        or not isinstance(url, str)
        or not isinstance(domain, str)
    ):
        return None

    return {
        "source_id": source_id,
        "rank": rank,
        "url": url,
        "domain": domain,
        "title": title,
        "snippet": snippet,
    }

# Updated process_normalized_data (integrated with analyze_result)

def process_normalized_data(data: dict) -> dict:
    """Processes normalized data payload with full validation, recovery, and audit tracking.
    Now uses the new analyze_result internally"""
    is_valid, validation_errors = validate_input_schema(data)
    if not is_valid:
        error_msg = "; ".join(validation_errors)
        logger.error("Schema validation failed: %s", error_msg)
        raise ValueError(f"Invalid input schema: {error_msg}")

    audit_stats = {
        "total_records_processed": 0,
        "total_results_processed": 0,
        "invalid_result_structures": 0,
        "missing_url_warnings": 0,
        "empty_text_warnings": 0,
        "malformed_records_skipped": 0,
        "missing_source_metadata_warnings": 0,
    }

    evidence_records = []
    records = data.get("records", [])

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            logger.warning("Skipping non-dict record at index %d", idx)
            audit_stats["malformed_records_skipped"] += 1
            continue

        query = record.get("query", "")
        if not query:
            logger.warning("Record at index %d has an empty or missing 'query' string", idx)

        results = record.get("results", [])
        if not isinstance(results, list):
            logger.warning("Record query '%s' has non-list 'results' field. Defaulting to empty list.", query)
            results = []

        analyzed_results = []
        for result in results:
            audit_stats["total_results_processed"] += 1
            analyzed = analyze_result(result, audit_stats)
            if analyzed is not None:
                analyzed_results.append(analyzed)

        evidence_records.append({
            "query": query,
            "results": analyzed_results,
        })
        audit_stats["total_records_processed"] += 1

    logger.info("Processing complete. Audit Summary: %s", json.dumps(audit_stats))


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
        description=( "Analyze normalized ASVIO search data.")
    )

    parser.add_argument("--input", default="data/normalized/normalized.json",help="Normalized JSON input",)
    parser.add_argument("--output",default= "data/evidence/evidence.json",help="Evidence JSON output",)
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    try:   
        data = load_json(input_path)
        evidence = process_normalized_data(data)
        save_json(evidence,output_path)

        print("ASVIO NLP processing complete.")
        print(f"Evidence output: {output_path}")

    except Exception as e:
            logger.error("Execution failed: %s", str(e), exc_info=True)
            sys.exit(1)

if __name__ == "__main__":
    main()
