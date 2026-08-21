"""
ASVIO - Search Visibility Data Processor (Optimized)
"""

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

# ---------------------------------------------------------
# Configuration & Pre-computation
# ---------------------------------------------------------

TRACKING_PARAMETERS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", 
    "utm_content", "gclid", "fbclid", "msclkid", "ref",
}

# CHANGE START: Added brand domain lookup profile for ownership matching
BRAND_PROFILES = {
    "HubSpot": {
        "domains": ["hubspot.com"],
        "aliases": ["hubspot", "hubspot crm"],
    },
    "Salesforce": {
        "domains": ["salesforce.com"],
        "aliases": ["salesforce", "salesforce crm"],
    },
    "Zendesk": {
        "domains": ["zendesk.com"],
        "aliases": ["zendesk"],
    },
    "Zoho": {
        "domains": ["zoho.com"],
        "aliases": ["zoho", "zoho crm"],
    },
    "Monday.com": {
        "domains": ["monday.com"],
        "aliases": ["monday.com", "monday crm", "monday"],
    },
    "Quickbase": {
        "domains": ["quickbase.com"],
        "aliases": ["quickbase"],
    },
    "Software Advice": {
        "domains": ["softwareadvice.com"],
        "aliases": ["software advice"],
    },
}

BRAND_ALIASES = {
    "hubspot": "HubSpot",
    "salesforce": "Salesforce",
    "zendesk": "Zendesk",
    "zoho": "Zoho",
    "monday": "Monday.com",
    "quickbase": "Quickbase",
    "softwareadvice": "Software Advice",
}
#Knowledge base configuration for source classification
HIGH_AUTHORITY_DOMAINS = {
    "news": ["wsj.com", "nytimes.com", "reuters.com", "bloomberg.com", "ft.com", "forbes.com", "techcrunch.com"],
    "review_platforms": ["g2.com", "capterra.com", "softwareadvice.com", "trustpilot.com", "getapp.com", "trustradius.com"],
    "social_and_forums": ["reddit.com", "quora.com", "stackoverflow.com", "linkedin.com", "twitter.com", "x.com"],
    "aggregators_and_markets": ["producthunt.com", "appsumo.com", "github.com"]
}

PAGE_PATH_PATTERNS = {
    "customer_review": [r"/review/", r"/reviews/", r"/customer-reviews/", r"/user-reviews/"],
    "comparison_site": [r"/vs/", r"/versus/", r"/compare/", r"/comparison/", r"/alternatives/"],
    "editorial": [r"/blog/", r"/article/", r"/insights/", r"/news/", r"/resources/"],
    "forum": [r"/community/", r"/forum/", r"/forums/", r"/discussion/", r"/r/", r"/q/"],
    "marketplace": [r"/marketplace/", r"/integrations/", r"/app-store/"]
}

FEATURE_WORDS = {"feature", "features", "integration", "integrations", "automation", "analytics", "dashboard", "workflow"}
PRICING_WORDS = {"price", "pricing", "cost", "costs", "cheap", "expensive", "free", "plan", "plans"}
REVIEW_WORDS = {"review", "reviews", "experience", "used", "using", "customer", "user"}
COMPARISON_WORDS = {"vs", "versus", "compare", "comparison", "alternative", "alternatives"}


# OPTIMIZATION: Compile regexes once at startup with word boundaries (\b)
# This prevents "monday" from matching the normal word "monday" inside a snippet.
COMPILED_BRANDS = {
    re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE): brand_name
    for alias, brand_name in BRAND_ALIASES.items()
}

# OPTIMIZATION: Compile AI overview unavailability phrases
UNAVAILABLE_REGEX = re.compile(
    r"(ai overview is not available|ai overview not available|not available for this search)",
    re.IGNORECASE
)

# ---------------------------------------------------------
# File helpers
# ---------------------------------------------------------

def load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

# ---------------------------------------------------------
# URL processing
# ---------------------------------------------------------

def normalize_url(url: str) -> str:
    """
    Remove tracking parameters, sort queries for exact deduplication, 
    and strip fragments.
    """
    if not url:
        return ""

    try:
        parsed = urlparse(url)

        query = ""
        if parsed.query:
            # OPTIMIZATION: Parse and sort query params to ensure that
            # ?a=1&b=2 and ?b=2&a=1 result in the exact same URL hash.
            params = parse_qsl(parsed.query, keep_blank_values=True)
            useful_params = sorted(
                (k, v) for k, v in params if k.lower() not in TRACKING_PARAMETERS
            )
            query = urlencode(useful_params)

        normalized = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/") or "/",
            "",
            query,
            "",
        ))
        return normalized
    except Exception:
        return url

def extract_domain(url: str) -> str:
    try:
        hostname = urlparse(url).hostname
        if not hostname:
            return ""
        
        hostname = hostname.lower()
        return hostname[4:] if hostname.startswith("www.") else hostname
    except Exception:
        return ""

def create_source_id(url: str) -> str:
    return f"src_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:8]}"

# ---------------------------------------------------------
# Brand detection
# ---------------------------------------------------------

def detect_brand(title: str, snippet: str, domain: str) -> str | None:
    """
    Finds brands using pre-compiled regex with word boundaries.
    """
    # Combining text prevents multiple regex searches
    text = f"{domain} {title} {snippet}"

    for pattern, brand_name in COMPILED_BRANDS.items():
        if pattern.search(text):
            return brand_name
    return None

# CHANGE START: Page-level source classification and authority analysis engine
def domain_matches_brand(domain: str, brand: str) -> bool:
    """Check whether domain belongs to brand's first-party domain configuration."""
    if not domain or not brand:
        return False
    domain = domain.lower()
    profile = BRAND_PROFILES.get(brand, {})

    for brand_domain in profile.get("domains", []):
        brand_domain = brand_domain.lower()
        if domain == brand_domain or domain.endswith("." + brand_domain):
            return True
    return False

def classify_ownership(domain: str, brand: str | None) -> str:
    """Determine ownership status (first_party, third_party, or unknown)."""
    if not brand:
        return "unknown"
    if domain_matches_brand(domain, brand):
        return "first_party"
    return "third_party"

def classify_page_source_type(url: str, domain: str, title: str, snippet: str) -> str:
    """
    Evaluate URL path, domain properties, title, and snippet to classify source type.
    Handles page-level differentiation on identical domains.
    """
    parsed_url = urlparse(url.lower() if url else "")
    path = parsed_url.path
    normalized_domain = domain.lower() if domain else ""
    text_content = f"{title} {snippet}".lower()

    # 1. Forum / Social
    if any(d in normalized_domain for d in ["reddit.com", "quora.com", "stackoverflow.com", "forums."]):
        return "forum"
    if any(re.search(pat, path) for pat in PAGE_PATH_PATTERNS["forum"]):
        return "forum"
    if any(d in normalized_domain for d in ["linkedin.com", "twitter.com", "x.com", "facebook.com"]):
        return "social"

    # 2. Comparison Site
    if any(re.search(pat, path) for pat in PAGE_PATH_PATTERNS["comparison_site"]) or " vs " in text_content or " versus " in text_content:
        return "comparison_site"

    # 3. Customer Review / Review Platform
    if any(d in normalized_domain for d in HIGH_AUTHORITY_DOMAINS["review_platforms"]):
        return "review_platform"
    if any(re.search(pat, path) for pat in PAGE_PATH_PATTERNS["customer_review"]) or "review" in text_content:
        return "customer_review"

    # 4. News
    if any(d in normalized_domain for d in HIGH_AUTHORITY_DOMAINS["news"]):
        if "/news/" in path or "/article/" in path:
            return "news"

    # 5. Marketplace / Aggregator
    if any(d in normalized_domain for d in HIGH_AUTHORITY_DOMAINS["aggregators_and_markets"]):
        return "aggregator"
    if any(re.search(pat, path) for pat in PAGE_PATH_PATTERNS["marketplace"]):
        return "marketplace"

    # 6. Editorial vs Independent Blog
    if any(re.search(pat, path) for pat in PAGE_PATH_PATTERNS["editorial"]):
        return "editorial"
    if "/blog/" in path or "blog." in normalized_domain:
        return "independent_blog"

    return "unknown"

def evaluate_source_authority(url: str, domain: str, title: str, snippet: str) -> Dict[str, str]:
    """Compute domain authority tier, topical relevance, and source reliability."""
    normalized_domain = domain.lower() if domain else ""
    text_tokens = set(re.findall(r"\b\w+\b", f"{title} {snippet}".lower()))

    # Domain Authority heuristic
    all_top_domains = [d for sublist in HIGH_AUTHORITY_DOMAINS.values() for d in sublist]
    if any(d in normalized_domain for d in all_top_domains) or normalized_domain.endswith((".edu", ".gov", ".org")):
        domain_auth = "high"
    else:
        domain_auth = "medium"

    # Topical Relevance heuristic
    relevant_signals = len(text_tokens & (FEATURE_WORDS | PRICING_WORDS | REVIEW_WORDS | COMPARISON_WORDS))
    topical_rel = "high" if relevant_signals >= 3 else ("medium" if relevant_signals >= 1 else "low")

    # Source Reliability heuristic
    if domain_auth == "high" and topical_rel in ["high", "medium"]:
        reliability = "high"
    elif "forum" in normalized_domain or "reddit" in normalized_domain:
        reliability = "medium"
    else:
        reliability = "medium"

    return {
        "domain_authority": domain_auth,
        "topical_relevance": topical_rel,
        "source_reliability": reliability
    }

def analyze_source_context(url: str, domain: str, title: str, snippet: str, brand: str | None) -> Dict[str, Any]:
    """Build full hierarchical source taxonomy payload."""
    return {
        "ownership": classify_ownership(domain, brand),
        "source_type": classify_page_source_type(url, domain, title, snippet),
        "authority": evaluate_source_authority(url, domain, title, snippet)
    }
    
# ---------------------------------------------------------
# Visibility scoring
# ---------------------------------------------------------

# OPTIMIZATION: Ranks are repetitive (1 to ~100). 
# Memoizing this skips the math.log2 calculation for thousands of rows.
@lru_cache(maxsize=200)
def position_weight(rank: int) -> float:
    if rank <= 0:
        return 0.0
    return 1.0 / math.log2(rank + 1)

# ---------------------------------------------------------
# Normalize organic results
# ---------------------------------------------------------

def normalize_results(raw_results: list) -> list:
    normalized = []
    seen_urls = set()

    for index, result in enumerate(raw_results, start=1):
        if not isinstance(result, dict):
            continue

        url = str(result.get("url", "")).strip()
        if not url:
            continue

        clean_url = normalize_url(url)
        if not clean_url or clean_url in seen_urls:
            continue

        seen_urls.add(clean_url)

        title = str(result.get("title", "")).strip()
        snippet = str(result.get("snippet", "")).strip()
        domain = extract_domain(clean_url)

        normalized.append({
            "source_id": create_source_id(clean_url),
            "rank": index,
            "domain": domain,
            "url": clean_url,
            "title": title,
            "snippet": snippet,
            "detected_brand": detect_brand(title, snippet, domain),
            "source": source_context,
            "visibility_weight": round(position_weight(index), 6),
        })

    return normalized

# ---------------------------------------------------------
# Brand analytics
# ---------------------------------------------------------

def calculate_brand_visibility(results: list) -> list:
    if not results:
        return []

    # OPTIMIZATION: Single pass evaluation for both total score and brand scores
    total_score = 0.0
    
    # defaultdict cleans up the dict-key initialization block
    brand_scores = defaultdict(lambda: {
        "score": 0.0, 
        "result_count": 0, 
        "best_rank": float('inf'), 
        "sources": []
    })

    for result in results:
        score = result["visibility_weight"]
        total_score += score
        brand = result.get("detected_brand")

        if brand:
            b_data = brand_scores[brand]
            b_data["score"] += score
            b_data["result_count"] += 1
            b_data["sources"].append(result["source_id"])
            if result["rank"] < b_data["best_rank"]:
                b_data["best_rank"] = result["rank"]

    if total_score == 0:
        return []

    analytics = [
        {
            "brand": brand,
            "best_rank": data["best_rank"],
            "result_count": data["result_count"],
            "visibility_score": round(data["score"], 6),
            "visibility_share": round(data["score"] / total_score, 4),
            "visibility_percentage": round((data["score"] / total_score) * 100, 2),
            "source_ids": data["sources"],
        }
        for brand, data in brand_scores.items()
    ]

    analytics.sort(key=lambda item: item["visibility_score"], reverse=True)
    return analytics

# ---------------------------------------------------------
# AI overview processing
# ---------------------------------------------------------

def process_ai_overview(ai_overview: dict) -> dict:
    default_response = {
        "available": False,
        "text": None,
        "citation_count": 0,
        "sources": [],
    }

    if not isinstance(ai_overview, dict):
        return default_response

    text = str(ai_overview.get("text", "")).strip()
    
    # OPTIMIZATION: Regex search is faster than looping `any(x in text)`
    if not text or UNAVAILABLE_REGEX.search(text):
        default_response["text"] = text or None
        return default_response

    normalized_sources = []
    seen_sources = set() # Avoid AI overview citation duplicates
    
    for source in ai_overview.get("sources", []):
        source_url = source if isinstance(source, str) else str(source.get("url", "")).strip()
        
        if not source_url:
            continue
            
        clean_url = normalize_url(source_url)
        if clean_url in seen_sources:
            continue
            
        seen_sources.add(clean_url)
        domain = extract_domain(clean_url)
        title = str(source.get("title", "")).strip() if isinstance(source, dict) and source.get("title") else ""
        detected_brand = detect_brand(title, "", domain)

        # CHANGE START: Classify source context for AI overview citations
        source_context = analyze_source_context(clean_url, domain, title, "", detected_brand)
    
        
        src_data = {
            "source_id": create_source_id(clean_url),
            "url": clean_url,
            "domain": domain,
            "detected_brand": detected_brand,
            # CHANGE START: Attach source metadata object to AI citation sources
            "source": source_context,
        } 
       if title:
            src_data["title"] = title

        normalized_sources.append(src_data)

    return {
        "available": True,
        "text": text,
        "citation_count": len(normalized_sources),
        "sources": normalized_sources,
    }

# ---------------------------------------------------------
# Main query processor
# ---------------------------------------------------------

def process_query(raw_record: dict) -> tuple[dict, dict]:
    query = str(
        raw_record.get("search_query")
        or raw_record.get("input", {}).get("search_query", "")
    ).strip()

    normalized_results = normalize_results(raw_record.get("organic_results", []))
    ai_overview = process_ai_overview(raw_record.get("ai_overview", {}))

    normalized = {
        "query": query,
        "results_count": len(normalized_results),
        "results": normalized_results,
        "ai_overview": ai_overview,
    }

    analytics = {
        "query": query,
        "organic_search": {
            "results_analyzed": len(normalized_results),
            "brands": calculate_brand_visibility(normalized_results),
        },
        "ai_visibility": {
            "available": ai_overview["available"],
            "citation_count": ai_overview["citation_count"],
            "citation_share": {"total_citations": len(ai_overview["sources"])} if ai_overview["available"] and ai_overview["sources"] else None,
        },
        "representation": {
            "status": "pending_nlp",
            "claims": [],
            "sentiment": None,
        },
        "content_gaps": [],
        "pipeline": {
            "source": "Bright Data",
            "stage": "processed",
            "nlp_completed": False,
        },
    }

    return normalized, analytics

# ---------------------------------------------------------
# Program entry point
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Process Bright Data JSON for ASVIO.")
    parser.add_argument("--input", default="data/raw/brightdata_sample.json", help="Path to Bright Data JSON")
    parser.add_argument("--normalized", default="data/normalized/normalized.json", help="Output normalized JSON")
    parser.add_argument("--analytics", default="data/analytics/analytics.json", help="Output analytics JSON")
    args = parser.parse_args()

    input_path, normalized_path, analytics_path = Path(args.input), Path(args.normalized), Path(args.analytics)
    raw_data = load_json(input_path)

    records = [raw_data] if isinstance(raw_data, dict) else raw_data
    if not isinstance(records, list):
        raise ValueError("Input JSON must contain an object or array.")

    all_normalized, all_analytics = [], []

    for record in records:
        if isinstance(record, dict):
            norm, ana = process_query(record)
            all_normalized.append(norm)
            all_analytics.append(ana)

    save_json({"project": "ASVIO", "version": "0.1.0", "records": all_normalized}, normalized_path)
    save_json({"project": "ASVIO", "version": "0.1.0", "records": all_analytics}, analytics_path)

    print("ASVIO processing complete.")
    print(f"Processed queries: {len(all_analytics)}")
    print(f"Normalized output: {normalized_path}")
    print(f"Analytics output: {analytics_path}")

if __name__ == "__main__":
    main()
