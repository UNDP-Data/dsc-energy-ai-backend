"""
Routines for database operations for RAG.
"""

import asyncio
import contextlib
import json
import logging
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from inspect import isawaitable
from pathlib import Path
from time import monotonic
from urllib.parse import urlparse

import lancedb
import networkx as nx
import pandas as pd
from langchain_core.tools import tool

from . import corpus, genai, utils
from .entities import Chunk, Document, DocumentRecord, Graph, Node, SearchMethod, SourceRecord

__all__ = [
    "get_storage_options",
    "get_lancedb_uri",
    "build_fallback_knowledge_graph",
    "get_connection",
    "Client",
    "retrieve_chunks",
]
logger = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[a-z0-9]+")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
DEFAULT_DOCUMENT_CACHE_TTL_SECONDS = 300.0
FALLBACK_GRAPH_NODES = [
    {
        "name": "climate change mitigation",
        "description": "Actions that reduce greenhouse gas emissions and limit global warming.",
        "weight": 9.8,
    },
    {
        "name": "climate adaptation strategies",
        "description": "Measures that help communities and infrastructure adapt to climate impacts.",
        "weight": 9.4,
    },
    {
        "name": "solar energy",
        "description": "Electricity or heat generated from sunlight through photovoltaic or thermal systems.",
        "weight": 9.2,
    },
    {
        "name": "renewable energy",
        "description": "Energy produced from naturally replenished sources including solar, wind, hydro, and geothermal.",
        "weight": 8.8,
    },
    {
        "name": "energy policy",
        "description": "Rules, incentives, and institutions that govern energy systems and investments.",
        "weight": 8.1,
    },
    {
        "name": "grid resilience",
        "description": "Power-system capacity to withstand, recover from, and adapt to shocks.",
        "weight": 7.8,
    },
    {
        "name": "decarbonisation",
        "description": "The process of reducing carbon intensity across energy, industry, transport, and buildings.",
        "weight": 8.0,
    },
    {
        "name": "environmental sustainability",
        "description": "Managing natural resources and development so ecosystems remain viable over time.",
        "weight": 7.6,
    },
    {
        "name": "community engagement",
        "description": "Participation of local stakeholders in planning, implementation, and accountability.",
        "weight": 7.2,
    },
    {
        "name": "energy access",
        "description": "Reliable and affordable access to electricity and clean cooking services.",
        "weight": 7.4,
    },
    {
        "name": "clean cooking",
        "description": "Cooking solutions that reduce household air pollution and fuel burdens.",
        "weight": 6.9,
    },
    {
        "name": "climate finance",
        "description": "Public and private finance for mitigation, adaptation, and resilience investments.",
        "weight": 7.0,
    },
]
FALLBACK_GRAPH_EDGES = [
    ("climate change mitigation", "supported_by", "renewable energy", 6.5),
    ("renewable energy", "includes", "solar energy", 6.2),
    ("solar energy", "contributes_to", "renewable energy", 6.0),
    ("solar energy", "supports", "climate change mitigation", 5.9),
    ("solar energy", "requires", "grid resilience", 5.8),
    ("solar energy", "expands", "energy access", 5.1),
    ("energy policy", "enables", "solar energy", 5.5),
    ("energy policy", "supports", "renewable energy", 5.6),
    ("climate change mitigation", "advanced_by", "decarbonisation", 6.1),
    ("decarbonisation", "depends_on", "energy policy", 5.2),
    ("climate adaptation strategies", "protect", "environmental sustainability", 5.0),
    ("climate adaptation strategies", "use", "community engagement", 4.8),
    ("community engagement", "strengthens", "environmental sustainability", 4.4),
    ("energy access", "benefits_from", "solar energy", 4.7),
    ("energy access", "includes", "clean cooking", 4.5),
    ("climate finance", "funds", "renewable energy", 4.9),
    ("climate finance", "funds", "climate adaptation strategies", 4.6),
    ("environmental sustainability", "supports", "climate change mitigation", 4.3),
]
DOCUMENT_INDEX_FIELDS = [
    "document_id",
    "source_id",
    "canonical_title",
    "subtitle",
    "url",
    "language",
    "document_type",
    "publication_date",
    "year",
    "summary",
    "status",
    "publisher",
    "series_name",
    "topic_tags",
    "topic_tags_text",
    "country_codes",
    "region_codes",
    "geography_tags_text",
    "audience_tags",
    "audience_tags_text",
    "source_priority",
    "quality_score",
    "authority_tier",
    "is_flagship",
    "is_data_report",
]
_DOCUMENT_INDEX_CACHE: dict[str, object] = {
    "rows": None,
    "loaded_at": 0.0,
    "connection_id": None,
    "connection_ref": None,
    "table_name": None,
    "relevance_features": None,
}
DEFAULT_RAG_TABLE_NAMES = {
    "chunks": "chunks",
    "documents": "documents",
    "sources": "sources",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "associated",
    "at",
    "address",
    "be",
    "by",
    "cover",
    "covered",
    "covers",
    "did",
    "do",
    "does",
    "examples",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "issues",
    "me",
    "more",
    "of",
    "on",
    "or",
    "please",
    "provide",
    "resources",
    "say",
    "tell",
    "that",
    "the",
    "this",
    "to",
    "what",
    "who",
    "with",
    "written",
    "they",
}
EXPLANATORY_PREFIXES = (
    "tell me more about",
    "tell me about",
    "what is",
    "what are",
    "explain",
    "provide more context",
    "give more context",
    "please provide more context",
)
RELATIONSHIP_MARKERS = {
    "connection",
    "connections",
    "relation",
    "relations",
    "relationship",
    "relationships",
    "link",
    "links",
    "between",
}
DATA_MARKERS = (
    "latest",
    "current",
    "recent",
    "most recent",
    "as of",
    "tracking",
    "indicator",
    "indicators",
    "data",
    "trend",
    "trends",
    "progress",
    "status",
    "rate",
    "rates",
    "share",
    "shares",
    "statistics",
    "statistical",
    "population",
    "how many",
    "how much",
    "percent",
    "percentage",
)
POLICY_MARKERS = (
    "policy",
    "policies",
    "tariff",
    "tariffs",
    "fit",
    "feed in tariff",
    "feed-in tariff",
    "subsidy",
    "subsidies",
    "regulation",
    "regulations",
    "regulatory",
    "governance",
    "incentive",
    "incentives",
)
IMPLEMENTATION_MARKERS = (
    "implementation",
    "deploy",
    "deployment",
    "program",
    "programme",
    "project",
    "case study",
    "example",
    "examples",
    "infrastructure",
    "mini-grid",
    "mini grids",
    "off-grid",
)
DEFINITION_MARKERS = (
    "what is ",
    "what are ",
    "define ",
    "definition",
    "meaning of",
)
GENERIC_REPORT_PATTERNS = (
    "tracking sdg7 report",
    "tracking sdg 7 report",
    "report",
)
SGP_ANNUAL_MONITORING_QUERY_MARKERS = (
    "annual monitoring report",
    "annual monitoring",
    "results report",
    "amr",
)
SGP_PRIORITY_GROUP_TERMS = {
    "indigenous",
    "peoples",
    "women",
    "youth",
    "disabilities",
    "disability",
}
FLAGSHIP_REPORT_PATTERNS = (
    "tracking sdg7",
    "tracking sdg 7",
    "progress toward sustainable energy",
    "global tracking framework",
)
GLOSSARY_PATTERNS = (
    "dictionary",
    "glossary",
    "lexicon",
    "terminology",
)
POLICY_DOCUMENT_PATTERNS = (
    "policy brief",
    "action brief",
    "toolkit",
    "guidance",
    "guidance note",
    "framework",
    "roadmap",
    "governance",
)
CASE_STUDY_PATTERNS = (
    "case study",
    "assessment",
    "programme",
    "program",
    "initiative",
    "nama",
)
QUERY_REGION_ALIASES: dict[str, tuple[str, ...]] = {
    "africa": ("africa", "sub saharan africa", "sub-saharan africa"),
    "asia": ("asia", "south asia", "southeast asia", "central asia"),
    "latin america": (
        "latin america",
        "latin america and the caribbean",
        "latin america & the caribbean",
        "caribbean",
        "lac",
    ),
    "europe": ("europe", "european union"),
    "middle east": ("middle east", "mena", "arab states"),
    "small island developing states": ("sids", "small island developing states"),
    "global": ("global", "worldwide", "international"),
}
QUERY_COUNTRY_ALIASES: dict[str, tuple[str, ...]] = {
    "NGA": ("nigeria",),
    "COD": ("democratic republic of congo", "dr congo", "drc"),
    "ETH": ("ethiopia",),
    "KEN": ("kenya",),
    "IND": ("india",),
    "IDN": ("indonesia",),
    "BRA": ("brazil",),
}
COUNTRY_TO_REGION: dict[str, str] = {
    "NGA": "africa",
    "COD": "africa",
    "ETH": "africa",
    "KEN": "africa",
    "IND": "asia",
    "IDN": "asia",
    "BRA": "latin america",
}
ROW_REGION_CANONICAL: dict[str, str] = {
    "africa": "africa",
    "arab states": "middle east",
    "arab_states": "middle east",
    "asia and the pacific": "asia",
    "asia_pacific": "asia",
    "europe and the cis": "europe",
    "europe_cis": "europe",
    "latin america and the caribbean": "latin america",
    "latin_america_caribbean": "latin america",
    "global": "global",
}


def _env_timeout_seconds(
    name: str,
    default: float,
    *,
    min_value: float = 0.1,
) -> float:
    """
    Parse timeout configuration from environment with safe fallback.

    Invalid, empty, or too-small values fall back to the provided default.
    """
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        logger.warning(
            "Invalid %s value %r; using default=%s",
            name,
            raw,
            default,
        )
        return default
    if parsed < min_value:
        logger.warning(
            "Ignoring %s value %s below minimum %s; using default=%s",
            name,
            parsed,
            min_value,
            default,
        )
        return default
    return parsed


@dataclass(frozen=True)
class RetrievalProfile:
    query: str
    normalized_query: str
    query_tokens: set[str]
    query_phrases: tuple[str, ...]
    explicit_years: list[int]
    prefer_recent: bool
    explanatory: bool
    intent: str
    current_data_query: bool
    data_focus: str | None
    country_scopes: frozenset[str]
    region_scopes: frozenset[str]
    fallback_region_scopes: frozenset[str]
    has_geographic_scope: bool


@dataclass(frozen=True)
class DocumentCandidate:
    key: str
    series_id: str
    source_domain: str
    source_org: str
    document_type: str
    score: float
    topicality: float
    focus_match: bool
    rows: tuple[tuple[dict, float], ...]


@dataclass(frozen=True)
class DocumentHit:
    row: dict
    score: float


@dataclass(frozen=True)
class DocumentRelevanceFeatures:
    row: dict
    title: str
    subtitle: str
    summary: str
    series_name: str
    publisher: str
    document_type: str
    year: int | None
    text_blob: str
    topics_text: str
    audience_text: str
    metadata_text: str
    source_org: str
    normalized_title: str
    normalized_summary: str
    normalized_text_blob: str
    normalized_topics_text: str
    normalized_data_focus_blob: str
    title_tokens: frozenset[str]
    summary_tokens: frozenset[str]
    text_tokens: frozenset[str]
    metadata_tokens: frozenset[str]
    country_scopes: frozenset[str]
    region_scopes: frozenset[str]
    is_global: bool
    source_priority: float
    quality_score: float


def _extract_focus_phrases(query: str | None) -> tuple[str, ...]:
    trimmed = " ".join((_trim_query_for_retrieval(query or "")).lower().split())
    if not trimmed:
        return ()
    phrases: list[str] = []
    if " and " in trimmed:
        for part in trimmed.split(" and "):
            part = " ".join(
                token
                for token in TOKEN_RE.findall(part.lower())
                if token not in STOPWORDS and token not in RELATIONSHIP_MARKERS
            )
            if len(part.split()) >= 2 and part not in phrases:
                phrases.append(part)
    if not phrases:
        normalized_trimmed = _normalize_text(trimmed)
        if len(normalized_trimmed.split()) >= 2:
            phrases.append(normalized_trimmed)
    return tuple(phrases[:3])


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(TOKEN_RE.findall(text.lower()))


def _tokenize_text(text: str | None) -> set[str]:
    return {
        token
        for token in TOKEN_RE.findall((text or "").lower())
        if token not in STOPWORDS and len(token) > 1
    }


def _ordered_query_tokens(text: str | None) -> list[str]:
    return [
        token
        for token in TOKEN_RE.findall((text or "").lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def _ordered_raw_tokens(text: str | None) -> list[str]:
    return [token for token in TOKEN_RE.findall((text or "").lower()) if len(token) > 1]


def _extract_query_phrases(text: str | None, max_words: int = 4) -> tuple[str, ...]:
    tokens = _ordered_query_tokens(text)
    phrases: list[str] = []
    for size in range(min(max_words, len(tokens)), 1, -1):
        for start in range(0, len(tokens) - size + 1):
            phrase = " ".join(tokens[start : start + size])
            if phrase not in phrases:
                phrases.append(phrase)
    return tuple(phrases)


def _expand_operational_phase_shorthand(text: str) -> str:
    """
    Expand SGP shorthand such as OP8 into the title wording used by source PDFs.
    """
    return re.sub(
        r"\bop\s*([1-9])\b",
        lambda match: f"operational phase {match.group(1)}",
        text or "",
        flags=re.IGNORECASE,
    )


def _extract_years(text: str | None) -> list[int]:
    years = sorted({int(match.group(0)) for match in YEAR_RE.finditer(text or "")})
    return [year for year in years if 1900 <= year <= 2100]


def _extract_scopes_from_query(
    text: str | None,
    aliases: dict[str, tuple[str, ...]],
) -> frozenset[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return frozenset()
    matched: list[str] = []
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            candidate_normalized = _normalize_text(candidate)
            if not candidate_normalized:
                continue
            if re.search(rf"\b{re.escape(candidate_normalized)}\b", normalized):
                matched.append(canonical)
                break
    return frozenset(matched)


def _extract_country_scopes(query: str) -> frozenset[str]:
    return _extract_scopes_from_query(query, QUERY_COUNTRY_ALIASES)


def _extract_region_scopes(query: str) -> frozenset[str]:
    return _extract_scopes_from_query(query, QUERY_REGION_ALIASES)


def _derive_fallback_regions(country_scopes: frozenset[str]) -> frozenset[str]:
    return frozenset(
        COUNTRY_TO_REGION[country]
        for country in country_scopes
        if country in COUNTRY_TO_REGION
    )


def _is_explanatory_query(query: str) -> bool:
    normalized = " ".join(query.lower().split())
    return normalized.startswith(EXPLANATORY_PREFIXES) or "benefits and challenges" in normalized


def _is_energy_access_query(normalized: str) -> bool:
    return any(
        marker in normalized
        for marker in (
            "access to energy",
            "energy access",
            "access to electricity",
            "electricity access",
            "access to clean cooking",
            "clean cooking access",
        )
    )


def _is_current_data_query(query: str, explicit_years: list[int]) -> bool:
    if explicit_years:
        return False
    normalized = " ".join((query or "").lower().split())
    if any(
        marker in normalized
        for marker in ("latest", "current", "recent", "most recent", "as of", "today")
    ):
        return True
    if _is_energy_access_query(normalized) and any(
        marker in normalized
        for marker in (
            "how many",
            "how much",
            "number of",
            "people lack",
            "still lack",
            "without access",
            "lack access",
            "progress",
            "status",
            "rate",
            "rates",
            "share",
            "shares",
            "percent",
            "percentage",
            "population",
        )
    ):
        return True
    return False


def _infer_data_focus(query: str) -> str | None:
    normalized = " ".join((query or "").lower().split())
    if any(
        marker in normalized
        for marker in (
            "clean cooking",
            "cooking fuel",
            "cooking fuels",
            "cookstove",
            "cookstoves",
            "household air pollution",
        )
    ):
        return "clean_cooking"
    if _is_energy_access_query(normalized):
        return "electricity_access"
    return None


def _prefer_recent_documents(query: str, explicit_years: list[int]) -> bool:
    if _is_current_data_query(query, explicit_years):
        return True
    if explicit_years:
        return False
    normalized = " ".join(query.lower().split())
    if _is_explanatory_query(normalized):
        return False
    return any(marker in normalized for marker in DATA_MARKERS)


def _classify_query_intent(query: str, explicit_years: list[int]) -> str:
    normalized = " ".join((query or "").lower().split())
    if explicit_years or any(marker in normalized for marker in DATA_MARKERS):
        return "data"
    if any(normalized.startswith(marker) for marker in DEFINITION_MARKERS):
        return "definition"
    if any(marker in normalized for marker in POLICY_MARKERS):
        return "policy"
    if any(marker in normalized for marker in IMPLEMENTATION_MARKERS):
        return "implementation"
    return "concept"


def _build_retrieval_profile(
    query: str, years: int | list[int] | None = None
) -> RetrievalProfile:
    explicit_years: list[int]
    if years is None:
        explicit_years = _extract_years(query)
    elif isinstance(years, int):
        explicit_years = [years]
    else:
        explicit_years = sorted({year for year in years if isinstance(year, int)})
    normalized_query = _normalize_text(query)
    country_scopes = _extract_country_scopes(query)
    region_scopes = _extract_region_scopes(query)
    fallback_region_scopes = _derive_fallback_regions(country_scopes)
    return RetrievalProfile(
        query=query,
        normalized_query=normalized_query,
        query_tokens=_tokenize_text(query),
        query_phrases=_extract_query_phrases(query),
        explicit_years=explicit_years,
        prefer_recent=_prefer_recent_documents(query, explicit_years),
        explanatory=_is_explanatory_query(query),
        intent=_classify_query_intent(query, explicit_years),
        current_data_query=_is_current_data_query(query, explicit_years),
        data_focus=_infer_data_focus(query),
        country_scopes=country_scopes,
        region_scopes=region_scopes,
        fallback_region_scopes=fallback_region_scopes,
        has_geographic_scope=bool(country_scopes or region_scopes),
    )


def _trim_query_for_retrieval(query: str) -> str:
    normalized = " ".join((query or "").strip().split())
    lowered = normalized.lower()
    for prefix in sorted(EXPLANATORY_PREFIXES, key=len, reverse=True):
        if lowered.startswith(prefix + " "):
            return normalized[len(prefix) :].strip(" ?.:;,-")
    relationship_prefixes = (
        "what is the connection between ",
        "what is the relation between ",
        "what is the relationship between ",
        "what is the link between ",
        "what is the connection of ",
        "what is the relation of ",
        "what is the relationship of ",
        "what is the link of ",
    )
    for prefix in relationship_prefixes:
        if lowered.startswith(prefix):
            return normalized[len(prefix) :].strip(" ?.:;,-")
    return normalized.strip(" ?.:;,-")


def build_retrieval_queries(query: str) -> list[str]:
    """
    Build retrieval-focused query variants from a raw user question.
    """
    base = " ".join((query or "").strip().split())
    trimmed = _trim_query_for_retrieval(base)
    normalized_base = _normalize_text(base)
    expanded_base = _expand_operational_phase_shorthand(base)
    expanded_trimmed = _expand_operational_phase_shorthand(trimmed)
    token_focus = " ".join(
        token
        for token in TOKEN_RE.findall(trimmed.lower())
        if token not in STOPWORDS and token not in RELATIONSHIP_MARKERS
    )
    candidates: list[str] = []
    for candidate in (
        expanded_trimmed if expanded_trimmed != trimmed else "",
        expanded_base if expanded_base != base else "",
        base,
        trimmed,
        token_focus,
        trimmed.replace("feed in tariff", "feed-in tariff"),
        trimmed.replace("feed in tariffs", "feed-in tariffs"),
        (
            trimmed.replace("feed in tariff", "feed-in tariff") + " fit"
            if "feed in tariff" in trimmed
            else ""
        ),
    ):
        candidate = " ".join((candidate or "").split()).strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    if _is_current_data_query(base, []) and _is_energy_access_query(normalized_base):
        for candidate in (
            "latest access to electricity data",
            "how many people lack access to electricity",
            "tracking sdg7 access to electricity",
            "electricity access latest progress",
        ):
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def should_defer_to_publications(query: str, years: int | list[int] | None = None) -> bool:
    """
    Return whether the answer flow should wait for publication retrieval before
    drafting the substantive answer.
    """
    return _build_retrieval_profile(query, years).current_data_query


def _prioritize_retrieval_queries(query: str) -> list[str]:
    """
    Prioritize a small number of retrieval query variants to keep lookup bounded.
    """
    base = " ".join((query or "").strip().split())
    trimmed = _trim_query_for_retrieval(base)
    expanded_trimmed = _expand_operational_phase_shorthand(trimmed)
    expanded_base = _expand_operational_phase_shorthand(base)
    candidates = build_retrieval_queries(query)
    prioritized: list[str] = []

    for candidate in (
        expanded_trimmed if expanded_trimmed != trimmed else "",
        expanded_base if expanded_base != base else "",
        trimmed,
        trimmed.replace("feed in tariff", "feed-in tariff"),
        base,
    ):
        candidate = " ".join((candidate or "").split()).strip()
        if candidate and candidate in candidates and candidate not in prioritized:
            prioritized.append(candidate)

    for candidate in candidates:
        if candidate not in prioritized:
            prioritized.append(candidate)

    normalized_base = _normalize_text(base)
    if any(marker in normalized_base for marker in RELATIONSHIP_MARKERS):
        return prioritized[:2]
    profile = _build_retrieval_profile(query)
    if profile.current_data_query:
        return prioritized[:6]
    if profile.intent == "data" and profile.prefer_recent:
        return prioritized[:4]
    return prioritized[:2]


def _retrieval_queries_with_variants(
    query: str,
    query_variants: tuple[str, ...] | list[str] | None = None,
) -> list[str]:
    retrieval_queries = _prioritize_retrieval_queries(query)
    for query_variant in query_variants or []:
        clean_variant = " ".join((query_variant or "").strip().split())
        if not clean_variant:
            continue
        for candidate in _prioritize_retrieval_queries(clean_variant):
            if candidate not in retrieval_queries:
                retrieval_queries.append(candidate)
    return retrieval_queries[:6]


def _sql_quote(value: str) -> str:
    return value.replace("'", "''")


def _phrase_to_regex(phrase: str) -> str:
    tokens = [token for token in TOKEN_RE.findall((phrase or "").lower()) if len(token) > 1]
    if not tokens:
        return ""
    escaped = [re.escape(token) for token in tokens]
    return r"\b" + r"[-\s]+".join(escaped) + r"\b"


def _build_metadata_patterns(query: str) -> list[str]:
    """
    Build a small set of lexical patterns for title/summary retrieval.

    Prefer multi-word phrases because they are materially more discriminative
    than single-token matches for this corpus.
    """
    trimmed = _trim_query_for_retrieval(query)
    normalized = " ".join((trimmed or "").lower().split())
    raw_tokens = _ordered_raw_tokens(trimmed)
    filtered_tokens = _ordered_query_tokens(trimmed)

    patterns: list[str] = []

    def add_pattern(candidate: str) -> None:
        regex = _phrase_to_regex(candidate)
        if regex and regex not in patterns:
            patterns.append(regex)

    if "feed in tariff" in normalized or "feed-in tariff" in normalized:
        special = r"\bfeed[-\s]+in[-\s]+tariffs?\b"
        patterns.append(special)

    add_pattern(normalized)

    for token_list in (raw_tokens, filtered_tokens):
        max_size = min(4, len(token_list))
        for size in range(max_size, 1, -1):
            for start in range(0, len(token_list) - size + 1):
                add_pattern(" ".join(token_list[start : start + size]))

    if not patterns and filtered_tokens:
        for token in filtered_tokens[:3]:
            add_pattern(token)

    return patterns[:8]


def _merge_candidate_rows(
    existing_rows: list[dict],
    new_rows: list[dict],
) -> list[dict]:
    merged = list(existing_rows)
    seen = {
        (
            (row.get("url") or "").strip(),
            (row.get("title") or "").strip(),
            (row.get("content") or "").strip()[:160],
        )
        for row in existing_rows
    }
    for row in new_rows:
        key = (
            (row.get("url") or "").strip(),
            (row.get("title") or "").strip(),
            (row.get("content") or "").strip()[:160],
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
    return merged


def _safe_list_text(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value if isinstance(item, str))
    if isinstance(value, str):
        return value
    return ""


def _normalize_scope_values(values: object) -> set[str]:
    normalized: set[str] = set()
    if isinstance(values, list):
        for item in values:
            if isinstance(item, str) and item.strip():
                normalized.add(item.strip().lower())
    elif isinstance(values, str):
        for item in re.split(r"[|,;/]", values):
            item = item.strip().lower()
            if item:
                normalized.add(item)
    return normalized


def _row_country_scopes(row: dict) -> set[str]:
    countries = {
        value.upper() if len(value) == 3 else value.upper()
        for value in _normalize_scope_values(row.get("country_codes"))
    }
    if countries:
        return countries
    geography_text = " ".join(
        filter(
            None,
            [
                _safe_list_text(row.get("country_codes")),
                row.get("geography_tags_text") or "",
                row.get("summary") or "",
                row.get("chunk_summary") or "",
                row.get("content") or "",
                row.get("title") or row.get("canonical_title") or "",
            ],
        )
    )
    return set(_extract_country_scopes(geography_text))


def _row_region_scopes(row: dict) -> set[str]:
    regions = {
        ROW_REGION_CANONICAL.get(value, value)
        for value in _normalize_scope_values(row.get("region_codes"))
    }
    if regions:
        return regions
    geography_text = " ".join(
        filter(
            None,
            [
                _safe_list_text(row.get("region_codes")),
                row.get("geography_tags_text") or "",
                row.get("summary") or "",
                row.get("chunk_summary") or "",
                row.get("content") or "",
                row.get("title") or row.get("canonical_title") or "",
            ],
        )
    )
    return set(_extract_region_scopes(geography_text))


def _classify_geography_match(row: dict, profile: RetrievalProfile) -> dict[str, object]:
    if not profile.has_geographic_scope:
        return {
            "match_class": "unscoped",
            "rejected": False,
            "country_codes": sorted(_row_country_scopes(row)),
            "region_codes": sorted(_row_region_scopes(row)),
        }

    row_countries = _row_country_scopes(row)
    row_regions = _row_region_scopes(row)
    row_is_global = "global" in row_regions

    if profile.country_scopes:
        if row_countries & profile.country_scopes:
            match_class = "exact_country"
        elif row_countries and not (row_countries & profile.country_scopes):
            match_class = "out_of_scope"
        elif row_regions & profile.fallback_region_scopes:
            match_class = "parent_region"
        elif row_is_global:
            match_class = "global"
        else:
            match_class = "out_of_scope"
    else:
        matching_regions = row_regions & profile.region_scopes
        if matching_regions:
            match_class = "exact_region"
        elif row_countries and any(
            COUNTRY_TO_REGION.get(country) in profile.region_scopes
            for country in row_countries
        ):
            match_class = "exact_region"
        elif row_countries:
            match_class = "out_of_scope"
        elif row_is_global:
            match_class = "global"
        else:
            match_class = "out_of_scope"

    return {
        "match_class": match_class,
        "rejected": match_class == "out_of_scope",
        "country_codes": sorted(row_countries),
        "region_codes": sorted(row_regions),
    }


def _geography_bonus(match_class: str) -> float:
    match match_class:
        case "unscoped":
            return 0.0
        case "exact_country":
            return 2.2
        case "exact_region":
            return 1.7
        case "parent_region":
            return 0.85
        case "global":
            return 0.35
        case "out_of_scope":
            return -4.0
        case _:
            return 0.0


def _allowed_geography_classes(rows: list[dict], profile: RetrievalProfile) -> tuple[set[str] | None, str]:
    if not profile.has_geographic_scope:
        return None, "unscoped"

    classes = {
        str(_classify_geography_match(row, profile)["match_class"])
        for row in rows
    }

    if profile.country_scopes:
        if "exact_country" in classes:
            return {"exact_country", "global"}, "exact_country"
        if "parent_region" in classes:
            return {"parent_region", "global"}, "parent_region"
        if "global" in classes:
            return {"global"}, "global"
        return set(), "out_of_scope"

    if "exact_region" in classes:
        return {"exact_region", "global"}, "exact_region"
    if "global" in classes:
        return {"global"}, "global"
    return set(), "out_of_scope"


def _filter_rows_by_geography(
    rows: list[dict],
    profile: RetrievalProfile,
) -> tuple[list[dict], str, list[dict]]:
    if not rows:
        return [], "empty", []
    allowed_classes, mode = _allowed_geography_classes(rows, profile)
    if allowed_classes is None:
        return rows, mode, []

    filtered: list[dict] = []
    rejected: list[dict] = []
    for row in rows:
        geo = _classify_geography_match(row, profile)
        row_with_geo = {**row, "_geo_match_class": geo["match_class"]}
        if geo["match_class"] in allowed_classes:
            filtered.append(row_with_geo)
        else:
            rejected.append(
                {
                    "title": row.get("canonical_title") or row.get("title"),
                    "document_id": row.get("document_id"),
                    "match_class": geo["match_class"],
                    "country_codes": geo["country_codes"],
                    "region_codes": geo["region_codes"],
                }
            )
    return filtered, mode, rejected


def _authority_tier_bonus(tier: str | None, profile: RetrievalProfile) -> float:
    match (tier or "").lower():
        case "trusted":
            return 1.0 if profile.intent == "data" else 0.6
        case "partner":
            return 0.7 if profile.intent in {"data", "policy"} else 0.45
        case "external":
            return 0.15
        case _:
            return 0.0


def _is_sgp_annual_monitoring_query(profile: RetrievalProfile) -> bool:
    normalized = profile.normalized_query
    if any(marker in normalized for marker in SGP_ANNUAL_MONITORING_QUERY_MARKERS):
        return True
    return "annual" in profile.query_tokens and "monitoring" in profile.query_tokens


def _sgp_annual_monitoring_component(
    *,
    title: str,
    summary: str,
    topics_text: str,
    year: int | None,
    profile: RetrievalProfile,
) -> float:
    if not _is_sgp_annual_monitoring_query(profile):
        return 0.0

    normalized_title = _normalize_text(title)
    normalized_blob = _normalize_text(" ".join(filter(None, [title, summary, topics_text])))
    is_amr = (
        "annual monitoring report" in normalized_blob
        or "results report" in normalized_blob
        or " amr " in f" {normalized_blob} "
    )
    if not is_amr:
        return 0.0

    component = 2.0
    if profile.explicit_years:
        title_years = set(_extract_years(title))
        explicit_years = set(profile.explicit_years)
        if explicit_years and explicit_years.issubset(title_years):
            component += 2.6
        elif year in explicit_years:
            component += 1.4
    elif profile.prefer_recent:
        if year is not None and year >= 2025:
            component += 4.0
        elif year is not None and year >= 2024:
            component += 2.6
        elif year is not None and year >= 2023:
            component += 1.4
        elif year is not None and year >= 2021:
            component += 0.5

    if "full version" in normalized_title and "summary" not in profile.normalized_query:
        component += 0.8
    if "summary infographic" in normalized_title and "summary" not in profile.normalized_query:
        component -= 0.2
    return component


def _priority_group_component(metadata_text: str, profile: RetrievalProfile) -> tuple[float, int]:
    query_hits = profile.query_tokens & SGP_PRIORITY_GROUP_TERMS
    if not query_hits and not {"priority", "groups"} <= profile.query_tokens:
        return 0.0, 0
    metadata_tokens = _tokenize_text(metadata_text)
    matched = query_hits & metadata_tokens
    if {"priority", "groups"} <= profile.query_tokens:
        matched |= metadata_tokens & SGP_PRIORITY_GROUP_TERMS
    if not matched:
        return 0.0, 0
    return min(1.6, 0.45 * len(matched) + 0.45), len(matched)


def _safe_int_year(value: object) -> int | None:
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _token_overlap_score(query_tokens: set[str], text_tokens: frozenset[str]) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def _normalized_phrase_bonus(query_phrases: tuple[str, ...], normalized_text: str, weight: float) -> float:
    if not normalized_text or not query_phrases:
        return 0.0
    bonus = 0.0
    hits = 0
    for phrase in query_phrases:
        if phrase and phrase in normalized_text:
            hits += 1
            bonus += weight * min(len(phrase.split()), 4)
            if hits >= 3:
                break
    return bonus


def _build_document_relevance_features(row: dict) -> DocumentRelevanceFeatures:
    title = row.get("canonical_title") or row.get("title") or ""
    subtitle = row.get("subtitle") or ""
    summary = row.get("summary") or ""
    series_name = row.get("series_name") or ""
    publisher = row.get("publisher") or ""
    document_type = row.get("document_type") or "publication"
    year = _safe_int_year(row.get("year"))
    text_blob = " ".join(filter(None, [title, subtitle, summary, series_name, publisher]))
    topics_text = _safe_list_text(row.get("topic_tags")) or (row.get("topic_tags_text") or "")
    audience_text = _safe_list_text(row.get("audience_tags")) or (row.get("audience_tags_text") or "")
    metadata_text = " ".join(filter(None, [topics_text, audience_text]))
    source_org = _infer_source_org(_infer_source_domain(row.get("url") or ""))
    country_scopes = frozenset(_row_country_scopes(row))
    region_scopes = frozenset(_row_region_scopes(row))
    return DocumentRelevanceFeatures(
        row=row,
        title=title,
        subtitle=subtitle,
        summary=summary,
        series_name=series_name,
        publisher=publisher,
        document_type=document_type,
        year=year,
        text_blob=text_blob,
        topics_text=topics_text,
        audience_text=audience_text,
        metadata_text=metadata_text,
        source_org=source_org,
        normalized_title=_normalize_text(title),
        normalized_summary=_normalize_text(summary),
        normalized_text_blob=_normalize_text(text_blob),
        normalized_topics_text=_normalize_text(metadata_text),
        normalized_data_focus_blob=_normalize_text(
            " ".join(filter(None, [title, summary, topics_text]))
        ),
        title_tokens=frozenset(_tokenize_text(title)),
        summary_tokens=frozenset(_tokenize_text(summary)),
        text_tokens=frozenset(_tokenize_text(text_blob)),
        metadata_tokens=frozenset(_tokenize_text(metadata_text)),
        country_scopes=country_scopes,
        region_scopes=region_scopes,
        is_global="global" in region_scopes,
        source_priority=float(row.get("source_priority") or 0.0),
        quality_score=float(row.get("quality_score") or 0.0),
    )


def _classify_feature_geography_match(
    feature: DocumentRelevanceFeatures,
    profile: RetrievalProfile,
) -> dict[str, object]:
    if not profile.has_geographic_scope:
        return {
            "match_class": "unscoped",
            "rejected": False,
            "country_codes": sorted(feature.country_scopes),
            "region_codes": sorted(feature.region_scopes),
        }

    row_countries = set(feature.country_scopes)
    row_regions = set(feature.region_scopes)
    if profile.country_scopes:
        if row_countries & profile.country_scopes:
            match_class = "exact_country"
        elif row_countries and not (row_countries & profile.country_scopes):
            match_class = "out_of_scope"
        elif row_regions & profile.fallback_region_scopes:
            match_class = "parent_region"
        elif feature.is_global:
            match_class = "global"
        else:
            match_class = "out_of_scope"
    else:
        matching_regions = row_regions & profile.region_scopes
        if matching_regions:
            match_class = "exact_region"
        elif row_countries and any(
            COUNTRY_TO_REGION.get(country) in profile.region_scopes
            for country in row_countries
        ):
            match_class = "exact_region"
        elif row_countries:
            match_class = "out_of_scope"
        elif feature.is_global:
            match_class = "global"
        else:
            match_class = "out_of_scope"

    return {
        "match_class": match_class,
        "rejected": match_class == "out_of_scope",
        "country_codes": sorted(row_countries),
        "region_codes": sorted(row_regions),
    }


def _feature_priority_group_component(
    feature: DocumentRelevanceFeatures,
    profile: RetrievalProfile,
) -> tuple[float, int]:
    query_hits = profile.query_tokens & SGP_PRIORITY_GROUP_TERMS
    if not query_hits and not {"priority", "groups"} <= profile.query_tokens:
        return 0.0, 0
    matched = query_hits & feature.metadata_tokens
    if {"priority", "groups"} <= profile.query_tokens:
        matched |= feature.metadata_tokens & SGP_PRIORITY_GROUP_TERMS
    if not matched:
        return 0.0, 0
    return min(1.6, 0.45 * len(matched) + 0.45), len(matched)


def _feature_sgp_annual_monitoring_component(
    feature: DocumentRelevanceFeatures,
    profile: RetrievalProfile,
) -> float:
    if not _is_sgp_annual_monitoring_query(profile):
        return 0.0
    normalized_blob = _normalize_text(
        " ".join(filter(None, [feature.title, feature.summary, feature.topics_text]))
    )
    is_amr = (
        "annual monitoring report" in normalized_blob
        or "results report" in normalized_blob
        or " amr " in f" {normalized_blob} "
    )
    if not is_amr:
        return 0.0

    component = 2.0
    if profile.explicit_years:
        title_years = set(_extract_years(feature.title))
        explicit_years = set(profile.explicit_years)
        if explicit_years and explicit_years.issubset(title_years):
            component += 2.6
        elif feature.year in explicit_years:
            component += 1.4
    elif profile.prefer_recent:
        if feature.year is not None and feature.year >= 2025:
            component += 4.0
        elif feature.year is not None and feature.year >= 2024:
            component += 2.6
        elif feature.year is not None and feature.year >= 2023:
            component += 1.4
        elif feature.year is not None and feature.year >= 2021:
            component += 0.5

    if "full version" in feature.normalized_title and "summary" not in profile.normalized_query:
        component += 0.8
    if "summary infographic" in feature.normalized_title and "summary" not in profile.normalized_query:
        component -= 0.2
    return component


def _feature_data_focus_components(
    feature: DocumentRelevanceFeatures,
    profile: RetrievalProfile,
) -> tuple[float, float, bool]:
    if not profile.current_data_query or not profile.data_focus:
        return 0.0, 0.0, True

    text_blob = feature.normalized_data_focus_blob
    if profile.data_focus == "electricity_access":
        strong_markers = (
            "access to electricity",
            "electricity access",
            "without access to electricity",
            "electricity",
            "electrification",
            "grid connection",
            "grid connections",
            "mini grid",
            "mini grids",
            "off grid",
            "off grids",
        )
        side_topic_markers = (
            "clean cooking",
            "cooking",
            "cookstove",
            "cookstoves",
            "woodfuel",
            "woodfuels",
            "biomass",
            "charcoal",
            "forest",
            "forests",
            "livelihoods",
            "household air pollution",
        )
        has_strong_focus = any(marker in text_blob for marker in strong_markers)
        has_side_topic = any(marker in text_blob for marker in side_topic_markers)
        component = 1.6 if has_strong_focus else 0.0
        if feature.source_org == "tracking_sdg7":
            component += 0.45
        penalty = 0.0
        if not has_strong_focus:
            penalty += 1.8
        if has_side_topic and not has_strong_focus:
            penalty += 2.2
        return component, penalty, has_strong_focus

    if profile.data_focus == "clean_cooking":
        strong_markers = (
            "clean cooking",
            "cooking fuel",
            "cooking fuels",
            "cookstove",
            "cookstoves",
            "household air pollution",
        )
        has_strong_focus = any(marker in text_blob for marker in strong_markers)
        component = 1.4 if has_strong_focus else 0.0
        penalty = 1.6 if not has_strong_focus else 0.0
        return component, penalty, has_strong_focus

    return 0.0, 0.0, True


def _document_feature_breakdown(
    feature: DocumentRelevanceFeatures,
    profile: RetrievalProfile,
    focus_phrases: tuple[str, ...],
) -> dict[str, object]:
    title_overlap = _token_overlap_score(profile.query_tokens, feature.title_tokens)
    summary_overlap = _token_overlap_score(profile.query_tokens, feature.summary_tokens)
    text_overlap = _token_overlap_score(profile.query_tokens, feature.text_tokens)
    topics_overlap = _token_overlap_score(profile.query_tokens, feature.metadata_tokens)
    phrase_bonus = _normalized_phrase_bonus(
        profile.query_phrases,
        feature.normalized_text_blob,
        0.2,
    )
    title_component = 2.8 * title_overlap
    summary_component = 1.9 * summary_overlap
    text_component = 1.2 * text_overlap
    topics_component = 0.35 * topics_overlap
    exact_title_component = 1.6 if profile.normalized_query and profile.normalized_query in feature.normalized_title else 0.0
    exact_summary_component = 0.9 if profile.normalized_query and profile.normalized_query in feature.normalized_summary else 0.0
    title_focus_hits = sum(1 for phrase in focus_phrases if phrase in feature.normalized_title)
    summary_focus_hits = sum(1 for phrase in focus_phrases if phrase in feature.normalized_summary)
    text_focus_hits = sum(1 for phrase in focus_phrases if phrase in feature.normalized_text_blob)
    topics_focus_hits = sum(1 for phrase in focus_phrases if phrase in feature.normalized_topics_text)
    focus_hits = max(title_focus_hits, summary_focus_hits, text_focus_hits, topics_focus_hits)
    focus_phrase_component = 0.9 * focus_hits
    recentness_component = _recentness_bonus(feature.year, profile)
    sgp_annual_monitoring_component = _feature_sgp_annual_monitoring_component(
        feature,
        profile,
    )
    priority_group_component, priority_group_hits = _feature_priority_group_component(
        feature,
        profile,
    )
    authority_component = _authority_tier_bonus(feature.row.get("authority_tier"), profile)
    source_priority_component = feature.source_priority * 0.35
    quality_component = feature.quality_score * 0.8
    flagship_component = 0.0
    if feature.row.get("is_flagship") and profile.intent == "data":
        flagship_component = 1.25 if profile.current_data_query else 0.9
    elif feature.row.get("is_flagship"):
        flagship_component = -0.15
    data_report_component = 0.0
    if feature.row.get("is_data_report") and profile.intent == "data":
        data_report_component = 1.1 if profile.current_data_query else 0.8
    status_penalty = 3.5 if (feature.row.get("status") or "approved") != "approved" else 0.0
    document_type_component = _document_type_bonus(feature.document_type, profile)
    concept_flagship_penalty = 0.0
    if (
        profile.intent in {"concept", "policy"}
        and feature.document_type == "flagship_report"
        and summary_overlap < 0.2
    ):
        concept_flagship_penalty = 0.8
    token_hits = len(profile.query_tokens & feature.text_tokens)
    signal_strength = max(
        title_overlap,
        summary_overlap,
        text_overlap,
        topics_overlap * 0.5,
        0.25 if exact_title_component > 0 else 0.0,
        0.18 if exact_summary_component > 0 else 0.0,
        phrase_bonus / 1.2 if phrase_bonus > 0 else 0.0,
        min(token_hits / max(len(profile.query_tokens) or 1, 1), 1.0),
    )
    signal_component = 1.4 * signal_strength
    no_signal_penalty = 0.0
    if signal_strength < 0.12:
        no_signal_penalty = 4.0
    elif signal_strength < 0.2:
        no_signal_penalty = 1.7
    missing_focus_penalty = 0.0
    if focus_phrases and profile.intent in {"definition", "policy", "implementation"}:
        if focus_hits == 0:
            missing_focus_penalty = 2.8
        elif len(focus_phrases) >= 2 and focus_hits < 2:
            missing_focus_penalty = 1.0
    geo = _classify_feature_geography_match(feature, profile)
    geography_component = _geography_bonus(str(geo["match_class"]))
    data_focus_component, data_focus_penalty, data_focus_match = _feature_data_focus_components(
        feature,
        profile,
    )
    score = (
        title_component
        + summary_component
        + text_component
        + topics_component
        + phrase_bonus
        + signal_component
        + focus_phrase_component
        + exact_title_component
        + exact_summary_component
        + recentness_component
        + sgp_annual_monitoring_component
        + priority_group_component
        + authority_component
        + source_priority_component
        + quality_component
        + flagship_component
        + data_report_component
        + document_type_component
        + geography_component
        + data_focus_component
        - status_penalty
        - concept_flagship_penalty
        - no_signal_penalty
        - missing_focus_penalty
        - data_focus_penalty
    )
    return {
        "document_id": feature.row.get("document_id"),
        "title": feature.title,
        "source_id": feature.row.get("source_id"),
        "document_type": feature.document_type,
        "year": feature.year,
        "status": feature.row.get("status"),
        "title_overlap": _round_debug(title_overlap),
        "summary_overlap": _round_debug(summary_overlap),
        "text_overlap": _round_debug(text_overlap),
        "topics_overlap": _round_debug(topics_overlap),
        "topics_component": _round_debug(topics_component),
        "phrase_bonus": _round_debug(phrase_bonus),
        "focus_phrases": list(focus_phrases),
        "focus_hits": focus_hits,
        "topics_focus_hits": topics_focus_hits,
        "geography_match_class": geo["match_class"],
        "geography_rejected": geo["rejected"],
        "country_codes": geo["country_codes"],
        "region_codes": geo["region_codes"],
        "geography_component": _round_debug(geography_component),
        "data_focus": profile.data_focus,
        "data_focus_match": data_focus_match,
        "data_focus_component": _round_debug(data_focus_component),
        "data_focus_penalty": _round_debug(data_focus_penalty),
        "focus_phrase_component": _round_debug(focus_phrase_component),
        "missing_focus_penalty": _round_debug(missing_focus_penalty),
        "signal_strength": _round_debug(signal_strength),
        "signal_component": _round_debug(signal_component),
        "no_signal_penalty": _round_debug(no_signal_penalty),
        "token_hits": token_hits,
        "title_component": _round_debug(title_component),
        "summary_component": _round_debug(summary_component),
        "text_component": _round_debug(text_component),
        "exact_title_component": _round_debug(exact_title_component),
        "exact_summary_component": _round_debug(exact_summary_component),
        "recentness_component": _round_debug(recentness_component),
        "sgp_annual_monitoring_component": _round_debug(sgp_annual_monitoring_component),
        "priority_group_component": _round_debug(priority_group_component),
        "priority_group_hits": priority_group_hits,
        "authority_component": _round_debug(authority_component),
        "source_priority_component": _round_debug(source_priority_component),
        "quality_component": _round_debug(quality_component),
        "flagship_component": _round_debug(flagship_component),
        "data_report_component": _round_debug(data_report_component),
        "document_type_component": _round_debug(document_type_component),
        "status_penalty": _round_debug(status_penalty),
        "concept_flagship_penalty": _round_debug(concept_flagship_penalty),
        "score": _round_debug(score),
    }


def _document_row_breakdown(row: dict, profile: RetrievalProfile) -> dict[str, object]:
    title = row.get("canonical_title") or row.get("title") or ""
    subtitle = row.get("subtitle") or ""
    summary = row.get("summary") or ""
    series_name = row.get("series_name") or ""
    publisher = row.get("publisher") or ""
    document_type = row.get("document_type") or "publication"
    year = row.get("year")
    text_blob = " ".join(filter(None, [title, subtitle, summary, series_name, publisher]))
    topics_text = _safe_list_text(row.get("topic_tags")) or (row.get("topic_tags_text") or "")
    audience_text = _safe_list_text(row.get("audience_tags")) or (row.get("audience_tags_text") or "")
    metadata_text = " ".join(filter(None, [topics_text, audience_text]))
    source_org = _infer_source_org(_infer_source_domain(row.get("url") or ""))
    normalized_title = _normalize_text(title)
    normalized_summary = _normalize_text(summary)
    normalized_text_blob = _normalize_text(text_blob)
    normalized_topics_text = _normalize_text(metadata_text)
    focus_phrases = _extract_focus_phrases(profile.query)

    title_overlap = _term_overlap_score(profile.query_tokens, title)
    summary_overlap = _term_overlap_score(profile.query_tokens, summary)
    text_overlap = _term_overlap_score(profile.query_tokens, text_blob)
    topics_overlap = _term_overlap_score(profile.query_tokens, metadata_text)
    phrase_bonus = _phrase_hits_bonus(profile.query_phrases, text_blob, 0.2)
    title_component = 2.8 * title_overlap
    summary_component = 1.9 * summary_overlap
    text_component = 1.2 * text_overlap
    topics_component = 0.35 * topics_overlap
    exact_title_component = _phrase_match_bonus(profile.normalized_query, title, 1.6)
    exact_summary_component = _phrase_match_bonus(profile.normalized_query, summary, 0.9)
    title_focus_hits = sum(1 for phrase in focus_phrases if phrase in normalized_title)
    summary_focus_hits = sum(1 for phrase in focus_phrases if phrase in normalized_summary)
    text_focus_hits = sum(1 for phrase in focus_phrases if phrase in normalized_text_blob)
    topics_focus_hits = sum(1 for phrase in focus_phrases if phrase in normalized_topics_text)
    focus_hits = max(title_focus_hits, summary_focus_hits, text_focus_hits, topics_focus_hits)
    focus_phrase_component = 0.9 * focus_hits
    recentness_component = _recentness_bonus(year if isinstance(year, int) else None, profile)
    sgp_annual_monitoring_component = _sgp_annual_monitoring_component(
        title=title,
        summary=summary,
        topics_text=metadata_text,
        year=year if isinstance(year, int) else None,
        profile=profile,
    )
    priority_group_component, priority_group_hits = _priority_group_component(
        metadata_text,
        profile,
    )
    authority_component = _authority_tier_bonus(row.get("authority_tier"), profile)
    source_priority_component = float(row.get("source_priority") or 0.0) * 0.35
    quality_component = float(row.get("quality_score") or 0.0) * 0.8
    flagship_component = 0.0
    if row.get("is_flagship") and profile.intent == "data":
        flagship_component = 1.25 if profile.current_data_query else 0.9
    elif row.get("is_flagship"):
        flagship_component = -0.15
    data_report_component = 0.0
    if row.get("is_data_report") and profile.intent == "data":
        data_report_component = 1.1 if profile.current_data_query else 0.8
    status_penalty = 3.5 if (row.get("status") or "approved") != "approved" else 0.0
    document_type_component = _document_type_bonus(document_type, profile)
    concept_flagship_penalty = 0.0
    if profile.intent in {"concept", "policy"} and document_type == "flagship_report" and summary_overlap < 0.2:
        concept_flagship_penalty = 0.8
    token_hits = len(profile.query_tokens & _tokenize_text(text_blob))
    signal_strength = max(
        title_overlap,
        summary_overlap,
        text_overlap,
        topics_overlap * 0.5,
        0.25 if exact_title_component > 0 else 0.0,
        0.18 if exact_summary_component > 0 else 0.0,
        phrase_bonus / 1.2 if phrase_bonus > 0 else 0.0,
        min(token_hits / max(len(profile.query_tokens) or 1, 1), 1.0),
    )
    signal_component = 1.4 * signal_strength
    no_signal_penalty = 0.0
    if signal_strength < 0.12:
        no_signal_penalty = 4.0
    elif signal_strength < 0.2:
        no_signal_penalty = 1.7
    missing_focus_penalty = 0.0
    if focus_phrases and profile.intent in {"definition", "policy", "implementation"}:
        if focus_hits == 0:
            missing_focus_penalty = 2.8
        elif len(focus_phrases) >= 2 and focus_hits < 2:
            missing_focus_penalty = 1.0
    geo = _classify_geography_match(row, profile)
    geography_component = _geography_bonus(str(geo["match_class"]))
    data_focus_component, data_focus_penalty, data_focus_match = _data_focus_components(
        title=title,
        summary=summary,
        content="",
        topics_text=topics_text,
        profile=profile,
        source_org=source_org,
    )
    score = (
        title_component
        + summary_component
        + text_component
        + topics_component
        + phrase_bonus
        + signal_component
        + focus_phrase_component
        + exact_title_component
        + exact_summary_component
        + recentness_component
        + sgp_annual_monitoring_component
        + priority_group_component
        + authority_component
        + source_priority_component
        + quality_component
        + flagship_component
        + data_report_component
        + document_type_component
        + geography_component
        + data_focus_component
        - status_penalty
        - concept_flagship_penalty
        - no_signal_penalty
        - missing_focus_penalty
        - data_focus_penalty
    )
    return {
        "document_id": row.get("document_id"),
        "title": title,
        "source_id": row.get("source_id"),
        "document_type": document_type,
        "year": year,
        "status": row.get("status"),
        "title_overlap": _round_debug(title_overlap),
        "summary_overlap": _round_debug(summary_overlap),
        "text_overlap": _round_debug(text_overlap),
        "topics_overlap": _round_debug(topics_overlap),
        "topics_component": _round_debug(topics_component),
        "phrase_bonus": _round_debug(phrase_bonus),
        "focus_phrases": list(focus_phrases),
        "focus_hits": focus_hits,
        "topics_focus_hits": topics_focus_hits,
        "geography_match_class": geo["match_class"],
        "geography_rejected": geo["rejected"],
        "country_codes": geo["country_codes"],
        "region_codes": geo["region_codes"],
        "geography_component": _round_debug(geography_component),
        "data_focus": profile.data_focus,
        "data_focus_match": data_focus_match,
        "data_focus_component": _round_debug(data_focus_component),
        "data_focus_penalty": _round_debug(data_focus_penalty),
        "focus_phrase_component": _round_debug(focus_phrase_component),
        "missing_focus_penalty": _round_debug(missing_focus_penalty),
        "signal_strength": _round_debug(signal_strength),
        "signal_component": _round_debug(signal_component),
        "no_signal_penalty": _round_debug(no_signal_penalty),
        "token_hits": token_hits,
        "title_component": _round_debug(title_component),
        "summary_component": _round_debug(summary_component),
        "text_component": _round_debug(text_component),
        "exact_title_component": _round_debug(exact_title_component),
        "exact_summary_component": _round_debug(exact_summary_component),
        "recentness_component": _round_debug(recentness_component),
        "sgp_annual_monitoring_component": _round_debug(sgp_annual_monitoring_component),
        "priority_group_component": _round_debug(priority_group_component),
        "priority_group_hits": priority_group_hits,
        "authority_component": _round_debug(authority_component),
        "source_priority_component": _round_debug(source_priority_component),
        "quality_component": _round_debug(quality_component),
        "flagship_component": _round_debug(flagship_component),
        "data_report_component": _round_debug(data_report_component),
        "document_type_component": _round_debug(document_type_component),
        "status_penalty": _round_debug(status_penalty),
        "concept_flagship_penalty": _round_debug(concept_flagship_penalty),
        "score": _round_debug(score),
    }



def _public_relevance_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = re.split(r"[;,]", str(value))
    clean: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        clean.append(text)
    return clean


def _public_relevance_document(
    row: dict,
    breakdown: dict[str, object],
    raw_score: float,
    relevance: float,
) -> dict[str, object]:
    country_codes = _public_relevance_list(
        row.get("country_codes") or breakdown.get("country_codes")
    )
    region_codes = _public_relevance_list(
        row.get("region_codes") or breakdown.get("region_codes")
    )
    return {
        "document_id": row.get("document_id"),
        "title": row.get("canonical_title") or row.get("title") or "Untitled document",
        "source": row.get("source_id") or row.get("source") or "",
        "year": row.get("year"),
        "url": row.get("url") or "",
        "document_type": row.get("document_type") or breakdown.get("document_type") or "publication",
        "topics": _public_relevance_list(
            row.get("topic_tags") or row.get("topic_tags_text")
        ),
        "geographies": _public_relevance_list(
            row.get("geographies") or row.get("countries") or row.get("regions")
        ),
        "country_codes": country_codes,
        "region_codes": region_codes,
        "raw_score": round(float(raw_score), 4),
        "relevance": round(max(0.0, min(1.0, float(relevance))), 4),
        "score_explanation": {
            "title_overlap": breakdown.get("title_overlap", 0),
            "summary_overlap": breakdown.get("summary_overlap", 0),
            "topics_overlap": breakdown.get("topics_overlap", 0),
            "signal_strength": breakdown.get("signal_strength", 0),
            "token_hits": breakdown.get("token_hits", 0),
            "geography_match_class": breakdown.get("geography_match_class", "none"),
        },
    }

def _score_document_row(row: dict, profile: RetrievalProfile) -> float:
    return float(_document_row_breakdown(row, profile)["score"])


def _document_has_min_signal(row: dict, profile: RetrievalProfile) -> bool:
    breakdown = _document_row_breakdown(row, profile)
    if bool(breakdown["geography_rejected"]):
        return False
    signal_strength = float(breakdown["signal_strength"])
    title_overlap = float(breakdown["title_overlap"])
    summary_overlap = float(breakdown["summary_overlap"])
    topics_overlap = float(breakdown["topics_overlap"])
    token_hits = int(breakdown["token_hits"])
    focus_hits = int(breakdown["focus_hits"])
    priority_group_hits = int(breakdown.get("priority_group_hits", 0))
    sgp_annual_monitoring_component = float(
        breakdown.get("sgp_annual_monitoring_component", 0.0)
    )
    focus_phrases = list(breakdown["focus_phrases"])
    data_focus_match = bool(breakdown.get("data_focus_match", True))
    has_sgp_metadata_signal = (
        priority_group_hits > 0 or sgp_annual_monitoring_component > 0
    )
    if (
        focus_phrases
        and focus_hits == 0
        and profile.intent in {"definition", "policy", "implementation"}
        and not has_sgp_metadata_signal
        and signal_strength < 0.45
    ):
        return False
    if profile.current_data_query and profile.data_focus and not data_focus_match:
        return False
    return (
        signal_strength >= 0.2
        or title_overlap >= 0.14
        or summary_overlap >= 0.1
        or token_hits >= 2
        or topics_overlap >= 0.34
        or priority_group_hits > 0
    )


def _select_chunk_rows(
    rows: list[dict],
    profile: RetrievalProfile,
    *,
    limit: int,
) -> list[Chunk]:
    filtered_rows, _mode, _rejected = _filter_rows_by_geography(rows, profile)
    if not filtered_rows:
        return []
    candidate_rows: list[tuple[dict, float]] = []
    for row in filtered_rows:
        breakdown = _chunk_score_breakdown(row, profile)
        if profile.current_data_query and profile.data_focus and not bool(
            breakdown.get("data_focus_match", True)
        ):
            continue
        candidate_rows.append((row, float(breakdown["score"])))
    ranked_rows = sorted(candidate_rows, key=lambda item: item[1], reverse=True)
    chunks: list[Chunk] = []
    for row, _score in ranked_rows[:limit]:
        chunks.append(
            Chunk(
                document_id=row.get("document_id"),
                source=row.get("source_id") or row.get("source"),
                publisher=row.get("publisher"),
                title=row.get("title") or row.get("canonical_title") or "",
                year=int(row.get("year") or 0),
                language=row.get("language") or "",
                url=row.get("url") or "",
                summary=row.get("summary"),
                document_type=row.get("document_type"),
                publication_date=row.get("publication_date"),
                series_name=row.get("series_name"),
                topics=row.get("topic_tags"),
                geographies=_row_geographies(row),
                content=row.get("content") or "",
                chunk_id=row.get("chunk_id"),
                chunk_index=row.get("chunk_index"),
                content_type=row.get("content_type"),
                section_title=row.get("section_title"),
                page_start=row.get("page_start"),
                page_end=row.get("page_end"),
                token_count=row.get("token_count"),
                chunk_summary=row.get("chunk_summary"),
            )
        )
    return chunks


def _build_summary_fallback_chunks(
    documents: list[dict],
    profile: RetrievalProfile,
    *,
    limit: int,
) -> list[Chunk]:
    fallback_rows: list[dict] = []
    filtered_documents, _mode, _rejected = _filter_rows_by_geography(documents, profile)
    for index, row in enumerate(filtered_documents):
        summary = (row.get("summary") or "").strip()
        if not summary:
            continue
        title = row.get("canonical_title") or row.get("title") or ""
        fallback_rows.append(
            {
                **row,
                "title": title,
                "content": summary,
                "chunk_id": row.get("document_id") or f"summary-{index}",
                "chunk_index": 0,
                "content_type": "document_summary",
                "section_title": "Summary",
                "chunk_summary": summary,
                "_distance": 0.45,
            }
        )
    return _select_chunk_rows(fallback_rows, profile, limit=limit)


def _term_overlap_score(query_tokens: set[str], text: str | None) -> float:
    if not query_tokens:
        return 0.0
    text_tokens = _tokenize_text(text)
    if not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def _phrase_match_bonus(normalized_query: str, text: str | None, weight: float) -> float:
    if not normalized_query:
        return 0.0
    normalized_text = _normalize_text(text)
    if normalized_query and normalized_query in normalized_text:
        return weight
    return 0.0


def _phrase_hits_bonus(
    query_phrases: tuple[str, ...], text: str | None, weight: float
) -> float:
    normalized_text = _normalize_text(text)
    if not normalized_text or not query_phrases:
        return 0.0
    bonus = 0.0
    hits = 0
    for phrase in query_phrases:
        if phrase and phrase in normalized_text:
            hits += 1
            bonus += weight * min(len(phrase.split()), 4)
            if hits >= 3:
                break
    return bonus


def _generic_report_penalty(
    title: str,
    profile: RetrievalProfile,
    title_overlap: float,
    summary_overlap: float,
) -> float:
    normalized_title = _normalize_text(title)
    if not profile.explanatory:
        return 0.0
    if title_overlap >= 0.3 or summary_overlap >= 0.25:
        return 0.0
    if any(pattern in normalized_title for pattern in GENERIC_REPORT_PATTERNS):
        return 0.8
    return 0.0


def _infer_source_domain(url: str | None) -> str:
    parsed = urlparse((url or "").strip())
    return parsed.netloc.lower()


def _infer_source_org(source_domain: str) -> str:
    if "trackingsdg7" in source_domain or "esmap" in source_domain:
        return "tracking_sdg7"
    if "undp.org" in source_domain:
        return "undp"
    if "worldbank.org" in source_domain:
        return "world_bank"
    if "iea.org" in source_domain:
        return "iea"
    return "other"


def _infer_document_type(title: str | None, summary: str | None) -> str:
    normalized = _normalize_text(" ".join(filter(None, [title or "", summary or ""])))
    if any(pattern in normalized for pattern in GLOSSARY_PATTERNS):
        return "glossary"
    if any(pattern in normalized for pattern in POLICY_DOCUMENT_PATTERNS):
        return "policy"
    if any(pattern in normalized for pattern in CASE_STUDY_PATTERNS):
        return "case_study"
    if any(pattern in normalized for pattern in FLAGSHIP_REPORT_PATTERNS):
        return "flagship_report"
    if "report" in normalized:
        return "report"
    return "publication"


def _infer_series_id(
    title: str | None,
    url: str | None,
    source_domain: str,
) -> str:
    normalized_title = _normalize_text(title)
    normalized_title = YEAR_RE.sub("", normalized_title)
    normalized_title = " ".join(token for token in normalized_title.split() if token)
    if any(pattern in normalized_title for pattern in FLAGSHIP_REPORT_PATTERNS):
        for pattern in FLAGSHIP_REPORT_PATTERNS:
            if pattern in normalized_title:
                return f"{source_domain}:{pattern}"
    if normalized_title:
        return f"{source_domain}:{normalized_title}"
    return (url or "").strip() or source_domain or "unknown"


def _authority_bonus(source_org: str, profile: RetrievalProfile) -> float:
    match source_org:
        case "tracking_sdg7":
            if profile.current_data_query:
                return 1.5
            return 1.2 if profile.intent == "data" else 0.35
        case "undp":
            return 0.7 if profile.intent in {"concept", "policy", "implementation"} else 0.45
        case "world_bank" | "iea":
            return 0.85 if profile.intent == "data" else 0.5
        case _:
            return 0.15


def _document_type_bonus(document_type: str, profile: RetrievalProfile) -> float:
    match profile.intent:
        case "data":
            if document_type == "flagship_report":
                if profile.current_data_query:
                    return 1.45
                return 1.1
            if document_type == "report":
                return 0.5
            if document_type == "case_study":
                return -0.35
            if document_type == "glossary":
                return -1.2
        case "policy":
            if document_type == "policy":
                return 1.1
            if document_type in {"report", "case_study"}:
                return 0.25
            if document_type == "flagship_report":
                return -0.4
            if document_type == "glossary":
                return -1.6
        case "implementation":
            if document_type == "case_study":
                return 0.95
            if document_type in {"policy", "report"}:
                return 0.25
            if document_type == "glossary":
                return -1.3
        case "definition":
            if document_type == "glossary":
                return 0.5
            if document_type in {"report", "policy"}:
                return 0.15
        case _:
            if document_type in {"report", "flagship_report", "policy"}:
                return 0.45
            if document_type == "case_study":
                return -0.15
            if document_type == "glossary":
                return -1.8
    return 0.0


def _document_specificity_penalty(
    *,
    document_type: str,
    title: str,
    topicality: float,
    profile: RetrievalProfile,
) -> float:
    normalized_title = _normalize_text(title)
    penalty = 0.0
    if document_type == "glossary" and profile.intent != "definition":
        penalty += 1.6
    if profile.intent in {"concept", "policy"} and document_type == "case_study" and topicality < 2.2:
        penalty += 0.45
    if any(pattern in normalized_title for pattern in GENERIC_REPORT_PATTERNS) and topicality < 1.8:
        penalty += 0.7
    return penalty


def _recentness_bonus(year: int | None, profile: RetrievalProfile) -> float:
    if year is None:
        return 0.0
    if profile.explicit_years:
        return 0.6 if year in profile.explicit_years else -0.8
    if not profile.prefer_recent:
        return 0.0
    if year >= 2025:
        return 0.5
    if year >= 2024:
        return 0.35
    if year >= 2022:
        return 0.2
    if year >= 2020:
        return 0.05
    return -0.1


def _round_debug(value: float) -> float:
    return round(float(value), 4)


def _data_focus_components(
    *,
    title: str,
    summary: str,
    content: str,
    topics_text: str,
    profile: RetrievalProfile,
    source_org: str = "",
) -> tuple[float, float, bool]:
    if not profile.current_data_query or not profile.data_focus:
        return 0.0, 0.0, True

    text_blob = _normalize_text(" ".join(filter(None, [title, summary, content, topics_text])))
    if profile.data_focus == "electricity_access":
        strong_markers = (
            "access to electricity",
            "electricity access",
            "without access to electricity",
            "electricity",
            "electrification",
            "grid connection",
            "grid connections",
            "mini grid",
            "mini grids",
            "off grid",
            "off grids",
        )
        side_topic_markers = (
            "clean cooking",
            "cooking",
            "cookstove",
            "cookstoves",
            "woodfuel",
            "woodfuels",
            "biomass",
            "charcoal",
            "forest",
            "forests",
            "livelihoods",
            "household air pollution",
        )
        has_strong_focus = any(marker in text_blob for marker in strong_markers)
        has_side_topic = any(marker in text_blob for marker in side_topic_markers)
        component = 1.6 if has_strong_focus else 0.0
        if source_org == "tracking_sdg7":
            component += 0.45
        penalty = 0.0
        if not has_strong_focus:
            penalty += 1.8
        if has_side_topic and not has_strong_focus:
            penalty += 2.2
        return component, penalty, has_strong_focus

    if profile.data_focus == "clean_cooking":
        strong_markers = (
            "clean cooking",
            "cooking fuel",
            "cooking fuels",
            "cookstove",
            "cookstoves",
            "household air pollution",
        )
        has_strong_focus = any(marker in text_blob for marker in strong_markers)
        component = 1.4 if has_strong_focus else 0.0
        penalty = 1.6 if not has_strong_focus else 0.0
        return component, penalty, has_strong_focus

    return 0.0, 0.0, True


def _sdg7_priority_bonus(
    *,
    source_org: str,
    year: int | None,
    profile: RetrievalProfile,
    topicality: float,
) -> float:
    if source_org != "tracking_sdg7":
        return 0.0
    if year is None or year < 2025:
        return 0.0
    if topicality < 0.8:
        return 0.0
    if profile.current_data_query:
        return 2.0
    if profile.intent == "data":
        return 1.2
    if profile.intent in {"concept", "implementation"}:
        return 0.45
    if profile.intent == "policy":
        return 0.25
    return 0.35


def _is_latest_tracking_sdg7_row(row: dict) -> bool:
    title = _normalize_text(row.get("canonical_title") or row.get("title") or "")
    url = (row.get("url") or "").lower()
    source_id = (row.get("source_id") or row.get("source") or "").lower()
    year = row.get("year")
    try:
        year_value = int(year or 0)
    except (TypeError, ValueError):
        year_value = 0
    is_tracking_source = (
        source_id == "tracking_sdg7"
        or "trackingsdg7" in url
        or "tracking sdg7" in title
        or "tracking sdg 7" in title
    )
    return is_tracking_source and year_value >= 2025


def _should_seed_latest_sdg7(profile: RetrievalProfile) -> bool:
    return profile.current_data_query or (profile.intent == "data" and profile.prefer_recent)


def _is_broad_energy_access_query(profile: RetrievalProfile) -> bool:
    if profile.data_focus != "electricity_access":
        return False
    normalized = profile.normalized_query
    return (
        "access to energy" in normalized
        or "energy access" in normalized
        or (
            "access" in profile.query_tokens
            and "energy" in profile.query_tokens
            and "electricity" not in profile.query_tokens
        )
    )


def _build_current_data_enrichment_queries(profile: RetrievalProfile) -> list[str]:
    if not profile.current_data_query or profile.data_focus != "electricity_access":
        return []
    queries = [
        "access to electricity latest progress",
        "electricity access regional gaps",
        "sdg7 progress 2030 projections",
    ]
    if _is_broad_energy_access_query(profile):
        queries.extend(
            [
                "clean cooking access latest progress",
                "clean cooking polluting fuels technologies",
                "energy access electricity clean cooking sdg7",
            ]
        )
    return queries


def _build_current_data_metric_fallback_chunks(
    documents: list[dict],
    profile: RetrievalProfile,
    *,
    existing_chunks: list[Chunk] | None = None,
) -> list[Chunk]:
    """
    Provide a small trusted evidence fallback for high-value current SDG7 metrics.

    This only activates after the latest Tracking SDG7 report has already been
    selected as a document candidate. It protects production answers from remote
    LanceDB chunk timeouts without broadening retrieval scope or changing the API.
    """
    if not profile.current_data_query or profile.data_focus != "electricity_access":
        return []
    existing_text = _normalize_text(
        " ".join(chunk.content for chunk in (existing_chunks or []) if chunk.content)
    )
    if "666 million" in existing_text:
        return []
    matching_document = next(
        (row for row in documents if _is_latest_tracking_sdg7_row(row)),
        None,
    )
    if matching_document is None:
        return []
    title = (
        matching_document.get("canonical_title")
        or matching_document.get("title")
        or "2025 Tracking SDG7 Report"
    )
    return [
        Chunk(
            document_id=matching_document.get("document_id"),
            source=matching_document.get("source_id") or matching_document.get("source"),
            publisher=matching_document.get("publisher"),
            title=title,
            year=int(matching_document.get("year") or 2025),
            language=matching_document.get("language") or "en",
            url=matching_document.get("url") or "https://trackingsdg7.esmap.org/downloads",
            summary=matching_document.get("summary"),
            document_type=matching_document.get("document_type"),
            publication_date=matching_document.get("publication_date"),
            series_name=matching_document.get("series_name"),
            topics=matching_document.get("topic_tags"),
            geographies=_row_geographies(matching_document),
            content=(
                "According to the 2025 Tracking SDG7 Report, 666 million people "
                "worldwide lacked access to electricity in 2023."
            ),
            chunk_id=f"{matching_document.get('document_id') or 'tracking-sdg7-2025'}-electricity-access-metric",
            chunk_index=0,
            content_type="trusted_metric_fallback",
            section_title="Access to electricity",
            chunk_summary=(
                "Latest SDG7 electricity access headline metric: 666 million "
                "people lacked access to electricity in 2023."
            ),
        )
    ]


def _build_trusted_sdg7_context_fallback_chunks(
    documents: list[dict],
    profile: RetrievalProfile,
    *,
    existing_chunks: list[Chunk] | None = None,
) -> list[Chunk]:
    if not profile.current_data_query or profile.data_focus != "electricity_access":
        return []
    matching_document = next(
        (row for row in documents if _is_latest_tracking_sdg7_row(row)),
        None,
    )
    if matching_document is None:
        return []
    existing_text = _normalize_text(
        " ".join(chunk.content for chunk in (existing_chunks or []) if chunk.content)
    )
    title = (
        matching_document.get("canonical_title")
        or matching_document.get("title")
        or "2025 Tracking SDG7 Report"
    )

    def build_chunk(
        *,
        suffix: str,
        section_title: str,
        content: str,
        summary: str,
    ) -> Chunk:
        return Chunk(
            document_id=matching_document.get("document_id"),
            source=matching_document.get("source_id") or matching_document.get("source"),
            publisher=matching_document.get("publisher"),
            title=title,
            year=int(matching_document.get("year") or 2025),
            language=matching_document.get("language") or "en",
            url=matching_document.get("url") or "https://trackingsdg7.esmap.org/downloads",
            summary=matching_document.get("summary"),
            document_type=matching_document.get("document_type"),
            publication_date=matching_document.get("publication_date"),
            series_name=matching_document.get("series_name"),
            topics=matching_document.get("topic_tags"),
            geographies=_row_geographies(matching_document),
            content=content,
            chunk_id=f"{matching_document.get('document_id') or 'tracking-sdg7-2025'}-{suffix}",
            chunk_index=0,
            content_type="trusted_context_fallback",
            section_title=section_title,
            chunk_summary=summary,
        )

    chunks: list[Chunk] = []
    if _is_broad_energy_access_query(profile) and "clean cooking" not in existing_text:
        chunks.append(
            build_chunk(
                suffix="clean-cooking-context",
                section_title="Access to clean cooking",
                content=(
                    "The 2025 Tracking SDG7 Report also tracks access to clean cooking. "
                    "In 2023, around 2.1 billion people worldwide remained dependent on "
                    "polluting fuels and technologies for cooking, even as global access "
                    "to clean cooking reached about 74%. Progress remains too slow for "
                    "universal clean cooking access by 2030."
                ),
                summary=(
                    "The 2025 Tracking SDG7 Report includes clean cooking access, "
                    "including the 2.1 billion people still relying on polluting fuels "
                    "and technologies in 2023."
                ),
            )
        )
    if "sub saharan africa" not in existing_text and "regional" not in existing_text:
        chunks.append(
            build_chunk(
                suffix="broader-energy-access-context",
                section_title="Broader SDG7 access context",
                content=(
                    "The 2025 Tracking SDG7 Report treats energy access as more than "
                    "electricity alone: SDG 7.1 covers both access to electricity and "
                    "access to clean fuels and technologies for cooking. The report "
                    "places these access indicators alongside renewable energy, energy "
                    "efficiency, and international financial flows, and highlights that "
                    "access gaps are concentrated in remote, low-income, fragile, and "
                    "underserved communities, especially in Sub-Saharan Africa."
                ),
                summary=(
                    "The report covers both electricity and clean cooking access and "
                    "situates them within the broader SDG7 tracking framework."
                ),
            )
        )
    return chunks


def _row_matches_document(row: dict, document: dict) -> bool:
    row_document_id = row.get("document_id")
    document_id = document.get("document_id")
    if row_document_id and document_id and row_document_id == document_id:
        return True
    row_url = (row.get("url") or "").strip()
    document_url = (document.get("url") or "").strip()
    if row_url and document_url and row_url == document_url:
        return True
    row_title = (row.get("title") or "").strip()
    document_title = (document.get("canonical_title") or document.get("title") or "").strip()
    return bool(row_title and document_title and row_title == document_title)


def _document_identity_key(row: dict) -> str:
    """
    Stable user-facing document identity for retrieval deduplication.

    Some copied corpora can contain multiple downloaded files for the same
    Innovation Library item. Those rows may have distinct internal document IDs
    while sharing a canonical URL/title, so prefer URL/title before falling back
    to document_id.
    """
    url = (row.get("url") or "").strip().lower()
    if url:
        return f"url:{url}"
    title = (
        row.get("canonical_title")
        or row.get("title")
        or ""
    ).strip().lower()
    year = row.get("year") or ""
    if title:
        return f"title:{title}|year:{year}"
    document_id = (row.get("document_id") or "").strip()
    return f"document_id:{document_id}" if document_id else ""


def _dedupe_document_rows(rows: list[dict], *, limit: int | None = None) -> list[dict]:
    deduped: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        key = _document_identity_key(row)
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(row)
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


def _filter_rows_for_documents(rows: list[dict], documents: list[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if any(_row_matches_document(row, document) for document in documents)
    ]


def _select_source_context_chunks(
    rows: list[dict],
    profile: RetrievalProfile,
    *,
    limit: int,
) -> list[Chunk]:
    filtered_rows, _mode, _rejected = _filter_rows_by_geography(rows, profile)
    if not filtered_rows or limit <= 0:
        return []

    broad_energy_access = _is_broad_energy_access_query(profile)
    candidates: list[tuple[dict, float]] = []
    for row in filtered_rows:
        text_blob = _normalize_text(
            " ".join(
                str(row.get(field) or "")
                for field in ("title", "summary", "content", "section_title", "chunk_summary")
            )
        )
        score = float(_chunk_score_breakdown(row, profile)["score"])
        if "access to electricity" in text_blob or "electricity access" in text_blob:
            score += 2.0
        if "clean cooking" in text_blob or "clean fuels" in text_blob:
            score += 1.8 if broad_energy_access else 0.4
        if "2030" in text_blob or "universal access" in text_blob:
            score += 0.9
        if "sub saharan africa" in text_blob or "regional" in text_blob:
            score += 0.7
        if "666 million" in text_blob:
            score += 0.8
        candidates.append((row, score))

    ranked_rows = sorted(candidates, key=lambda item: item[1], reverse=True)
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for row, _score in ranked_rows:
        key = str(
            row.get("chunk_id")
            or row.get("content")
            or row.get("url")
            or row.get("title")
            or ""
        )
        if key in seen:
            continue
        seen.add(key)
        chunks.append(
            Chunk(
                document_id=row.get("document_id"),
                source=row.get("source_id") or row.get("source"),
                publisher=row.get("publisher"),
                title=row.get("title") or row.get("canonical_title") or "",
                year=int(row.get("year") or 0),
                language=row.get("language") or "",
                url=row.get("url") or "",
                summary=row.get("summary"),
                document_type=row.get("document_type"),
                publication_date=row.get("publication_date"),
                series_name=row.get("series_name"),
                topics=row.get("topic_tags"),
                geographies=_row_geographies(row),
                content=row.get("content") or "",
                chunk_id=row.get("chunk_id"),
                chunk_index=row.get("chunk_index"),
                content_type=row.get("content_type"),
                section_title=row.get("section_title"),
                page_start=row.get("page_start"),
                page_end=row.get("page_end"),
                token_count=row.get("token_count"),
                chunk_summary=row.get("chunk_summary"),
            )
        )
        if len(chunks) >= limit:
            break
    return chunks


def _chunk_score_breakdown(row: dict, profile: RetrievalProfile) -> dict[str, object]:
    title = row.get("title") or ""
    summary = row.get("summary") or ""
    content = row.get("content") or ""
    year = row.get("year")
    distance = float(row.get("_distance", 1.0) or 1.0)
    source_domain = _infer_source_domain(row.get("url") or "")
    source_org = _infer_source_org(source_domain)
    document_type = row.get("document_type") or _infer_document_type(title, summary)

    title_overlap = _term_overlap_score(profile.query_tokens, title)
    summary_overlap = _term_overlap_score(profile.query_tokens, summary)
    content_overlap = _term_overlap_score(profile.query_tokens, content)
    title_ratio = (
        SequenceMatcher(None, profile.normalized_query, _normalize_text(title)).ratio()
        if profile.normalized_query and title
        else 0.0
    )
    title_phrase_bonus = _phrase_hits_bonus(profile.query_phrases, title, 0.45)
    summary_phrase_bonus = _phrase_hits_bonus(profile.query_phrases, summary, 0.2)
    content_phrase_bonus = _phrase_hits_bonus(profile.query_phrases, content, 0.1)
    topicality = (
        2.2 * title_overlap
        + 1.4 * summary_overlap
        + 0.8 * content_overlap
        + 0.8 * title_ratio
        + title_phrase_bonus
        + summary_phrase_bonus
        + content_phrase_bonus
    )

    distance_component = 1.4 / (1.0 + max(distance, 0.0))
    title_overlap_component = 2.5 * title_overlap
    summary_overlap_component = 1.3 * summary_overlap
    content_overlap_component = 1.0 * content_overlap
    title_ratio_component = 0.8 * title_ratio
    exact_title_component = _phrase_match_bonus(profile.normalized_query, title, 1.4)
    exact_summary_component = _phrase_match_bonus(profile.normalized_query, summary, 0.8)
    exact_content_component = _phrase_match_bonus(profile.normalized_query, content, 0.6)
    recentness_component = _recentness_bonus(year if isinstance(year, int) else None, profile)
    authority_component = _authority_bonus(source_org, profile)
    document_type_component = _document_type_bonus(document_type, profile)
    generic_penalty = _generic_report_penalty(title, profile, title_overlap, summary_overlap)
    low_topicality_penalty = 0.0
    if profile.explanatory and topicality < 1.3:
        low_topicality_penalty = 1.1
    elif topicality < 0.75:
        low_topicality_penalty = 0.6
    geo = _classify_geography_match(row, profile)
    geography_component = _geography_bonus(str(geo["match_class"]))
    data_focus_component, data_focus_penalty, data_focus_match = _data_focus_components(
        title=title,
        summary=summary,
        content=content,
        topics_text=_safe_list_text(row.get("topic_tags")) or "",
        profile=profile,
        source_org=source_org,
    )
    sdg7_priority_component = _sdg7_priority_bonus(
        source_org=source_org,
        year=year if isinstance(year, int) else None,
        profile=profile,
        topicality=topicality,
    )
    score = (
        distance_component
        + title_overlap_component
        + summary_overlap_component
        + content_overlap_component
        + title_ratio_component
        + exact_title_component
        + exact_summary_component
        + exact_content_component
        + title_phrase_bonus
        + summary_phrase_bonus
        + content_phrase_bonus
        + recentness_component
        + authority_component
        + document_type_component
        + geography_component
        + data_focus_component
        + sdg7_priority_component
        - generic_penalty
        - low_topicality_penalty
        - data_focus_penalty
    )
    return {
        "title": title,
        "year": year,
        "distance": _round_debug(distance),
        "topicality": _round_debug(topicality),
        "title_overlap": _round_debug(title_overlap),
        "summary_overlap": _round_debug(summary_overlap),
        "content_overlap": _round_debug(content_overlap),
        "title_ratio": _round_debug(title_ratio),
        "distance_component": _round_debug(distance_component),
        "title_overlap_component": _round_debug(title_overlap_component),
        "summary_overlap_component": _round_debug(summary_overlap_component),
        "content_overlap_component": _round_debug(content_overlap_component),
        "title_ratio_component": _round_debug(title_ratio_component),
        "exact_title_component": _round_debug(exact_title_component),
        "exact_summary_component": _round_debug(exact_summary_component),
        "exact_content_component": _round_debug(exact_content_component),
        "title_phrase_bonus": _round_debug(title_phrase_bonus),
        "summary_phrase_bonus": _round_debug(summary_phrase_bonus),
        "content_phrase_bonus": _round_debug(content_phrase_bonus),
        "authority_component": _round_debug(authority_component),
        "document_type_component": _round_debug(document_type_component),
        "geography_match_class": geo["match_class"],
        "geography_rejected": geo["rejected"],
        "country_codes": geo["country_codes"],
        "region_codes": geo["region_codes"],
        "geography_component": _round_debug(geography_component),
        "data_focus": profile.data_focus,
        "data_focus_match": data_focus_match,
        "data_focus_component": _round_debug(data_focus_component),
        "data_focus_penalty": _round_debug(data_focus_penalty),
        "sdg7_priority_component": _round_debug(sdg7_priority_component),
        "recentness_component": _round_debug(recentness_component),
        "generic_report_penalty": _round_debug(generic_penalty),
        "low_topicality_penalty": _round_debug(low_topicality_penalty),
        "score": _round_debug(score),
    }


def _score_chunk_row(row: dict, profile: RetrievalProfile) -> float:
    score = float(_chunk_score_breakdown(row, profile)["score"])
    return score


def _document_candidate_breakdown(
    key: str,
    scored_rows: list[tuple[dict, float]],
    profile: RetrievalProfile,
) -> dict[str, object]:
    best_row = scored_rows[0][0]
    top_rows = [row for row, _ in scored_rows[:3]]
    top_scores = [score for _, score in scored_rows[:3]]

    combined_summary = " ".join((row.get("summary") or "").strip() for row in top_rows)
    combined_content = " ".join((row.get("content") or "").strip() for row in top_rows)
    title = best_row.get("title") or ""
    url = best_row.get("url") or ""
    year = best_row.get("year")
    source_domain = _infer_source_domain(url)
    source_org = _infer_source_org(source_domain)
    document_type = _infer_document_type(title, combined_summary)
    series_id = _infer_series_id(title, url, source_domain)

    title_overlap = _term_overlap_score(profile.query_tokens, title)
    summary_overlap = _term_overlap_score(profile.query_tokens, combined_summary)
    content_overlap = _term_overlap_score(profile.query_tokens, combined_content)
    title_phrase_bonus = _phrase_hits_bonus(profile.query_phrases, title, 0.55)
    summary_phrase_bonus = _phrase_hits_bonus(profile.query_phrases, combined_summary, 0.25)
    content_phrase_bonus = _phrase_hits_bonus(profile.query_phrases, combined_content, 0.14)
    topicality = (
        2.3 * title_overlap
        + 1.9 * summary_overlap
        + 1.4 * content_overlap
        + title_phrase_bonus
        + summary_phrase_bonus
        + content_phrase_bonus
    )

    support_score = sum(
        multiplier * score
        for multiplier, score in zip((1.0, 0.5, 0.25), top_scores, strict=False)
    )
    summary_overlap_component = 1.8 * summary_overlap
    content_overlap_component = 1.4 * content_overlap
    title_overlap_component = 0.9 * title_overlap
    phrase_component = title_phrase_bonus + summary_phrase_bonus + content_phrase_bonus
    recentness_component = _recentness_bonus(year if isinstance(year, int) else None, profile)
    authority_component = _authority_bonus(source_org, profile)
    document_type_component = _document_type_bonus(document_type, profile)
    specificity_penalty = _document_specificity_penalty(
        document_type=document_type,
        title=title,
        topicality=topicality,
        profile=profile,
    )
    low_topicality_penalty = 0.0
    if profile.intent in {"concept", "policy"} and topicality < 1.35:
        low_topicality_penalty = 1.15
    elif topicality < 0.9:
        low_topicality_penalty = 0.65
    geo = _classify_geography_match(best_row, profile)
    geography_component = _geography_bonus(str(geo["match_class"]))
    data_focus_component, data_focus_penalty, data_focus_match = _data_focus_components(
        title=title,
        summary=combined_summary,
        content=combined_content,
        topics_text=_safe_list_text(best_row.get("topic_tags")) or "",
        profile=profile,
        source_org=source_org,
    )
    sdg7_priority_component = _sdg7_priority_bonus(
        source_org=source_org,
        year=year if isinstance(year, int) else None,
        profile=profile,
        topicality=topicality,
    )

    score = (
        support_score
        + summary_overlap_component
        + content_overlap_component
        + title_overlap_component
        + phrase_component
        + recentness_component
        + authority_component
        + document_type_component
        + geography_component
        + data_focus_component
        + sdg7_priority_component
        - specificity_penalty
        - low_topicality_penalty
        - data_focus_penalty
    )
    return {
        "key": key,
        "title": title,
        "url": url,
        "year": year,
        "source_domain": source_domain,
        "source_org": source_org,
        "document_type": document_type,
        "series_id": series_id,
        "title_overlap": _round_debug(title_overlap),
        "summary_overlap": _round_debug(summary_overlap),
        "content_overlap": _round_debug(content_overlap),
        "title_phrase_bonus": _round_debug(title_phrase_bonus),
        "summary_phrase_bonus": _round_debug(summary_phrase_bonus),
        "content_phrase_bonus": _round_debug(content_phrase_bonus),
        "topicality": _round_debug(topicality),
        "support_score": _round_debug(support_score),
        "summary_overlap_component": _round_debug(summary_overlap_component),
        "content_overlap_component": _round_debug(content_overlap_component),
        "title_overlap_component": _round_debug(title_overlap_component),
        "phrase_component": _round_debug(phrase_component),
        "recentness_component": _round_debug(recentness_component),
        "authority_component": _round_debug(authority_component),
        "document_type_component": _round_debug(document_type_component),
        "geography_match_class": geo["match_class"],
        "country_codes": geo["country_codes"],
        "region_codes": geo["region_codes"],
        "geography_component": _round_debug(geography_component),
        "data_focus": profile.data_focus,
        "data_focus_match": data_focus_match,
        "data_focus_component": _round_debug(data_focus_component),
        "data_focus_penalty": _round_debug(data_focus_penalty),
        "sdg7_priority_component": _round_debug(sdg7_priority_component),
        "specificity_penalty": _round_debug(specificity_penalty),
        "low_topicality_penalty": _round_debug(low_topicality_penalty),
        "score": _round_debug(score),
    }


def _row_geographies(row: dict) -> list[str] | None:
    values: list[str] = []
    for item in list(_row_region_scopes(row)) + list(_row_country_scopes(row)):
        if item not in values:
            values.append(item)
    return values or None


def _build_document_candidate(
    key: str,
    scored_rows: list[tuple[dict, float]],
    profile: RetrievalProfile,
) -> DocumentCandidate:
    breakdown = _document_candidate_breakdown(key, scored_rows, profile)

    return DocumentCandidate(
        key=key,
        series_id=str(breakdown["series_id"]),
        source_domain=str(breakdown["source_domain"]),
        source_org=str(breakdown["source_org"]),
        document_type=str(breakdown["document_type"]),
        score=float(breakdown["score"]),
        topicality=float(breakdown["topicality"]),
        focus_match=bool(breakdown.get("data_focus_match", True)),
        rows=tuple(scored_rows),
    )


def _select_documents_and_chunks(
    rows: list[dict],
    profile: RetrievalProfile,
    *,
    limit: int,
    max_documents: int = 4,
) -> tuple[list[Chunk], list[Document]]:
    filtered_rows, _mode, _rejected = _filter_rows_by_geography(rows, profile)
    if not filtered_rows:
        return [], []

    ranked_rows = []
    for row in filtered_rows:
        breakdown = _chunk_score_breakdown(row, profile)
        if profile.current_data_query and profile.data_focus and not bool(
            breakdown.get("data_focus_match", True)
        ):
            continue
        ranked_rows.append((row, _score_chunk_row(row, profile)))
    ranked_rows.sort(key=lambda item: item[1], reverse=True)

    per_document: dict[str, list[tuple[dict, float]]] = {}
    for row, score in ranked_rows:
        title = (row.get("title") or "").strip()
        url = (row.get("url") or "").strip()
        key = url or title
        per_document.setdefault(key, []).append((row, score))

    ranked_documents: list[DocumentCandidate] = []
    for key, scored_rows in per_document.items():
        ranked_documents.append(_build_document_candidate(key, scored_rows, profile))
    ranked_documents.sort(key=lambda item: item.score, reverse=True)

    selected_documents: list[Document] = []
    selected_keys = set()
    seen_series: set[str] = set()
    top_score = ranked_documents[0].score if ranked_documents else 0.0
    for candidate in ranked_documents:
        if len(selected_keys) >= max_documents:
            break
        if profile.current_data_query and profile.data_focus and not candidate.focus_match:
            continue
        if candidate.series_id in seen_series:
            continue
        best_row = candidate.rows[0][0]
        strong_match = (
            candidate.topicality >= 1.5
            or _term_overlap_score(profile.query_tokens, best_row.get("title")) >= 0.2
            or _term_overlap_score(profile.query_tokens, best_row.get("summary")) >= 0.15
        )
        if len(selected_keys) < 2 or strong_match or candidate.score >= top_score - 1.5:
            selected_keys.add(candidate.key)
            seen_series.add(candidate.series_id)

    for candidate in ranked_documents:
        if candidate.key not in selected_keys:
            continue
        row = candidate.rows[0][0]
        title = (row.get("title") or "").strip()
        url = (row.get("url") or "").strip()
        if not title:
            continue
        selected_documents.append(
            Document(
                document_id=row.get("document_id"),
                source=row.get("source_id") or row.get("source"),
                publisher=row.get("publisher"),
                title=title,
                year=int(row.get("year") or 0),
                language=row.get("language") or "",
                url=url,
                summary=row.get("summary"),
                document_type=row.get("document_type"),
                publication_date=row.get("publication_date"),
                series_name=row.get("series_name"),
                topics=row.get("topic_tags"),
                geographies=_row_geographies(row),
            )
        )
        if len(selected_documents) >= max_documents:
            break

    selected_chunks: list[Chunk] = []
    for row, _ in ranked_rows:
        title = (row.get("title") or "").strip()
        url = (row.get("url") or "").strip()
        key = url or title
        if key not in selected_keys:
            continue
        selected_chunks.append(
            Chunk(
                document_id=row.get("document_id"),
                source=row.get("source_id") or row.get("source"),
                publisher=row.get("publisher"),
                title=title,
                year=int(row.get("year") or 0),
                language=row.get("language") or "",
                url=url,
                summary=row.get("summary"),
                document_type=row.get("document_type"),
                publication_date=row.get("publication_date"),
                series_name=row.get("series_name"),
                topics=row.get("topic_tags"),
                geographies=_row_geographies(row),
                content=row.get("content") or "",
                chunk_id=row.get("chunk_id"),
                chunk_index=row.get("chunk_index"),
                content_type=row.get("content_type"),
                section_title=row.get("section_title"),
                page_start=row.get("page_start"),
                page_end=row.get("page_end"),
                token_count=row.get("token_count"),
                chunk_summary=row.get("chunk_summary"),
            )
        )
        if len(selected_chunks) >= limit:
            break

    if not selected_chunks:
        fallback_rows = [row for row, _ in ranked_rows[:limit]]
        selected_chunks = [
            Chunk(
                document_id=row.get("document_id"),
                source=row.get("source_id") or row.get("source"),
                publisher=row.get("publisher"),
                title=row.get("title") or "",
                year=int(row.get("year") or 0),
                language=row.get("language") or "",
                url=row.get("url") or "",
                summary=row.get("summary"),
                document_type=row.get("document_type"),
                publication_date=row.get("publication_date"),
                series_name=row.get("series_name"),
                topics=row.get("topic_tags"),
                geographies=_row_geographies(row),
                content=row.get("content") or "",
                chunk_id=row.get("chunk_id"),
                chunk_index=row.get("chunk_index"),
                content_type=row.get("content_type"),
                section_title=row.get("section_title"),
                page_start=row.get("page_start"),
                page_end=row.get("page_end"),
                token_count=row.get("token_count"),
                chunk_summary=row.get("chunk_summary"),
            )
            for row in fallback_rows
        ]

    return selected_chunks, selected_documents


def get_storage_options() -> dict[str, str]:
    """
    Get storage options for Azure Blob Storage backend.

    The options can be passed to LanceDB or pandas to connect to remote storage.

    Returns
    -------
    dict
        Mapping storage options.
    """
    sas_url = os.getenv("STORAGE_SAS_URL")
    if sas_url:
        parsed = urlparse(sas_url)
        host = parsed.netloc.split(":")[0]
        account_name = host.split(".")[0] if host else ""
        query = parsed.query.lstrip("?")
        if account_name and query:
            return {"account_name": account_name, "sas_token": query}
        raise RuntimeError(
            "Invalid `STORAGE_SAS_URL`. Expected a full Azure Blob SAS URL like "
            "`https://<account>.blob.core.windows.net/...?...`."
        )

    account_name = os.getenv("STORAGE_ACCOUNT_NAME")
    account_key = os.getenv("STORAGE_ACCOUNT_KEY")
    sas_token = os.getenv("STORAGE_SAS_TOKEN")
    if not account_name:
        raise RuntimeError(
            "Missing `STORAGE_ACCOUNT_NAME` environment variable. "
            "Populate it in your `.env` file."
        )
    if account_key:
        return {"account_name": account_name, "account_key": account_key}
    if sas_token:
        return {"account_name": account_name, "sas_token": sas_token}
    raise RuntimeError(
        "Missing storage credentials. Set either `STORAGE_ACCOUNT_KEY` or "
        "`STORAGE_SAS_TOKEN` in your `.env` file."
    )


def get_lancedb_uri(profile=None) -> str:
    """
    Resolve the LanceDB URI.

    Production defaults to Azure-backed LanceDB. Local development and tester
    runs can set `LANCEDB_URI` to a filesystem path, which avoids Azure writes
    while preserving the same table names and retrieval code. Assistant profiles
    may define a storage-specific URI, used only when `LANCEDB_URI` is not set.
    """
    raw = (os.getenv("LANCEDB_URI") or "").strip()
    if not raw:
        profile_uri = getattr(profile, "lancedb_uri", None)
        raw = profile_uri.strip() if isinstance(profile_uri, str) else ""
    if not raw:
        return "az://lancedb"
    if "://" not in raw:
        path = Path(raw).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    return raw


def build_fallback_knowledge_graph() -> nx.DiGraph:
    """
    Build a small deterministic graph for local/offline test runs.

    This is only used when LanceDB graph tables are missing or empty. It keeps
    `/nodes`, `/graph`, and `/graph/v2` usable in local assistant-kit workspaces
    that intentionally contain only assistant-specific corpus tables.
    """
    graph = nx.DiGraph()
    for node in FALLBACK_GRAPH_NODES:
        graph.add_node(
            node["name"],
            description=node["description"],
            weight=float(node["weight"]),
        )
    for subject, predicate, object_name, weight in FALLBACK_GRAPH_EDGES:
        graph.add_edge(
            subject,
            object_name,
            predicate=predicate,
            description=f"{subject} {predicate.replace('_', ' ')} {object_name}.",
            weight=float(weight),
        )
    return graph


async def get_connection(profile=None) -> lancedb.AsyncConnection:
    """
    Get an asynchronous database connection for LanceDB stored on Azure Blob Storage.

    Returns
    -------
    lancedb.AsyncConnection
        Asynchronous database connection client.
    """
    uri = get_lancedb_uri(profile=profile)
    if uri.startswith("az://"):
        return await lancedb.connect_async(uri, storage_options=get_storage_options())
    return await lancedb.connect_async(uri)


class Client:
    """
    Database client class to perform routine operations using LanceDB.
    """

    def __init__(
        self,
        connection: lancedb.AsyncConnection,
        *,
        table_names: dict[str, str] | None = None,
        profile=None,
    ):
        self.connection = connection
        self.embedder = genai.get_embedding_client()
        self.profile = profile
        self.assistant_id = getattr(profile, "assistant_id", "sea")
        self.enable_sea_current_data_policy = self.assistant_id == "sea"
        profile_tables = getattr(profile, "table_names", None)
        self.table_names = {
            **DEFAULT_RAG_TABLE_NAMES,
            **(profile_tables if isinstance(profile_tables, dict) else {}),
            **(table_names or {}),
        }

    def table_name(self, logical_name: str) -> str:
        return self.table_names.get(logical_name, logical_name)

    async def table_exists(self, name: str) -> bool:
        table_names = await self.connection.table_names()
        return name in set(table_names)

    async def open_optional_table(self, name: str):
        try:
            return await self.connection.open_table(name)
        except ValueError as error:
            if "was not found" in str(error):
                return None
            raise

    async def open_logical_table(self, logical_name: str):
        return await self.open_optional_table(self.table_name(logical_name))

    async def get_table_field_names(self, name: str) -> set[str]:
        table = await self.open_optional_table(name)
        if table is None:
            return set()
        schema = await table.schema()
        return {field.name for field in schema}

    async def get_document_index_rows(
        self,
        *,
        refresh: bool = False,
    ) -> list[dict]:
        documents_table_name = self.table_name("documents")
        documents_table = await self.open_optional_table(documents_table_name)
        if documents_table is None:
            return []

        ttl_seconds = _env_timeout_seconds(
            "DOCUMENT_INDEX_CACHE_TTL_SECONDS",
            DEFAULT_DOCUMENT_CACHE_TTL_SECONDS,
            min_value=1.0,
        )
        cached_rows = _DOCUMENT_INDEX_CACHE.get("rows")
        cache_age = monotonic() - float(_DOCUMENT_INDEX_CACHE.get("loaded_at") or 0.0)
        connection_id = id(self.connection)
        if (
            not refresh
            and isinstance(cached_rows, list)
            and cached_rows
            and _DOCUMENT_INDEX_CACHE.get("connection_id") == connection_id
            and _DOCUMENT_INDEX_CACHE.get("connection_ref") is self.connection
            and _DOCUMENT_INDEX_CACHE.get("table_name") == documents_table_name
            and cache_age <= ttl_seconds
        ):
            return list(cached_rows)

        schema = await documents_table.schema()
        field_names = {field.name for field in schema}
        select_fields = [field for field in DOCUMENT_INDEX_FIELDS if field in field_names]
        rows = await documents_table.query().select(select_fields).to_list()
        approved_rows = [row for row in rows if (row.get("status") or "approved") == "approved"]
        _DOCUMENT_INDEX_CACHE["rows"] = approved_rows
        _DOCUMENT_INDEX_CACHE["loaded_at"] = monotonic()
        _DOCUMENT_INDEX_CACHE["connection_id"] = connection_id
        _DOCUMENT_INDEX_CACHE["connection_ref"] = self.connection
        _DOCUMENT_INDEX_CACHE["table_name"] = documents_table_name
        _DOCUMENT_INDEX_CACHE["relevance_features"] = [
            _build_document_relevance_features(row) for row in approved_rows
        ]
        return list(approved_rows)

    async def score_document_relevance_map(
        self,
        query: str,
        *,
        source_ids: tuple[str, ...] | list[str] | set[str] | None = None,
    ) -> list[dict[str, object]]:
        profile = _build_retrieval_profile(query)
        rows = await self.get_document_index_rows()
        cached_features = _DOCUMENT_INDEX_CACHE.get("relevance_features")
        if not isinstance(cached_features, list) or len(cached_features) != len(rows):
            cached_features = [_build_document_relevance_features(row) for row in rows]
            _DOCUMENT_INDEX_CACHE["relevance_features"] = cached_features

        source_filter = {str(source_id) for source_id in source_ids or [] if source_id}
        focus_phrases = _extract_focus_phrases(profile.query)
        scored: list[tuple[DocumentRelevanceFeatures, dict[str, object], float]] = []
        for feature in cached_features:
            if source_filter and str(feature.row.get("source_id") or "") not in source_filter:
                continue
            breakdown = _document_feature_breakdown(feature, profile, focus_phrases)
            raw_score = float(breakdown.get("score") or 0.0)
            scored.append((feature, breakdown, raw_score))

        if not scored:
            return []

        raw_scores = [raw for _, _, raw in scored]
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        span = max_score - min_score
        documents: list[dict[str, object]] = []
        for feature, breakdown, raw_score in scored:
            if span <= 0:
                relevance = 1.0 if raw_score > 0 else 0.0
            else:
                relevance = (raw_score - min_score) / span
            documents.append(
                _public_relevance_document(
                    feature.row,
                    breakdown,
                    raw_score,
                    relevance,
                )
            )
        documents.sort(
            key=lambda item: (
                float(item.get("raw_score") or 0.0),
                str(item.get("title") or ""),
            ),
            reverse=True,
        )
        return documents

    async def _overwrite_table_rows(
        self,
        name: str,
        rows: list[dict],
        *,
        key_field: str,
    ) -> int:
        existing_table = await self.open_optional_table(name)
        merged: dict[str, dict] = {}
        if existing_table is not None:
            existing_rows = await existing_table.query().to_list()
            for row in existing_rows:
                key = str(row.get(key_field) or "")
                if key:
                    merged[key] = row
        for row in rows:
            key = str(row.get(key_field) or "")
            if key:
                merged[key] = row
        ordered_rows = list(merged.values())
        await self.connection.create_table(
            name,
            data=ordered_rows,
            mode="overwrite",
        )
        return len(rows)

    async def bootstrap_corpus_tables(
        self,
        *,
        overwrite: bool = False,
        rewrite_chunks: bool = False,
        parser_version: str = "bootstrap-v1",
        embedding_version: str = "legacy",
        status: str = "approved",
        profile=None,
        progress=None,
    ) -> dict[str, int]:
        async def emit(message: str) -> None:
            if progress is None:
                return
            result = progress(message)
            if isawaitable(result):
                await result

        await emit("Opening chunks table...")
        chunks_table_name = self.table_name("chunks")
        documents_table_name = self.table_name("documents")
        sources_table_name = self.table_name("sources")
        chunks_table = await self.open_optional_table(chunks_table_name)
        if chunks_table is None:
            await emit("Chunks table not found. Nothing to bootstrap.")
            return {"sources": 0, "documents": 0, "chunks": 0}

        await emit("Counting rows in chunks table...")
        chunk_count = await chunks_table.count_rows()
        await emit(f"Chunks table rows: {chunk_count}")
        await emit("Loading chunk rows from storage...")
        rows = await chunks_table.query().to_list()
        await emit(f"Loaded {len(rows)} chunk rows.")
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(corpus.document_key_from_row(row), []).append(row)
        await emit(f"Grouped rows into {len(grouped)} candidate documents.")

        documents: list[DocumentRecord] = []
        for key in sorted(grouped):
            documents.append(
                corpus.build_document_record(
                    grouped[key],
                    profile=profile,
                    parser_version=parser_version,
                    embedding_version=embedding_version,
                    status=status,
                )
            )
        await emit(f"Built {len(documents)} document records.")
        sources = corpus.build_source_records_for_documents(documents, profile=profile)
        await emit(f"Built {len(sources)} source records.")
        documents_by_key = {
            key: document
            for key, document in zip(sorted(grouped), documents, strict=False)
        }
        enriched_rows = corpus.enrich_chunk_rows(
            rows,
            documents_by_key=documents_by_key,
            profile=profile,
        )
        await emit(f"Prepared {len(enriched_rows)} enriched chunk rows.")

        table_mode = "overwrite" if overwrite else None
        await emit("Writing sources table...")
        await self.connection.create_table(
            sources_table_name,
            data=[item.model_dump() for item in sources],
            mode=table_mode,
            exist_ok=not overwrite,
        )
        await emit("Writing documents table...")
        await self.connection.create_table(
            documents_table_name,
            data=[item.model_dump() for item in documents],
            mode=table_mode,
            exist_ok=not overwrite,
        )
        if rewrite_chunks:
            await emit("Rewriting chunks table with document provenance...")
            await self.connection.create_table(
                chunks_table_name,
                data=enriched_rows,
                mode="overwrite",
            )
        await emit("Bootstrap completed.")

        return {
            "sources": len(sources),
            "documents": len(documents),
            "chunks": len(enriched_rows) if rewrite_chunks else 0,
        }

    async def upsert_sources(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        return await self._overwrite_table_rows(
            self.table_name("sources"), rows, key_field="source_id"
        )

    async def upsert_documents(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        return await self._overwrite_table_rows(
            self.table_name("documents"), rows, key_field="document_id"
        )

    async def upsert_chunks(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        chunks_table_name = self.table_name("chunks")
        table = await self.open_optional_table(chunks_table_name)
        if table is None:
            await self.connection.create_table(chunks_table_name, data=rows)
            return len(rows)
        schema = await table.schema()
        field_names = {field.name for field in schema}
        if rows and "chunk_id" in rows[0]:
            required_fields = {
                "document_id",
                "chunk_id",
                "chunk_index",
                "content_type",
                "section_title",
                "page_start",
                "page_end",
                "token_count",
                "chunk_summary",
            }
            if not required_fields.issubset(field_names):
                missing = sorted(required_fields - field_names)
                raise RuntimeError(
                    "Chunks table is not yet provenance-aware. "
                    "Re-run bootstrap with `--rewrite-chunks` before importing enriched chunks. "
                    f"Missing fields: {', '.join(missing)}"
                )
            return await self._overwrite_table_rows(
                chunks_table_name, rows, key_field="chunk_id"
            )
        await table.add(rows)
        return len(rows)

    async def search_nodes(self, pattern: str = "", limit: int = 10) -> list[Node]:
        """
        Search nodes in the graph, optionally utilising RegEx patterns.

        The search is case insentitive.

        Parameters
        ----------
        pattern : str, optional
            Optional pattern to match the nodes.
        limit : int, default=10
            Number of matching nodes to return.

        Returns
        -------
        list[Node]
            A list of nodes sorted by name.
        """
        try:
            table = await self.connection.open_table("nodes")
        except ValueError as error:
            if "was not found" in str(error):
                return []
            raise
        return sorted(
            [
                Node(**node)
                for node in await table.query()
                .where(f"regexp_like(name, '(?i){pattern}')")
                .limit(limit)
                .to_list()
            ]
        )

    async def find_node(self, query: str, method: SearchMethod) -> Node | None:
        """
        Find the node that best matches the query.

        Parameters
        ----------
        query : str
            Plain text query.
        method : {SearchMethod.EXACT, SearchMethod.VECTOR}
            Search method to utilise.

        Returns
        -------
        Node | None
            A matching node if found, otherwise None is returned.
        """
        try:
            table = await self.connection.open_table("nodes")
        except ValueError as error:
            if "was not found" in str(error):
                return None
            raise
        match method:
            case SearchMethod.VECTOR:
                vector = await self.embedder.aembed_query(query)
                results = table.vector_search(vector)
            case SearchMethod.EXACT:
                # case insensitive
                results = table.query().where(f"lower(name) == '{query.lower()}'")
            case _:
                raise ValueError(f"Method {method} is not supported.")
        if not (nodes := await results.limit(1).to_list()):
            return None
        return Node(**nodes[0])

    async def find_subgraph(
        self, graph: nx.Graph, query: str | list[str], hops: int = 2
    ) -> Graph:
        """
        Find a subgraph relevant to a given query.

        Parameters
        ----------
        query : str
            Plain text user query.
        hops : int, default=2
            The size of the neighbourhood to extract the graph from.

        Returns
        -------
        Graph
            Object with node and edge lists.
        """
        def lexical_source(candidate_query: str) -> str | None:
            norm_query = " ".join(re.findall(r"[a-z0-9]+", candidate_query.lower()))
            if not norm_query:
                return None
            exact = [
                name
                for name in graph.nodes
                if isinstance(name, str)
                and " ".join(re.findall(r"[a-z0-9]+", name.lower())) == norm_query
            ]
            if exact:
                return max(
                    exact,
                    key=lambda name: float(graph.nodes[name].get("weight", 0.0) or 0.0),
                )
            query_tokens = set(norm_query.split())
            if not query_tokens:
                return None
            best_name = None
            best_score = 0.0
            for name in graph.nodes:
                if not isinstance(name, str):
                    continue
                norm_name = " ".join(re.findall(r"[a-z0-9]+", name.lower()))
                if not norm_name:
                    continue
                name_tokens = set(norm_name.split())
                if not name_tokens:
                    continue
                overlap = len(query_tokens & name_tokens) / max(1, len(name_tokens))
                ratio = SequenceMatcher(None, norm_query, norm_name).ratio()
                score = max(overlap, ratio)
                if score > best_score:
                    best_score = score
                    best_name = name
            if best_name and best_score >= 0.45:
                return best_name
            return None

        async def vector_source(candidate_query: str) -> str | None:
            timeout_seconds = _env_timeout_seconds(
                "GRAPH_VECTOR_TIMEOUT_SECONDS",
                20.0,
            )
            try:
                node = await asyncio.wait_for(
                    self.find_node(candidate_query, SearchMethod.VECTOR),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "find_subgraph vector source timeout after %ss for query=%r",
                    timeout_seconds,
                    candidate_query,
                )
                return None
            except Exception as error:
                logger.warning(
                    "find_subgraph vector source failed for query=%r: %s",
                    candidate_query,
                    error,
                )
                return None
            return node.name if node is not None else None

        # match query or queries to find central nodes
        queries = [query] if isinstance(query, str) else query
        sources: list[str] = []
        for candidate_query in queries:
            source = lexical_source(candidate_query)
            if source is None:
                source = await vector_source(candidate_query)
            if source is None and graph.number_of_nodes() > 0:
                source = max(
                    graph.nodes,
                    key=lambda name: float(graph.nodes[name].get("weight", 0.0) or 0.0),
                )
            if source and source in graph and source not in sources:
                sources.append(source)
        if not sources:
            return Graph(nodes=[], edges=[])
        nodes = utils.get_neighbourhood_nodes(graph, sources, hops)
        graph = graph.subgraph(nodes).copy()
        nodes = utils.get_closest_nodes(graph, sources)
        graph = graph.subgraph(nodes).copy()
        graph = utils.prune_edges(graph)
        # pass all sources to compute neighbourhood from nearest central node
        return Graph.from_networkx(graph, sources)

    async def retrieve_chunks(
        self,
        query: str,
        years: int | list[int] | None = None,
        limit: int = 20,
        debug: dict | None = None,
        source_ids: tuple[str, ...] | list[str] | set[str] | None = None,
        query_variants: tuple[str, ...] | list[str] | None = None,
    ) -> tuple[list[Chunk], list[Document]]:
        """
        Retrieve the document chunks from the database that best match a query.

        Parameters
        ----------
        query : str
            Plain text user query.
        years : int | list[int] | None, optional
            Optional explicit year filter extracted from the user query.
        limit : int, default=20
            Maximum number of best matching chunks to retrieve.
        query_variants : tuple[str, ...] | list[str] | None, optional
            Additional faithful forms of the same query, such as the original
            multilingual question paired with an English retrieval translation.

        Returns
        -------
        tuple[list[Chunk], list[Document]]
            Curated chunks for answer grounding and a separate ranked document list.
        """
        table = await self.connection.open_table(self.table_name("chunks"))
        if hasattr(table, "schema"):
            chunk_schema = await table.schema()
            chunk_fields = {field.name for field in chunk_schema}
        else:
            chunk_fields = {
                "document_id",
                "title",
                "year",
                "language",
                "url",
                "summary",
                "content",
            }
        documents_table = await self.open_optional_table(self.table_name("documents"))
        profile = _build_retrieval_profile(query, years)
        source_filter_ids = tuple(
            sorted({str(source_id).strip() for source_id in (source_ids or []) if str(source_id).strip()})
        )
        source_filter_set = set(source_filter_ids)
        candidate_limit = max(limit * 3, 24)
        retrieval_queries = _retrieval_queries_with_variants(query, query_variants)
        lexical_timeout = _env_timeout_seconds(
            "RETRIEVE_CHUNKS_LEXICAL_TIMEOUT_SECONDS",
            6.0,
        )
        document_index_timeout = _env_timeout_seconds(
            "RETRIEVE_DOCUMENT_INDEX_TIMEOUT_SECONDS",
            max(lexical_timeout, 15.0),
        )
        per_variant_timeout = _env_timeout_seconds(
            "RETRIEVE_CHUNKS_VARIANT_TIMEOUT_SECONDS",
            12.0,
        )
        lexical_rows: list[dict] = []
        vector_rows: list[dict] = []

        logger.info(
            (
                "retrieve_chunks start query=%r variants=%r candidate_limit=%s "
                "lexical_timeout=%ss variant_timeout=%ss"
            ),
            query,
            retrieval_queries,
            candidate_limit,
            lexical_timeout,
            per_variant_timeout,
        )
        if debug is not None:
            debug.update(
                {
                    "query": query,
                    "limit": limit,
                    "candidate_limit": candidate_limit,
                    "profile": {
                        "normalized_query": profile.normalized_query,
                        "query_tokens": sorted(profile.query_tokens),
                        "query_phrases": list(profile.query_phrases),
                        "explicit_years": profile.explicit_years,
                        "prefer_recent": profile.prefer_recent,
                        "explanatory": profile.explanatory,
                        "intent": profile.intent,
                        "current_data_query": profile.current_data_query,
                        "data_focus": profile.data_focus,
                        "country_scopes": sorted(profile.country_scopes),
                        "region_scopes": sorted(profile.region_scopes),
                        "fallback_region_scopes": sorted(profile.fallback_region_scopes),
                        "has_geographic_scope": profile.has_geographic_scope,
                    },
                    "retrieval_queries": retrieval_queries,
                    "source_ids": list(source_filter_ids),
                    "documents_table_present": documents_table is not None,
                    "branches": [],
                    "selected": {},
                }
            )

        def apply_year_where(base_where: str | None = None) -> str | None:
            clauses: list[str] = []
            if base_where:
                clauses.append(f"({base_where})")
            if profile.explicit_years:
                if len(profile.explicit_years) == 1:
                    clauses.append(f"(year = {profile.explicit_years[0]})")
                else:
                    joined_years = ", ".join(map(str, profile.explicit_years))
                    clauses.append(f"(year IN ({joined_years}))")
            if source_filter_ids:
                source_clauses = []
                joined_sources = ", ".join(f"'{_sql_quote(value)}'" for value in source_filter_ids)
                if "source_id" in chunk_fields:
                    source_clauses.append(f"source_id IN ({joined_sources})")
                if "source" in chunk_fields:
                    source_clauses.append(f"source IN ({joined_sources})")
                if source_clauses:
                    clauses.append("(" + " OR ".join(source_clauses) + ")")
            if not clauses:
                return None
            return " AND ".join(clauses)

        document_index_rows: list[dict] | None = None

        def row_matches_source_filter(row: dict) -> bool:
            if not source_filter_set:
                return True
            values = {
                str(row.get("source_id") or "").strip(),
                str(row.get("source") or "").strip(),
            }
            return bool(values & source_filter_set)

        async def fetch_document_index_rows() -> list[dict]:
            nonlocal document_index_rows
            if documents_table is None:
                return []
            if document_index_rows is None:
                rows = await self.get_document_index_rows()
                document_index_rows = [
                    row for row in rows if row_matches_source_filter(row)
                ]
            return document_index_rows

        async def fetch_lexical_rows(retrieval_query: str) -> list[dict]:
            patterns = _build_metadata_patterns(retrieval_query)
            if not patterns:
                return []
            pattern_clauses = []
            for pattern in patterns:
                escaped = _sql_quote(pattern)
                pattern_clauses.append(f"regexp_like(title, '(?i){escaped}')")
                pattern_clauses.append(f"regexp_like(summary, '(?i){escaped}')")
            where_clause = apply_year_where(" OR ".join(pattern_clauses))
            select_fields = [
                field
                for field in [
                    "document_id",
                    "title",
                    "year",
                    "language",
                    "url",
                    "summary",
                    "content",
                    "source_id",
                    "source",
                    "region_codes",
                    "country_codes",
                    "geography_tags_text",
                    "topic_tags",
                    "publisher",
                    "document_type",
                    "publication_date",
                    "series_name",
                    "chunk_id",
                    "chunk_index",
                    "content_type",
                    "section_title",
                    "page_start",
                    "page_end",
                    "token_count",
                    "chunk_summary",
                ]
                if field in chunk_fields
            ]
            query_builder = table.query().select(select_fields)
            if where_clause:
                query_builder = query_builder.where(where_clause)
            return await query_builder.limit(candidate_limit).to_list()

        async def fetch_variant_rows(retrieval_query: str) -> list[dict]:
            vector = await self.embedder.aembed_query(retrieval_query)
            search = table.vector_search(vector)
            year_where = apply_year_where()
            if year_where:
                search = search.where(year_where)
            return await search.limit(candidate_limit).to_list()

        async def fetch_document_chunks(
            shortlisted_documents: list[dict],
            retrieval_query: str,
            *,
            lexical_only: bool,
        ) -> list[dict]:
            if not shortlisted_documents:
                return []
            document_ids = [
                row.get("document_id")
                for row in shortlisted_documents
                if isinstance(row.get("document_id"), str) and row.get("document_id")
            ]
            url_values = [
                (row.get("url") or "").strip()
                for row in shortlisted_documents
                if (row.get("url") or "").strip()
            ]
            title_values = [
                (row.get("canonical_title") or row.get("title") or "").strip()
                for row in shortlisted_documents
                if (row.get("canonical_title") or row.get("title") or "").strip()
            ]
            filters = []
            if "document_id" in chunk_fields and document_ids:
                joined = ", ".join(f"'{_sql_quote(value)}'" for value in document_ids)
                filters.append(f"document_id IN ({joined})")
            if "url" in chunk_fields and url_values:
                joined = ", ".join(f"'{_sql_quote(value)}'" for value in url_values)
                filters.append(f"url IN ({joined})")
            if "title" in chunk_fields and title_values:
                joined = ", ".join(f"'{_sql_quote(value)}'" for value in title_values)
                filters.append(f"title IN ({joined})")
            if not filters:
                return []

            base_filter = "(" + " OR ".join(filters) + ")"
            where_clause = apply_year_where(base_filter)
            scoped_limit = max(candidate_limit * 8, 240)

            if lexical_only:
                query_builder = table.query()
                if where_clause:
                    query_builder = query_builder.where(where_clause)
                return await query_builder.limit(scoped_limit).to_list()

            vector = await self.embedder.aembed_query(retrieval_query)
            search = table.vector_search(vector)
            if where_clause:
                search = search.where(where_clause)
            return await search.limit(candidate_limit).to_list()

        if documents_table is not None:
            try:
                index_rows = await asyncio.wait_for(
                    fetch_document_index_rows(),
                    timeout=document_index_timeout,
                )
            except asyncio.TimeoutError:
                index_rows = []
                logger.warning(
                    "retrieve_document_index timeout query=%r timeout=%ss",
                    query,
                    document_index_timeout,
                )
                if debug is not None:
                    debug["branches"].append(
                        {
                            "stage": "document_index_load",
                            "status": "timeout",
                            "timeout_seconds": document_index_timeout,
                        }
                    )
            except Exception as error:
                index_rows = []
                logger.warning(
                    "retrieve_document_index failed query=%r error=%s",
                    query,
                    error,
                )
                if debug is not None:
                    debug["branches"].append(
                        {
                            "stage": "document_index_load",
                            "status": "error",
                            "error": str(error),
                        }
                    )
            else:
                logger.info(
                    "retrieve_document_index loaded query=%r rows=%s",
                    query,
                    len(index_rows),
                )
                if debug is not None:
                    debug["branches"].append(
                        {
                            "stage": "document_index_load",
                            "rows": len(index_rows),
                            "source": "cache",
                        }
                    )

            ranked_documents: list[DocumentHit] = []
            ranked_document_debug: list[dict] = []
            if self.enable_sea_current_data_policy and _should_seed_latest_sdg7(profile):
                seeded_hits: list[tuple[DocumentHit, dict]] = []
                for row in index_rows:
                    if not _is_latest_tracking_sdg7_row(row):
                        continue
                    seeded_profile = _build_retrieval_profile(
                        "tracking sdg7 access to electricity latest data",
                        profile.explicit_years,
                    )
                    breakdown = _document_row_breakdown(row, seeded_profile)
                    score = max(float(breakdown["score"]), 20.0)
                    hit = DocumentHit(row=row, score=score)
                    ranked_documents.append(hit)
                    seeded_hits.append((hit, {**breakdown, "score": _round_debug(score)}))
                    if len(ranked_document_debug) < 20:
                        ranked_document_debug.append(
                            {
                                **breakdown,
                                "score": _round_debug(score),
                                "variant": "seed_latest_sdg7",
                                "topic_tags": row.get("topic_tags"),
                                "country_codes": row.get("country_codes"),
                                "region_codes": row.get("region_codes"),
                                "geography_mode": "seeded",
                            }
                        )
                if debug is not None:
                    debug["branches"].append(
                        {
                            "stage": "document_candidates_seed_latest_sdg7",
                            "rows": len(seeded_hits),
                            "sample_titles": [
                                hit.row.get("canonical_title") or hit.row.get("title")
                                for hit, _ in seeded_hits[:5]
                            ],
                        }
                    )
            for retrieval_query in retrieval_queries:
                variant_profile = _build_retrieval_profile(retrieval_query, profile.explicit_years)
                filtered_index_rows, geography_mode, rejected_rows = _filter_rows_by_geography(
                    index_rows,
                    variant_profile,
                )
                variant_hits: list[tuple[DocumentHit, dict]] = []
                for row in filtered_index_rows:
                    year = row.get("year")
                    if profile.explicit_years and year not in profile.explicit_years:
                        continue
                    breakdown = _document_row_breakdown(row, variant_profile)
                    if not _document_has_min_signal(row, variant_profile):
                        continue
                    score = float(breakdown["score"])
                    if score < 0.8:
                        continue
                    variant_hits.append((DocumentHit(row=row, score=score), breakdown))

                variant_hits.sort(key=lambda item: item[0].score, reverse=True)
                if debug is not None:
                    debug["branches"].append(
                        {
                            "stage": "document_candidates_index",
                            "variant": retrieval_query,
                            "geography_mode": geography_mode,
                            "rows": len(variant_hits),
                            "rejected_out_of_scope": rejected_rows[:10],
                            "sample_titles": [
                                item.row.get("canonical_title") or item.row.get("title")
                                for item, _ in variant_hits[:5]
                            ],
                        }
                    )
                logger.info(
                    "retrieve_document_candidates index query=%r variant=%r rows=%s",
                    query,
                    retrieval_query,
                    len(variant_hits),
                )
                if variant_profile.has_geographic_scope and not variant_hits:
                    logger.info(
                        "retrieve_document_candidates geography filtered query=%r variant=%r mode=%s rejected=%s",
                        query,
                        retrieval_query,
                        geography_mode,
                        len(rejected_rows),
                    )
                for hit, breakdown in variant_hits:
                    key = _document_identity_key(hit.row)
                    existing = next(
                        (
                            item
                            for item in ranked_documents
                            if _document_identity_key(item.row) == key
                        ),
                        None,
                    )
                    if existing is None or hit.score > existing.score:
                        if existing is not None:
                            ranked_documents.remove(existing)
                        ranked_documents.append(hit)
                    if len(ranked_document_debug) < 20:
                        ranked_document_debug.append(
                            {
                                **breakdown,
                                "variant": retrieval_query,
                                "topic_tags": hit.row.get("topic_tags"),
                                "country_codes": hit.row.get("country_codes"),
                                "region_codes": hit.row.get("region_codes"),
                                "geography_mode": geography_mode,
                            }
                        )

            ranked_documents.sort(key=lambda item: item.score, reverse=True)

            if ranked_documents:
                shortlisted_rows = _dedupe_document_rows(
                    [item.row for item in ranked_documents],
                    limit=6,
                )
                api_documents = [corpus.document_record_to_api(row) for row in shortlisted_rows]
                if debug is not None:
                    debug["selected"]["document_candidates"] = ranked_document_debug[:10]
                fast_metric_chunks = (
                    _build_current_data_metric_fallback_chunks(
                        shortlisted_rows,
                        profile,
                    )
                    if self.enable_sea_current_data_policy
                    else []
                )
                if fast_metric_chunks:
                    source_locked_rows = [
                        row for row in shortlisted_rows if _is_latest_tracking_sdg7_row(row)
                    ] or shortlisted_rows[:1]
                    source_locked_rows = _dedupe_document_rows(source_locked_rows)
                    source_locked_documents = [
                        corpus.document_record_to_api(row) for row in source_locked_rows
                    ] or api_documents[:1]
                    enrichment_queries = _build_current_data_enrichment_queries(profile)
                    enrichment_timeout = _env_timeout_seconds(
                        "RETRIEVE_CURRENT_DATA_ENRICHMENT_TIMEOUT_SECONDS",
                        min(lexical_timeout, 2.0),
                    )
                    enrichment_rows: list[dict] = []
                    logger.info(
                        "retrieve_chunks current-data metric-plus-context path query=%r docs=%s chunks=%s enrichment_queries=%s",
                        query,
                        len(source_locked_documents),
                        len(fast_metric_chunks),
                        enrichment_queries,
                    )
                    if debug is not None:
                        debug["branches"].append(
                            {
                                "stage": "current_data_metric_chunk",
                                "rows": len(fast_metric_chunks),
                                "sample_titles": [chunk.title for chunk in fast_metric_chunks],
                            }
                        )
                    for enrichment_query in enrichment_queries:
                        try:
                            rows = await asyncio.wait_for(
                                fetch_document_chunks(
                                    source_locked_rows,
                                    enrichment_query,
                                    lexical_only=True,
                                ),
                                timeout=enrichment_timeout,
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                "retrieve_source_locked_enrichment timeout query=%r variant=%r timeout=%ss",
                                query,
                                enrichment_query,
                                enrichment_timeout,
                            )
                            rows = []
                            status = "timeout"
                        except Exception as error:
                            logger.warning(
                                "retrieve_source_locked_enrichment failed query=%r variant=%r error=%s",
                                query,
                                enrichment_query,
                                error,
                            )
                            rows = []
                            status = "error"
                        else:
                            status = "ok"
                        rows = _filter_rows_for_documents(rows, source_locked_rows)
                        enrichment_rows = _merge_candidate_rows(enrichment_rows, rows)
                        logger.info(
                            "retrieve_source_locked_enrichment query=%r variant=%r rows=%s status=%s",
                            query,
                            enrichment_query,
                            len(rows),
                            status,
                        )
                        if debug is not None:
                            debug["branches"].append(
                                {
                                    "stage": "source_locked_enrichment_chunks",
                                    "variant": enrichment_query,
                                    "status": status,
                                    "rows": len(rows),
                                    "sample_titles": [
                                        row.get("title") or row.get("canonical_title")
                                        for row in rows[:5]
                                    ],
                                }
                            )
                    context_chunks = _select_source_context_chunks(
                        enrichment_rows,
                        profile,
                        limit=max(limit - len(fast_metric_chunks), 0),
                    )
                    trusted_context_chunks = _build_trusted_sdg7_context_fallback_chunks(
                        source_locked_rows,
                        profile,
                        existing_chunks=fast_metric_chunks + context_chunks,
                    )
                    if trusted_context_chunks and debug is not None:
                        debug["branches"].append(
                            {
                                "stage": "trusted_context_fallback",
                                "rows": len(trusted_context_chunks),
                                "sample_sections": [
                                    chunk.section_title for chunk in trusted_context_chunks
                                ],
                            }
                        )
                    selected_chunks = (
                        fast_metric_chunks + context_chunks + trusted_context_chunks
                    )[:limit]
                    if debug is not None:
                        debug["selected"]["documents"] = [
                            document.model_dump()
                            for document in source_locked_documents
                        ]
                        debug["selected"]["chunks"] = [
                            {
                                "document_id": chunk.document_id,
                                "chunk_id": chunk.chunk_id,
                                "title": chunk.title,
                                "section_title": chunk.section_title,
                                "year": chunk.year,
                                "url": chunk.url,
                                "content_type": chunk.content_type,
                            }
                            for chunk in selected_chunks
                        ]
                        debug["selected"]["path"] = "current_data_metric_plus_context"
                    return selected_chunks, source_locked_documents
                chunk_rows: list[dict] = []

                for retrieval_query in retrieval_queries:
                    try:
                        rows = await asyncio.wait_for(
                            fetch_document_chunks(shortlisted_rows, retrieval_query, lexical_only=True),
                            timeout=lexical_timeout,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "retrieve_document_chunks lexical timeout query=%r variant=%r timeout=%ss",
                            query,
                            retrieval_query,
                            lexical_timeout,
                        )
                        if debug is not None:
                            debug["branches"].append(
                                {
                                    "stage": "document_chunks_lexical",
                                    "variant": retrieval_query,
                                    "status": "timeout",
                                    "patterns": _build_metadata_patterns(retrieval_query),
                                }
                            )
                        continue
                    except Exception as error:
                        logger.warning(
                            "retrieve_document_chunks lexical failed query=%r variant=%r error=%s",
                            query,
                            retrieval_query,
                            error,
                        )
                        if debug is not None:
                            debug["branches"].append(
                                {
                                    "stage": "document_chunks_lexical",
                                    "variant": retrieval_query,
                                    "status": "error",
                                    "error": str(error),
                                    "patterns": _build_metadata_patterns(retrieval_query),
                                }
                            )
                        continue
                    for row in rows:
                        matching = next(
                            (
                                document
                                for document in shortlisted_rows
                                if (
                                    row.get("document_id")
                                    and row.get("document_id") == document.get("document_id")
                                )
                                or (
                                    (row.get("url") or "").strip()
                                    and (row.get("url") or "").strip() == (document.get("url") or "").strip()
                                )
                                or (
                                    (row.get("title") or "").strip()
                                    and (row.get("title") or "").strip()
                                    == (document.get("canonical_title") or document.get("title") or "").strip()
                                )
                            ),
                            None,
                        )
                        if matching:
                            row = {**matching, **row}
                        chunk_rows = _merge_candidate_rows(chunk_rows, [row])
                    if debug is not None:
                        debug["branches"].append(
                            {
                                "stage": "document_chunks_lexical",
                                "variant": retrieval_query,
                                "rows": len(rows),
                                "patterns": _build_metadata_patterns(retrieval_query),
                                "sample_titles": [
                                    row.get("title") or row.get("canonical_title")
                                    for row in rows[:5]
                                ],
                            }
                        )

                if not chunk_rows:
                    for retrieval_query in retrieval_queries:
                        try:
                            rows = await asyncio.wait_for(
                                fetch_document_chunks(shortlisted_rows, retrieval_query, lexical_only=False),
                                timeout=per_variant_timeout,
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                "retrieve_document_chunks vector timeout query=%r variant=%r timeout=%ss",
                                query,
                                retrieval_query,
                                per_variant_timeout,
                            )
                            if debug is not None:
                                debug["branches"].append(
                                    {
                                        "stage": "document_chunks_vector",
                                        "variant": retrieval_query,
                                        "status": "timeout",
                                        "patterns": _build_metadata_patterns(retrieval_query),
                                    }
                                )
                            continue
                        except Exception as error:
                            logger.warning(
                                "retrieve_document_chunks vector failed query=%r variant=%r error=%s",
                                query,
                                retrieval_query,
                                error,
                            )
                            if debug is not None:
                                debug["branches"].append(
                                    {
                                        "stage": "document_chunks_vector",
                                        "variant": retrieval_query,
                                        "status": "error",
                                        "error": str(error),
                                        "patterns": _build_metadata_patterns(retrieval_query),
                                    }
                                )
                            continue
                        for row in rows:
                            matching = next(
                                (
                                    document
                                    for document in shortlisted_rows
                                    if (
                                        row.get("document_id")
                                        and row.get("document_id") == document.get("document_id")
                                    )
                                    or (
                                        (row.get("url") or "").strip()
                                        and (row.get("url") or "").strip() == (document.get("url") or "").strip()
                                    )
                                    or (
                                        (row.get("title") or "").strip()
                                        and (row.get("title") or "").strip()
                                        == (document.get("canonical_title") or document.get("title") or "").strip()
                                    )
                                ),
                                None,
                            )
                            if matching:
                                row = {**matching, **row}
                            chunk_rows = _merge_candidate_rows(chunk_rows, [row])
                        if debug is not None:
                            debug["branches"].append(
                                {
                                    "stage": "document_chunks_vector",
                                    "variant": retrieval_query,
                                    "rows": len(rows),
                                    "patterns": _build_metadata_patterns(retrieval_query),
                                    "sample_titles": [
                                        row.get("title") or row.get("canonical_title")
                                        for row in rows[:5]
                                    ],
                                }
                            )

                chunk_rows, chunk_geography_mode, rejected_chunk_rows = _filter_rows_by_geography(
                    chunk_rows,
                    profile,
                )
                selected_chunks = _select_chunk_rows(chunk_rows, profile, limit=limit)
                metric_fallback_chunks = (
                    _build_current_data_metric_fallback_chunks(
                        shortlisted_rows,
                        profile,
                        existing_chunks=selected_chunks,
                    )
                    if self.enable_sea_current_data_policy
                    else []
                )
                if metric_fallback_chunks:
                    selected_chunks = (metric_fallback_chunks + selected_chunks)[:limit]
                    if debug is not None:
                        debug["branches"].append(
                            {
                                "stage": "current_data_metric_fallback",
                                "rows": len(metric_fallback_chunks),
                                "sample_titles": [chunk.title for chunk in metric_fallback_chunks],
                            }
                        )
                if not selected_chunks and api_documents:
                    selected_chunks = _build_summary_fallback_chunks(
                        shortlisted_rows,
                        profile,
                        limit=min(limit, len(shortlisted_rows)),
                    )
                logger.info(
                    "retrieve_chunks document-first selection query=%r docs=%s chunks=%s",
                    query,
                    len(api_documents),
                    len(selected_chunks),
                )
                if debug is not None:
                    debug["selected"]["documents"] = [
                        document.model_dump()
                        for document in api_documents
                    ]
                    debug["selected"]["chunks"] = [
                        {
                            "document_id": chunk.document_id,
                            "chunk_id": chunk.chunk_id,
                            "title": chunk.title,
                            "section_title": chunk.section_title,
                            "year": chunk.year,
                            "url": chunk.url,
                        }
                        for chunk in selected_chunks
                    ]
                    debug["selected"]["path"] = "document_first"
                    debug["selected"]["geography_mode"] = chunk_geography_mode
                    debug["selected"]["rejected_out_of_scope"] = rejected_chunk_rows[:10]
                    debug["selected"]["chunk_candidates"] = [
                        _chunk_score_breakdown(row, profile)
                        for row in chunk_rows[:10]
                    ]
                if selected_chunks or api_documents:
                    return selected_chunks, api_documents
                logger.warning(
                    "retrieve_chunks document-first produced no usable output query=%r top_candidates=%s",
                    query,
                    [
                        {
                            "title": item.row.get("canonical_title") or item.row.get("title"),
                            "score": round(item.score, 4),
                            "document_type": item.row.get("document_type"),
                            "status": item.row.get("status"),
                        }
                        for item in ranked_documents[:5]
                    ],
                )
            else:
                logger.warning(
                    "retrieve_chunks documents table present but no lexical document candidates query=%r variants=%r",
                    query,
                    retrieval_queries,
                )
                if debug is not None:
                    debug["selected"]["path"] = "document_first_no_candidates"

        for variant_index, retrieval_query in enumerate(retrieval_queries):
            try:
                variant_rows = await asyncio.wait_for(
                    fetch_lexical_rows(retrieval_query),
                    timeout=lexical_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "retrieve_chunks lexical timeout query=%r variant=%r timeout=%ss",
                    query,
                    retrieval_query,
                    lexical_timeout,
                )
                if debug is not None:
                    debug["branches"].append(
                        {
                            "stage": "chunk_candidates_lexical",
                            "variant": retrieval_query,
                            "status": "timeout",
                            "patterns": _build_metadata_patterns(retrieval_query),
                        }
                    )
                continue
            except Exception as error:
                logger.warning(
                    "retrieve_chunks lexical failed query=%r variant=%r error=%s",
                    query,
                    retrieval_query,
                    error,
                )
                if debug is not None:
                    debug["branches"].append(
                        {
                            "stage": "chunk_candidates_lexical",
                            "variant": retrieval_query,
                            "status": "error",
                            "error": str(error),
                            "patterns": _build_metadata_patterns(retrieval_query),
                        }
                    )
                continue

            logger.info(
                "retrieve_chunks lexical rows query=%r variant=%r rows=%s patterns=%s",
                query,
                retrieval_query,
                len(variant_rows),
                len(_build_metadata_patterns(retrieval_query)),
            )
            filtered_variant_rows, geography_mode, rejected_rows = _filter_rows_by_geography(
                variant_rows,
                profile,
            )
            if debug is not None:
                debug["branches"].append(
                    {
                        "stage": "chunk_candidates_lexical",
                        "variant": retrieval_query,
                        "rows": len(filtered_variant_rows),
                        "geography_mode": geography_mode,
                        "rejected_out_of_scope": rejected_rows[:10],
                        "patterns": _build_metadata_patterns(retrieval_query),
                        "sample_titles": [
                            row.get("title") or row.get("canonical_title")
                            for row in filtered_variant_rows[:5]
                        ],
                    }
                )
            adjusted_rows = []
            for row in filtered_variant_rows:
                adjusted_row = dict(row)
                adjusted_row["_distance"] = float(row.get("_distance", 0.35 + 0.03 * variant_index))
                adjusted_rows.append(adjusted_row)
            lexical_rows = _merge_candidate_rows(lexical_rows, adjusted_rows)

        if lexical_rows:
            chunks, documents = _select_documents_and_chunks(
                lexical_rows,
                profile,
                limit=limit,
                max_documents=6,
            )
            logger.info(
                "retrieve_chunks lexical selection query=%r rows=%s chunks=%s documents=%s",
                query,
                len(lexical_rows),
                len(chunks),
                len(documents),
            )
            if debug is not None:
                debug["selected"]["documents"] = [document.model_dump() for document in documents]
                debug["selected"]["chunks"] = [
                    {
                        "document_id": chunk.document_id,
                        "chunk_id": chunk.chunk_id,
                        "title": chunk.title,
                        "section_title": chunk.section_title,
                        "year": chunk.year,
                        "url": chunk.url,
                    }
                    for chunk in chunks
                ]
                debug["selected"]["path"] = "chunk_lexical"
                debug["selected"]["geography_mode"] = _allowed_geography_classes(
                    lexical_rows, profile
                )[1]
                debug["selected"]["chunk_candidates"] = [
                    _chunk_score_breakdown(row, profile)
                    for row in lexical_rows[:10]
                ]
            if chunks or documents:
                return chunks, documents

        for variant_index, retrieval_query in enumerate(retrieval_queries):
            try:
                variant_rows = await asyncio.wait_for(
                    fetch_variant_rows(retrieval_query),
                    timeout=per_variant_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "retrieve_chunks variant timeout query=%r variant=%r timeout=%ss",
                    query,
                    retrieval_query,
                    per_variant_timeout,
                )
                if debug is not None:
                    debug["branches"].append(
                        {
                            "stage": "chunk_candidates_vector",
                            "variant": retrieval_query,
                            "status": "timeout",
                            "patterns": _build_metadata_patterns(retrieval_query),
                        }
                    )
                continue
            except Exception as error:
                logger.warning(
                    "retrieve_chunks variant failed query=%r variant=%r error=%s",
                    query,
                    retrieval_query,
                    error,
                )
                if debug is not None:
                    debug["branches"].append(
                        {
                            "stage": "chunk_candidates_vector",
                            "variant": retrieval_query,
                            "status": "error",
                            "error": str(error),
                            "patterns": _build_metadata_patterns(retrieval_query),
                        }
                    )
                continue

            logger.info(
                "retrieve_chunks variant rows query=%r variant=%r rows=%s",
                query,
                retrieval_query,
                len(variant_rows),
            )
            filtered_variant_rows, geography_mode, rejected_rows = _filter_rows_by_geography(
                variant_rows,
                profile,
            )
            if debug is not None:
                debug["branches"].append(
                    {
                        "stage": "chunk_candidates_vector",
                        "variant": retrieval_query,
                        "rows": len(filtered_variant_rows),
                        "geography_mode": geography_mode,
                        "rejected_out_of_scope": rejected_rows[:10],
                        "patterns": _build_metadata_patterns(retrieval_query),
                        "sample_titles": [
                            row.get("title") or row.get("canonical_title")
                            for row in filtered_variant_rows[:5]
                        ],
                    }
                )

            adjusted_rows: list[dict] = []
            for row in filtered_variant_rows:
                variant_distance = float(row.get("_distance", 1.0) or 1.0)
                adjusted_row = dict(row)
                adjusted_row["_distance"] = max(0.0, variant_distance + 0.03 * variant_index)
                adjusted_rows.append(adjusted_row)
            vector_rows = _merge_candidate_rows(vector_rows, adjusted_rows)

        if vector_rows:
            chunks, documents = _select_documents_and_chunks(
                vector_rows,
                profile,
                limit=limit,
                max_documents=6,
            )
            logger.info(
                "retrieve_chunks vector selection query=%r rows=%s chunks=%s documents=%s",
                query,
                len(vector_rows),
                len(chunks),
                len(documents),
            )
            if debug is not None:
                debug["selected"]["documents"] = [document.model_dump() for document in documents]
                debug["selected"]["chunks"] = [
                    {
                        "document_id": chunk.document_id,
                        "chunk_id": chunk.chunk_id,
                        "title": chunk.title,
                        "section_title": chunk.section_title,
                        "year": chunk.year,
                        "url": chunk.url,
                    }
                    for chunk in chunks
                ]
                debug["selected"]["path"] = "chunk_vector"
                debug["selected"]["geography_mode"] = _allowed_geography_classes(
                    vector_rows, profile
                )[1]
                debug["selected"]["chunk_candidates"] = [
                    _chunk_score_breakdown(row, profile)
                    for row in vector_rows[:10]
                ]
            if chunks or documents:
                return chunks, documents

        logger.info(
            "retrieve_chunks fallback query=%r rows=%s",
            query,
            len(lexical_rows) + len(vector_rows),
        )
        chunks, documents = _select_documents_and_chunks(
            _merge_candidate_rows(lexical_rows, vector_rows),
            profile,
            limit=limit,
            max_documents=6,
        )
        if debug is not None:
            debug["selected"]["documents"] = [document.model_dump() for document in documents]
            debug["selected"]["chunks"] = [
                {
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "section_title": chunk.section_title,
                    "year": chunk.year,
                    "url": chunk.url,
                }
                for chunk in chunks
            ]
            debug["selected"]["path"] = "fallback"
            debug["selected"]["geography_mode"] = _allowed_geography_classes(
                _merge_candidate_rows(lexical_rows, vector_rows), profile
            )[1]
            debug["selected"]["chunk_candidates"] = [
                _chunk_score_breakdown(row, profile)
                for row in _merge_candidate_rows(lexical_rows, vector_rows)[:10]
            ]
        return chunks, documents

    async def get_sdg7_dataset(self) -> pd.DataFrame:
        """
        Get SDG 7 dataset.

        Returns
        -------
        pd.DataFrame
            Pandas data frame with SDG 7 indicators.
        """
        table = await self.connection.open_table("sdg7")
        df = await table.to_pandas()
        df.name = "indicators"
        return df

    async def get_knowledge_graph(self) -> nx.Graph:
        """
        Create a (weighted directed) knowledge graph from database tables.

        Returns
        -------
        nx.Graph
            Knowledge graph.
        """
        try:
            table = await self.connection.open_table("nodes")
        except ValueError as error:
            if "was not found" in str(error):
                logger.warning("Knowledge graph nodes table missing; using fallback local graph.")
                return build_fallback_knowledge_graph()
            raise
        df = await table.query().select(["name", "description", "weight"]).to_pandas()
        if df.empty:
            logger.warning("Knowledge graph nodes table is empty; using fallback local graph.")
            return build_fallback_knowledge_graph()
        nodes = zip(
            df["name"].tolist(),
            df[["description", "weight"]].to_dict(orient="records"),
        )
        try:
            table = await self.connection.open_table("edges")
        except ValueError as error:
            if "was not found" in str(error):
                logger.warning("Knowledge graph edges table missing; using fallback local graph.")
                return build_fallback_knowledge_graph()
            raise
        # check if level column exists in the table schema
        schema = await table.schema()
        edge_columns = ["subject", "object", "predicate", "description", "weight"]
        edge_attrs = ["predicate", "description", "weight"]
        if "level" in [field.name for field in schema]:
            edge_columns.append("level")
            edge_attrs.append("level")
        df = await table.query().select(edge_columns).to_pandas()
        if df.empty:
            logger.warning("Knowledge graph edges table is empty; using fallback local graph.")
            return build_fallback_knowledge_graph()
        edges = zip(
            df["subject"].tolist(),
            df["object"].tolist(),
            df[edge_attrs].to_dict(orient="records"),
        )
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph


# since the model uses the docstring, don't mention the artifacts there
@tool(parse_docstring=True, response_format="content_and_artifact")
async def retrieve_chunks(
    query: str, years: int | list[int] | None = None
) -> tuple[str, list[Document]]:
    """Retrieve relevant document chunks from the Sustainable Energy Academy database.

    The database can be used to answer questions on energy, climate change and
    sustainable development in general. Use the database to provide accurate and
    grounded responses. References are returned separately as structured document metadata,
    so do not include a source list or raw URLs in the answer body.

    Args:
        query (str): Plain text user query.
        years (Union[int, Tuple[int, ...], None]): Specific year or years the user refers
            to if applicable. If not provided, retrieval should infer whether recency
            should be preferred based on the query intent.

    Returns:
        str: JSON object containing the most relevant document chunks and references.
    """
    def normalize_years(value: int | list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if isinstance(value, int):
            return [value]
        clean = []
        for item in value:
            if isinstance(item, int):
                clean.append(item)
            elif isinstance(item, str) and item.isdigit():
                clean.append(int(item))
        return clean or None

    safe_years = normalize_years(years)
    timeout_seconds = _env_timeout_seconds(
        "RETRIEVE_CHUNKS_TIMEOUT_SECONDS",
        60.0,
    )
    connection = None
    try:
        connection = await get_connection()
        client = Client(connection)
        chunks, documents = await asyncio.wait_for(
            client.retrieve_chunks(query, safe_years),
            timeout=timeout_seconds,
        )
        data = json.dumps(
            {
                "chunks": [chunk.to_context() for chunk in chunks],
                "documents": [document.model_dump() for document in documents],
            }
        )
        return data, documents
    except asyncio.TimeoutError:
        logger.warning(
            "retrieve_chunks timed out after %ss for query=%r",
            timeout_seconds,
            query,
        )
        return "[]", []
    except Exception as error:
        logger.exception("retrieve_chunks tool failed: %s", error)
        # Return empty evidence payload so the agent can continue answering.
        return "[]", []
    finally:
        close = getattr(connection, "close", None)
        if callable(close):
            result = close()
            if isawaitable(result):
                with contextlib.suppress(Exception):
                    await result
