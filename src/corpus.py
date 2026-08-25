"""
Corpus metadata enrichment and bootstrap helpers.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid5

from .entities import Document, DocumentRecord, SourceRecord

TOKEN_RE = re.compile(r"[a-z0-9]+")

TOPIC_RULES: dict[str, tuple[str, ...]] = {
    "energy access": (
        "energy access",
        "electricity access",
        "electrification",
        "mini-grid",
        "off-grid",
    ),
    "renewable energy": (
        "renewable",
        "solar",
        "wind",
        "hydro",
        "geothermal",
        "bioenergy",
    ),
    "energy efficiency": ("energy efficiency", "energy intensity", "efficiency"),
    "climate mitigation": (
        "climate mitigation",
        "decarbon",
        "emissions",
        "greenhouse gas",
    ),
    "climate adaptation": (
        "climate adaptation",
        "resilience",
        "adaptive capacity",
        "disaster risk",
    ),
    "energy finance": ("finance", "investment", "green bond", "concessional", "tariff"),
    "grid infrastructure": (
        "grid",
        "transmission",
        "distribution",
        "storage",
        "interconnection",
    ),
    "clean cooking": ("clean cooking", "cookstove", "biofuel", "traditional biomass"),
}

GEOGRAPHY_RULES: dict[str, tuple[str, ...]] = {
    "africa": ("africa", "sub-saharan africa"),
    "asia": ("asia", "south asia", "southeast asia", "central asia"),
    "latin america": ("latin america", "caribbean", "lac"),
    "europe": ("europe", "european union"),
    "middle east": ("middle east", "mena", "arab states"),
    "small island developing states": ("sids", "small island developing states"),
    "global": ("global", "worldwide", "international"),
}

COUNTRY_RULES: dict[str, tuple[str, ...]] = {
    "NGA": ("nigeria",),
    "COD": ("democratic republic of congo", "dr congo", "drc"),
    "ETH": ("ethiopia",),
    "KEN": ("kenya",),
    "IND": ("india",),
    "IDN": ("indonesia",),
    "BRA": ("brazil",),
}

SDG_RULES: dict[str, tuple[str, ...]] = {
    "SDG7": (
        "sdg 7",
        "sustainable development goal 7",
        "energy access",
        "renewable",
        "clean cooking",
    ),
    "SDG13": ("sdg 13", "climate action", "mitigation", "adaptation"),
}

SECTOR_RULES: dict[str, tuple[str, ...]] = {
    "power": ("electricity", "power", "grid", "renewable power"),
    "transport": ("transport", "mobility", "vehicle", "biofuel"),
    "industry": ("industry", "industrial", "manufacturing"),
    "buildings": ("building", "buildings", "cooling", "heating"),
    "cooking": ("clean cooking", "cookstove", "cooking"),
    "finance": ("finance", "investment", "bond", "loan"),
}

AUDIENCE_RULES: dict[str, tuple[str, ...]] = {
    "policy-makers": ("policy", "roadmap", "guidance", "framework", "regulation"),
    "practitioners": (
        "toolkit",
        "implementation",
        "case study",
        "programme",
        "project",
    ),
    "researchers": ("analysis", "report", "assessment", "scenario"),
}


@dataclass(frozen=True)
class SourceDefinition:
    source_id: str
    name: str
    organization: str
    authority_tier: str
    base_url: str
    ingestion_method: str
    review_policy: str = "hybrid_editorial"
    license_policy: str | None = None
    robots_policy: str | None = None
    domains: tuple[str, ...] = ()


SOURCE_DEFINITIONS: tuple[SourceDefinition, ...] = (
    SourceDefinition(
        source_id="tracking_sdg7",
        name="Tracking SDG7 / ESMAP",
        organization="World Bank / ESMAP",
        authority_tier="trusted",
        base_url="https://trackingsdg7.esmap.org",
        ingestion_method="official_index",
        domains=("trackingsdg7.esmap.org", "esmap.org"),
    ),
    SourceDefinition(
        source_id="undp",
        name="UNDP Publications",
        organization="UNDP",
        authority_tier="trusted",
        base_url="https://www.undp.org",
        ingestion_method="official_index",
        domains=("undp.org",),
    ),
    SourceDefinition(
        source_id="world_bank",
        name="World Bank Publications",
        organization="World Bank",
        authority_tier="trusted",
        base_url="https://www.worldbank.org",
        ingestion_method="official_index",
        domains=("worldbank.org",),
    ),
    SourceDefinition(
        source_id="iea",
        name="IEA Publications",
        organization="International Energy Agency",
        authority_tier="partner",
        base_url="https://www.iea.org",
        ingestion_method="official_index",
        domains=("iea.org",),
    ),
    SourceDefinition(
        source_id="seforall",
        name="SEforALL Publications",
        organization="Sustainable Energy for All",
        authority_tier="partner",
        base_url="https://www.seforall.org",
        ingestion_method="official_index",
        domains=("seforall.org",),
    ),
    SourceDefinition(
        source_id="irena",
        name="IRENA Publications",
        organization="IRENA",
        authority_tier="partner",
        base_url="https://www.irena.org",
        ingestion_method="official_index",
        domains=("irena.org",),
    ),
    SourceDefinition(
        source_id="unep",
        name="UNEP Publications",
        organization="UNEP",
        authority_tier="partner",
        base_url="https://www.unep.org",
        ingestion_method="official_index",
        domains=("unep.org",),
    ),
)


def _source_definitions_from_profile(profile=None) -> tuple[SourceDefinition, ...]:
    """
    Return source definitions from a RAG profile, falling back to SEA defaults.
    """
    raw_definitions = getattr(profile, "source_definitions", None)
    if not raw_definitions:
        return SOURCE_DEFINITIONS
    definitions: list[SourceDefinition] = []
    for item in raw_definitions:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id") or "").strip()
        name = str(item.get("name") or source_id).strip()
        organization = str(item.get("organization") or name).strip()
        if not source_id or not name:
            continue
        domains = item.get("domains")
        definitions.append(
            SourceDefinition(
                source_id=source_id,
                name=name,
                organization=organization,
                authority_tier=str(item.get("authority_tier") or "partner"),
                base_url=str(item.get("base_url") or ""),
                ingestion_method=str(item.get("ingestion_method") or "manual_manifest"),
                review_policy=str(item.get("review_policy") or "hybrid_editorial"),
                license_policy=item.get("license_policy"),
                robots_policy=item.get("robots_policy"),
                domains=(
                    tuple(value for value in domains if isinstance(value, str))
                    if isinstance(domains, list)
                    else ()
                ),
            )
        )
    return tuple(definitions) or SOURCE_DEFINITIONS


def _rules_from_profile(
    profile, group: str, default: dict[str, tuple[str, ...]]
) -> dict[str, tuple[str, ...]]:
    """
    Return tag rules from a profile, preserving legacy defaults when absent.
    """
    tag_rules = getattr(profile, "tag_rules", None)
    if not isinstance(tag_rules, dict):
        return default
    raw_group = tag_rules.get(group)
    if not isinstance(raw_group, dict):
        return default
    rules: dict[str, tuple[str, ...]] = {}
    for tag, patterns in raw_group.items():
        if isinstance(patterns, str):
            clean_patterns = (patterns,)
        elif isinstance(patterns, list):
            clean_patterns = tuple(
                item for item in patterns if isinstance(item, str) and item.strip()
            )
        else:
            clean_patterns = ()
        if clean_patterns:
            rules[str(tag)] = clean_patterns
    return rules or default


def _utcnow_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def _normalize_text(text: str | None) -> str:
    return " ".join(TOKEN_RE.findall((text or "").lower()))


def _join_unique(values: list[str]) -> str | None:
    clean = []
    seen = set()
    for value in values:
        value = value.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        clean.append(value)
    return " | ".join(clean) if clean else None


def _tag_matches(text: str, rules: dict[str, tuple[str, ...]]) -> list[str]:
    matches = []
    for tag, patterns in rules.items():
        if any(pattern in text for pattern in patterns):
            matches.append(tag)
    return matches


def _infer_document_type(title: str, summary: str, content: str) -> str:
    text = " ".join(
        filter(
            None,
            [
                _normalize_text(title),
                _normalize_text(summary),
                _normalize_text(content),
            ],
        )
    )
    if any(
        pattern in text
        for pattern in ("policy brief", "guidance", "toolkit", "roadmap", "framework")
    ):
        return "policy"
    if any(
        pattern in text
        for pattern in ("case study", "programme", "program", "initiative")
    ):
        return "case_study"
    if any(
        pattern in text
        for pattern in (
            "tracking sdg7",
            "progress toward sustainable energy",
            "annual report",
        )
    ):
        return "flagship_report"
    if "report" in text:
        return "report"
    return "publication"


def _infer_series(title: str) -> tuple[str | None, str | None]:
    normalized = _normalize_text(title)
    if (
        "tracking sdg7" in normalized
        or "progress toward sustainable energy" in normalized
    ):
        return "Tracking SDG7", "tracking_sdg7"
    if "world energy outlook" in normalized:
        return "World Energy Outlook", "world_energy_outlook"
    return None, None


def _infer_quality_score(
    summary: str | None, content: str | None, url: str | None
) -> float:
    score = 0.3
    if summary:
        score += 0.2
    if content and len(content.split()) >= 80:
        score += 0.25
    if url:
        score += 0.15
    if content and any(char.isdigit() for char in content):
        score += 0.1
    return min(score, 1.0)


def match_source(url: str | None, *, profile=None) -> SourceDefinition:
    domain = urlparse((url or "").strip()).netloc.lower()
    for definition in _source_definitions_from_profile(profile):
        if any(domain.endswith(candidate) for candidate in definition.domains):
            return definition
    return SourceDefinition(
        source_id="manual_external",
        name="Manual / External",
        organization="External",
        authority_tier="external",
        base_url=f"https://{domain}" if domain else "",
        ingestion_method="manual_manifest",
        domains=(domain,) if domain else (),
    )


def build_source_record(
    source: SourceDefinition, *, timestamp: str | None = None
) -> SourceRecord:
    now = timestamp or _utcnow_iso()
    return SourceRecord(
        source_id=source.source_id,
        name=source.name,
        organization=source.organization,
        authority_tier=source.authority_tier,
        license_policy=source.license_policy,
        robots_policy=source.robots_policy,
        base_url=source.base_url,
        ingestion_method=source.ingestion_method,
        enabled=True,
        review_policy=source.review_policy,
        created_at=now,
        updated_at=now,
    )


def _stable_document_id(source_id: str, url: str, title: str, year: int) -> str:
    seed = f"{source_id}|{url or title}|{year}"
    return str(uuid5(NAMESPACE_URL, seed))


def _stable_chunk_id(document_id: str, chunk_index: int, content: str) -> str:
    seed = f"{document_id}|{chunk_index}|{hashlib.sha1(content.encode('utf-8')).hexdigest()[:16]}"
    return str(uuid5(NAMESPACE_URL, seed))


def document_key_from_row(row: dict) -> str:
    return ((row.get("url") or "").strip() or (row.get("title") or "").strip()).strip()


def build_document_record(
    rows: list[dict],
    *,
    profile=None,
    parser_version: str = "bootstrap-v1",
    embedding_version: str = "legacy",
    status: str = "approved",
    timestamp: str | None = None,
) -> DocumentRecord:
    if not rows:
        raise ValueError("Expected at least one row to build a document record.")
    now = timestamp or _utcnow_iso()
    best = rows[0]
    title = (best.get("title") or "").strip() or "Untitled document"
    url = (best.get("url") or "").strip()
    language = (best.get("language") or "").strip() or "en"
    year = int(best.get("year") or 0)
    summary = next(
        (
            (row.get("summary") or "").strip()
            for row in rows
            if (row.get("summary") or "").strip()
        ),
        "",
    )
    content = "\n".join(
        (row.get("content") or "").strip()
        for row in rows
        if (row.get("content") or "").strip()
    )
    source = match_source(url, profile=profile)
    source_record = build_source_record(source, timestamp=now)
    document_type = _infer_document_type(title, summary, content)
    series_name, series_id = _infer_series(title)
    text = " ".join(
        filter(
            None,
            [
                _normalize_text(title),
                _normalize_text(summary),
                _normalize_text(content),
            ],
        )
    )
    topic_tags = (
        _tag_matches(text, _rules_from_profile(profile, "topics", TOPIC_RULES)) or None
    )
    geographies = (
        _tag_matches(text, _rules_from_profile(profile, "geographies", GEOGRAPHY_RULES))
        or None
    )
    country_codes = (
        _tag_matches(text, _rules_from_profile(profile, "countries", COUNTRY_RULES))
        or None
    )
    sdg_tags = (
        _tag_matches(text, _rules_from_profile(profile, "sdgs", SDG_RULES)) or None
    )
    sector_tags = (
        _tag_matches(text, _rules_from_profile(profile, "sectors", SECTOR_RULES))
        or None
    )
    audience_tags = (
        _tag_matches(text, _rules_from_profile(profile, "audiences", AUDIENCE_RULES))
        or None
    )
    publication_date = f"{year:04d}-01-01" if year else None
    content_hash = hashlib.sha256(
        "\n".join([title, url, summary, content]).encode("utf-8")
    ).hexdigest()
    document_id = _stable_document_id(source_record.source_id, url, title, year)
    normalized_title = _normalize_text(title)
    is_flagship = bool(series_id) or "tracking sdg7" in normalized_title
    is_data_report = document_type in {"flagship_report", "report"} and bool(
        sdg_tags or ("SDG7" in (sdg_tags or []))
    )
    quality_score = _infer_quality_score(summary, content, url)
    review_notes = None
    if not summary:
        review_notes = "Auto-generated record missing summary; review recommended."

    return DocumentRecord(
        document_id=document_id,
        source_id=source_record.source_id,
        canonical_title=title,
        url=url,
        language=language,
        document_type=document_type,
        publication_date=publication_date,
        year=year,
        summary=summary or None,
        status=status,
        ingested_at=now,
        updated_at=now,
        content_hash=content_hash,
        parser_version=parser_version,
        embedding_version=embedding_version,
        publisher=source_record.organization,
        series_name=series_name,
        series_id=series_id,
        country_codes=country_codes,
        region_codes=geographies,
        topic_tags=topic_tags,
        sdg_tags=sdg_tags,
        sector_tags=sector_tags,
        audience_tags=audience_tags,
        source_priority=1.0 if source_record.authority_tier == "trusted" else 0.7,
        quality_score=quality_score,
        dedupe_group_id=str(uuid5(NAMESPACE_URL, (url or normalized_title or title))),
        is_flagship=is_flagship,
        is_data_report=is_data_report,
        has_tables=any(
            token in text for token in ("data", "table", "indicator", "statistics")
        ),
        has_figures=any(token in text for token in ("figure", "chart", "graph")),
        page_count=None,
        review_notes=review_notes,
        topic_tags_text=_join_unique(topic_tags or []),
        geography_tags_text=_join_unique((geographies or []) + (country_codes or [])),
        audience_tags_text=_join_unique(audience_tags or []),
    )


def build_source_records_for_documents(
    documents: list[DocumentRecord], *, profile=None
) -> list[SourceRecord]:
    timestamp = _utcnow_iso()
    records: dict[str, SourceRecord] = {}
    definitions = {
        definition.source_id: definition
        for definition in _source_definitions_from_profile(profile)
    }
    definitions["manual_external"] = match_source("", profile=profile)
    for document in documents:
        definition = definitions.get(document.source_id)
        if definition is None:
            definition = SourceDefinition(
                source_id=document.source_id,
                name=document.source_id.replace("_", " ").title(),
                organization=document.publisher or "External",
                authority_tier="external",
                base_url="",
                ingestion_method="manual_manifest",
            )
        records[document.source_id] = build_source_record(
            definition, timestamp=timestamp
        )
    return sorted(records.values(), key=lambda item: item.source_id)


def enrich_chunk_rows(
    rows: list[dict],
    *,
    documents_by_key: dict[str, DocumentRecord],
    profile=None,
) -> list[dict]:
    enriched = []
    per_document_index: dict[str, int] = {}
    for row in rows:
        key = document_key_from_row(row)
        document = documents_by_key.get(key)
        if document is None:
            document = build_document_record([row], profile=profile)
            documents_by_key[key] = document
        chunk_index = per_document_index.get(document.document_id, 0)
        per_document_index[document.document_id] = chunk_index + 1
        content = (row.get("content") or "").strip()
        enriched.append(
            {
                **row,
                "document_id": document.document_id,
                "chunk_id": row.get("chunk_id")
                or _stable_chunk_id(document.document_id, chunk_index, content),
                "chunk_index": chunk_index,
                "content_type": row.get("content_type") or "text",
                "section_title": row.get("section_title"),
                "page_start": row.get("page_start"),
                "page_end": row.get("page_end"),
                "token_count": row.get("token_count") or len(content.split()),
                "chunk_summary": row.get("chunk_summary") or row.get("summary"),
            }
        )
    return enriched


def document_record_to_api(record: DocumentRecord | dict) -> Document:
    row = record.model_dump() if hasattr(record, "model_dump") else dict(record)
    geographies = []
    for value in (row.get("region_codes") or []) + (row.get("country_codes") or []):
        if isinstance(value, str) and value not in geographies:
            geographies.append(value)
    return Document(
        document_id=row.get("document_id"),
        source=row.get("source_id") or row.get("source"),
        publisher=row.get("publisher"),
        title=row.get("canonical_title") or row.get("title") or "",
        year=int(row.get("year") or 0),
        language=row.get("language") or "",
        url=row.get("url") or "",
        summary=row.get("summary"),
        document_type=row.get("document_type"),
        publication_date=row.get("publication_date"),
        series_name=row.get("series_name"),
        topics=row.get("topic_tags") or row.get("topics"),
        geographies=geographies or row.get("geographies"),
    )
