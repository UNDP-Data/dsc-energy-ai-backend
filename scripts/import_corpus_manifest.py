#!/usr/bin/env python3
"""
Import source, document, and chunk records from a local YAML manifest.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src import corpus, database
from src.entities import DocumentRecord, SourceRecord
from src.rag_system import get_profile


def _load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Manifest must be a mapping.")
    return payload


def _document_record_from_manifest(item: dict, *, profile) -> DocumentRecord:
    base_row = {
        "title": item.get("canonical_title") or item.get("title") or "",
        "year": int(item.get("year") or 0),
        "language": item.get("language") or "en",
        "url": item.get("url") or "",
        "summary": item.get("summary"),
        "content": item.get("content") or item.get("summary") or "",
    }
    record = corpus.build_document_record(
        [base_row],
        profile=profile,
        parser_version=item.get("parser_version", "manifest-v1"),
        embedding_version=item.get("embedding_version", "manifest-v1"),
        status=item.get("status", "approved"),
    )
    overrides = {
        "source_id": item.get("source_id", record.source_id),
        "document_type": item.get("document_type") or record.document_type,
        "canonical_title": item.get("canonical_title")
        or item.get("title")
        or record.canonical_title,
        "subtitle": item.get("subtitle"),
        "authors": item.get("authors"),
        "publisher": item.get("publisher") or record.publisher,
        "series_name": item.get("series_name") or record.series_name,
        "series_id": item.get("series_id") or record.series_id,
        "country_codes": item.get("country_codes") or record.country_codes,
        "region_codes": item.get("region_codes") or record.region_codes,
        "topic_tags": item.get("topic_tags") or record.topic_tags,
        "sdg_tags": item.get("sdg_tags") or record.sdg_tags,
        "sector_tags": item.get("sector_tags") or record.sector_tags,
        "audience_tags": item.get("audience_tags") or record.audience_tags,
        "source_priority": item.get("source_priority", record.source_priority),
        "quality_score": item.get("quality_score", record.quality_score),
        "is_flagship": item.get("is_flagship", record.is_flagship),
        "is_data_report": item.get("is_data_report", record.is_data_report),
        "has_tables": item.get("has_tables", record.has_tables),
        "has_figures": item.get("has_figures", record.has_figures),
        "page_count": item.get("page_count", record.page_count),
        "review_notes": item.get("review_notes") or record.review_notes,
        "status": item.get("status", record.status),
        "project_ids": item.get("project_ids"),
        "project_numbers": item.get("project_numbers"),
        "data_classification": item.get("data_classification"),
        "publication_status": item.get("publication_status"),
        "review_state": item.get("review_state"),
        "review_required": item.get("review_required"),
        "sensitive_content_flags": item.get("sensitive_content_flags"),
        "validation_category": item.get("validation_category"),
        "classifier_version": item.get("classifier_version"),
        "ruleset_version": (
            str(item["ruleset_version"])
            if item.get("ruleset_version") is not None
            else None
        ),
        "ruleset_revision": item.get("ruleset_revision"),
        "policy_version": item.get("policy_version"),
        "source_sha256": item.get("source_sha256"),
    }
    if overrides["topic_tags"]:
        overrides["topic_tags_text"] = " | ".join(overrides["topic_tags"])
    if overrides["region_codes"] or overrides["country_codes"]:
        overrides["geography_tags_text"] = " | ".join(
            (overrides["region_codes"] or []) + (overrides["country_codes"] or [])
        )
    if overrides["audience_tags"]:
        overrides["audience_tags_text"] = " | ".join(overrides["audience_tags"])
    if item.get("document_id"):
        overrides["document_id"] = item["document_id"]
    return DocumentRecord.model_validate({**record.model_dump(), **overrides})


def _source_record_from_manifest(item: dict) -> SourceRecord:
    return SourceRecord(
        source_id=item["source_id"],
        name=item["name"],
        organization=item["organization"],
        authority_tier=item.get("authority_tier", "partner"),
        license_policy=item.get("license_policy"),
        robots_policy=item.get("robots_policy"),
        base_url=item.get("base_url"),
        ingestion_method=item.get("ingestion_method", "manual_manifest"),
        enabled=item.get("enabled", True),
        review_policy=item.get("review_policy", "hybrid_editorial"),
        created_at=item.get("created_at"),
        updated_at=item.get("updated_at"),
    )


def _row_for_chunk(chunk: dict, document: DocumentRecord) -> dict:
    chunk_identifier = chunk.get("chunk_id")
    if (
        not chunk_identifier
        and chunk.get("document_id")
        and chunk.get("document_id") != document.document_id
    ):
        # Older manifests used chunk.document_id as the chunk identifier.
        chunk_identifier = chunk.get("document_id")
    row = {
        "document_id": document.document_id,
        "title": document.canonical_title,
        "year": document.year,
        "language": document.language,
        "url": document.url,
        "summary": document.summary,
        "source_id": document.source_id,
        "publisher": document.publisher,
        "document_type": document.document_type,
        "publication_date": document.publication_date,
        "series_name": document.series_name,
        "topic_tags": document.topic_tags,
        "region_codes": document.region_codes,
        "country_codes": document.country_codes,
        "status": document.status,
        "project_ids": document.project_ids,
        "project_numbers": document.project_numbers,
        "data_classification": document.data_classification,
        "publication_status": document.publication_status,
        "review_state": document.review_state,
        "review_required": document.review_required,
        "sensitive_content_flags": document.sensitive_content_flags,
        "validation_category": document.validation_category,
        "classifier_version": document.classifier_version,
        "ruleset_version": document.ruleset_version,
        "ruleset_revision": document.ruleset_revision,
        "policy_version": document.policy_version,
        "source_sha256": document.source_sha256,
        "content": chunk.get("content") or chunk.get("text") or "",
        "section_title": chunk.get("section_title") or chunk.get("section_heading"),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "content_type": chunk.get("content_type", "text"),
        "chunk_summary": chunk.get("chunk_summary"),
        "token_count": chunk.get("token_count"),
    }
    if chunk_identifier:
        row["chunk_id"] = chunk_identifier
    return row


def _chunk_rows_for_document(
    item: dict,
    document: DocumentRecord,
    *,
    profile,
    allow_summary_fallback: bool = True,
) -> list[dict]:
    chunks = item.get("chunks")
    rows = []
    if isinstance(chunks, list) and chunks:
        rows = [
            _row_for_chunk(chunk, document)
            for chunk in chunks
            if isinstance(chunk, dict)
        ]
    elif allow_summary_fallback and (item.get("content") or item.get("summary")):
        rows = [
            _row_for_chunk(
                {"content": item.get("content") or item.get("summary") or ""}, document
            )
        ]
    if not rows:
        return []
    return corpus.enrich_chunk_rows(
        rows,
        documents_by_key={document.url or document.canonical_title: document},
        profile=profile,
    )


def _manifest_chunk_paths(
    manifest_path: Path, manifest: dict, cli_paths: list[str] | None
) -> list[Path]:
    paths = [Path(item).expanduser() for item in cli_paths or []]
    for item in manifest.get("chunk_files") or []:
        if not isinstance(item, str) or not item.strip():
            continue
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = manifest_path.parent / path
        paths.append(path)
    clean = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        clean.append(resolved)
    return clean


def _chunk_rows_from_jsonl(
    paths: list[Path],
    documents_by_id: dict[str, DocumentRecord],
    *,
    profile,
) -> list[dict]:
    rows_by_document: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Chunk JSONL not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if not isinstance(chunk, dict):
                    continue
                document_id = str(chunk.get("document_id") or "")
                document = documents_by_id.get(document_id)
                if document is None:
                    raise ValueError(
                        f"Unknown document_id {document_id!r} in {path}:{line_number}"
                    )
                rows_by_document[document_id].append(_row_for_chunk(chunk, document))

    enriched: list[dict] = []
    for document_id in sorted(rows_by_document):
        document = documents_by_id[document_id]
        enriched.extend(
            corpus.enrich_chunk_rows(
                rows_by_document[document_id],
                documents_by_key={document.url or document.canonical_title: document},
                profile=profile,
            )
        )
    return enriched


async def _run(args) -> dict[str, int]:
    print(f"[import] Loading manifest: {args.manifest}", flush=True)
    manifest = _load_manifest(Path(args.manifest))
    assistant_id = args.assistant_id or manifest.get("assistant_id") or "sea"
    profile = get_profile(assistant_id)
    print(f"[import] Using assistant profile: {profile.assistant_id}", flush=True)
    source_records = [
        _source_record_from_manifest(item)
        for item in manifest.get("sources", [])
        if isinstance(item, dict)
    ]
    document_records = []
    chunk_rows = []
    documents_by_id = {}
    chunk_paths = (
        _manifest_chunk_paths(Path(args.manifest), manifest, args.chunks_jsonl)
        if args.include_chunks
        else []
    )
    for item in manifest.get("documents", []):
        if not isinstance(item, dict):
            continue
        document = _document_record_from_manifest(item, profile=profile)
        document_records.append(document)
        documents_by_id[document.document_id] = document
        if args.include_chunks:
            chunk_rows.extend(
                _chunk_rows_for_document(
                    item,
                    document,
                    profile=profile,
                    allow_summary_fallback=not bool(chunk_paths),
                )
            )
    if args.include_chunks and chunk_paths:
        chunk_rows.extend(
            _chunk_rows_from_jsonl(chunk_paths, documents_by_id, profile=profile)
        )

    inferred_sources = corpus.build_source_records_for_documents(
        document_records,
        profile=profile,
    )
    by_source = {record.source_id: record for record in inferred_sources}
    for record in source_records:
        by_source[record.source_id] = record

    print(
        f"[import] Prepared {len(document_records)} documents and {len(chunk_rows)} chunks for import.",
        flush=True,
    )
    print("[import] Opening database connection...", flush=True)
    connection = await database.get_connection(profile=profile)
    try:
        client = database.Client(connection, profile=profile)
        print("[import] Upserting sources...", flush=True)
        source_count = await client.upsert_sources(
            [item.model_dump() for item in by_source.values()]
        )
        print("[import] Upserting documents...", flush=True)
        document_count = await client.upsert_documents(
            [item.model_dump() for item in document_records]
        )
        chunk_count = 0
        if args.include_chunks:
            print("[import] Upserting chunks...", flush=True)
            chunk_count = await client.upsert_chunks(chunk_rows)
        return {
            "sources": source_count,
            "documents": document_count,
            "chunks": chunk_count,
        }
    finally:
        print("[import] Closing database connection...", flush=True)
        close = getattr(connection, "close", None)
        if callable(close):
            result = close()
            if asyncio.iscoroutine(result):
                await result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", required=True, help="Path to YAML corpus manifest."
    )
    parser.add_argument(
        "--assistant-id",
        default=None,
        help="RAG assistant profile id. Defaults to manifest assistant_id or sea.",
    )
    parser.add_argument(
        "--include-chunks",
        action="store_true",
        help="Also upsert chunk rows generated from the manifest or external chunk JSONL files.",
    )
    parser.add_argument(
        "--chunks-jsonl",
        action="append",
        default=[],
        help="External chunk JSONL file. May be repeated. Relative manifest chunk_files are also used.",
    )
    args = parser.parse_args()
    print(json.dumps(asyncio.run(_run(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
