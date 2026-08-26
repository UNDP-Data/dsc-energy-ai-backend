#!/usr/bin/env python3
"""Safely replace selected assistant-corpus source partitions in LanceDB.

The command validates a local assistant-kit manifest, prepares canonical source,
document, and chunk rows, embeds every chunk that lacks a compatible vector,
builds isolated staging tables, validates source-level counts, and swaps the
staging tables into place. It never treats an upsert as a partition replacement.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import sys
from array import array
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pyarrow as pa
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
load_dotenv(ROOT / ".env")

from import_corpus_manifest import _chunk_rows_for_document  # noqa: E402
from import_corpus_manifest import (
    _chunk_rows_from_jsonl,
    _document_record_from_manifest,
    _load_manifest,
    _manifest_chunk_paths,
    _source_record_from_manifest,
)

from src import corpus, database, genai  # noqa: E402
from src.entities import DocumentRecord, SourceRecord  # noqa: E402
from src.rag_system import get_profile  # noqa: E402

ALLOWED_REPLACE_SOURCE_IDS = {"gef_sgp_intranet_projects"}
DEFAULT_VECTOR_DIMENSIONS = 1_024


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sql_quote(value: str) -> str:
    return value.replace("'", "''")


def _chunk_schema(dimensions: int) -> pa.Schema:
    list_of_strings = pa.list_(pa.string())
    return pa.schema(
        [
            pa.field("document_id", pa.string(), nullable=False),
            pa.field("title", pa.string(), nullable=False),
            pa.field("year", pa.int64(), nullable=False),
            pa.field("language", pa.string(), nullable=False),
            pa.field("url", pa.string(), nullable=False),
            pa.field("summary", pa.string()),
            pa.field("source_id", pa.string(), nullable=False),
            pa.field("publisher", pa.string()),
            pa.field("document_type", pa.string()),
            pa.field("publication_date", pa.string()),
            pa.field("series_name", pa.string()),
            pa.field("topic_tags", list_of_strings),
            pa.field("region_codes", list_of_strings),
            pa.field("country_codes", list_of_strings),
            pa.field("status", pa.string(), nullable=False),
            pa.field("project_ids", list_of_strings),
            pa.field("project_numbers", list_of_strings),
            pa.field("data_classification", pa.string()),
            pa.field("publication_status", pa.string()),
            pa.field("review_state", pa.string()),
            pa.field("review_required", pa.bool_()),
            pa.field("sensitive_content_flags", list_of_strings),
            pa.field("validation_category", pa.string()),
            pa.field("classifier_version", pa.string()),
            pa.field("ruleset_version", pa.string()),
            pa.field("ruleset_revision", pa.string()),
            pa.field("policy_version", pa.string()),
            pa.field("source_sha256", pa.string()),
            pa.field("content", pa.string(), nullable=False),
            pa.field("section_title", pa.string()),
            pa.field("page_start", pa.int64()),
            pa.field("page_end", pa.int64()),
            pa.field("content_type", pa.string(), nullable=False),
            pa.field("chunk_summary", pa.string()),
            pa.field("token_count", pa.int64(), nullable=False),
            pa.field("chunk_id", pa.string(), nullable=False),
            pa.field("chunk_index", pa.int64(), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), dimensions), nullable=False),
        ]
    )


class EmbeddingCache:
    """Compact resumable float32 embedding cache keyed by model and content."""

    def __init__(self, path: Path, *, model_key: str, dimensions: int):
        self.path = path
        self.model_key = model_key
        self.dimensions = dimensions
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
              cache_key TEXT PRIMARY KEY,
              model_key TEXT NOT NULL,
              dimensions INTEGER NOT NULL,
              vector BLOB NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def key_for(self, content: str) -> str:
        payload = f"{self.model_key}\0{self.dimensions}\0{content}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, content: str) -> array | None:
        row = self.connection.execute(
            "SELECT vector FROM embeddings WHERE cache_key=? AND model_key=? AND dimensions=?",
            (self.key_for(content), self.model_key, self.dimensions),
        ).fetchone()
        if row is None:
            return None
        vector = array("f")
        vector.frombytes(row[0])
        return vector if len(vector) == self.dimensions else None

    def put(self, content: str, vector: array) -> None:
        self.connection.execute(
            """
            INSERT INTO embeddings(cache_key,model_key,dimensions,vector,created_at)
            VALUES(?,?,?,?,?)
            ON CONFLICT(cache_key) DO UPDATE SET
              model_key=excluded.model_key,
              dimensions=excluded.dimensions,
              vector=excluded.vector,
              created_at=excluded.created_at
            """,
            (
                self.key_for(content),
                self.model_key,
                self.dimensions,
                sqlite3.Binary(vector.tobytes()),
                _utcnow(),
            ),
        )

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def _coerce_vector(value: Any, dimensions: int) -> array | None:
    if value is None:
        return None
    try:
        vector = array("f", value)
    except (TypeError, ValueError, OverflowError):
        return None
    return vector if len(vector) == dimensions else None


def _prepare_manifest_rows(
    manifest_path: Path,
    *,
    assistant_id: str | None,
    chunks_jsonl: list[str],
) -> tuple[Any, dict, list[dict], list[dict], list[dict], list[Path]]:
    manifest = _load_manifest(manifest_path)
    profile = get_profile(assistant_id or manifest.get("assistant_id") or "sea")
    source_records = [
        _source_record_from_manifest(item)
        for item in manifest.get("sources", [])
        if isinstance(item, dict)
    ]
    document_records: list[DocumentRecord] = []
    chunk_rows: list[dict] = []
    documents_by_id: dict[str, DocumentRecord] = {}
    chunk_paths = _manifest_chunk_paths(manifest_path, manifest, chunks_jsonl)
    for item in manifest.get("documents", []):
        if not isinstance(item, dict):
            continue
        document = _document_record_from_manifest(item, profile=profile)
        document_records.append(document)
        documents_by_id[document.document_id] = document
        chunk_rows.extend(
            _chunk_rows_for_document(
                item,
                document,
                profile=profile,
                allow_summary_fallback=not bool(chunk_paths),
            )
        )
    if chunk_paths:
        chunk_rows.extend(
            _chunk_rows_from_jsonl(chunk_paths, documents_by_id, profile=profile)
        )

    inferred_sources = corpus.build_source_records_for_documents(
        document_records, profile=profile
    )
    by_source = {record.source_id: record for record in inferred_sources}
    for record in source_records:
        by_source[record.source_id] = record
    source_rows = [item.model_dump() for item in by_source.values()]
    document_rows = [item.model_dump() for item in document_records]
    return profile, manifest, source_rows, document_rows, chunk_rows, chunk_paths


def _replace_sources(manifest: dict, document_rows: list[dict]) -> set[str]:
    declared = {
        str(item.get("source_id") or "").strip()
        for item in manifest.get("sources", [])
        if isinstance(item, dict)
    }
    observed = {str(row.get("source_id") or "").strip() for row in document_rows}
    source_ids = {value for value in declared | observed if value}
    if not source_ids:
        raise ValueError("Manifest does not declare a replaceable source partition.")
    unsupported = source_ids - ALLOWED_REPLACE_SOURCE_IDS
    if unsupported:
        raise ValueError(
            "This command may replace only the governed intranet-project partition; "
            f"unsupported sources: {', '.join(sorted(unsupported))}"
        )
    return source_ids


def _merge_partition_rows(
    existing_rows: list[dict],
    incoming_rows: list[dict],
    *,
    replace_source_ids: set[str],
    key_field: str,
) -> list[dict]:
    merged: dict[str, dict] = {}
    for row in existing_rows:
        if str(row.get("source_id") or "") in replace_source_ids:
            continue
        key = str(row.get(key_field) or "")
        if key:
            merged[key] = row
    for row in incoming_rows:
        key = str(row.get(key_field) or "")
        if not key:
            raise ValueError(f"Incoming row has no {key_field}.")
        if key in merged:
            raise ValueError(
                f"Incoming {key_field} collides with a retained source partition: {key}"
            )
        merged[key] = row
    return list(merged.values())


async def _source_counts(table, source_ids: set[str]) -> dict[str, int]:
    if table is None:
        return {source_id: 0 for source_id in source_ids}
    return {
        source_id: int(await table.count_rows(f"source_id = '{_sql_quote(source_id)}'"))
        for source_id in sorted(source_ids)
    }


async def _add_embedded_rows(
    table,
    rows: list[dict],
    *,
    schema: pa.Schema,
    embedder,
    cache: EmbeddingCache,
    dimensions: int,
    embed_batch_size: int,
    progress: dict[str, int],
) -> None:
    for start in range(0, len(rows), embed_batch_size):
        batch = rows[start : start + embed_batch_size]
        missing: list[tuple[dict, str]] = []
        for row in batch:
            row["status"] = str(row.get("status") or "approved")
            vector = _coerce_vector(row.get("vector"), dimensions)
            content = str(row.get("content") or "").strip()
            if not content:
                raise ValueError(f"Chunk {row.get('chunk_id')} has no content.")
            if vector is None:
                vector = cache.get(content)
                if vector is not None:
                    progress["cache_hits"] += 1
            if vector is None:
                missing.append((row, content))
            else:
                row["vector"] = vector
        if missing:
            vectors = embedder.embed_documents([content for _, content in missing])
            if len(vectors) != len(missing):
                raise RuntimeError(
                    "Embedding provider returned an unexpected vector count."
                )
            for (row, content), raw_vector in zip(missing, vectors, strict=True):
                vector = _coerce_vector(raw_vector, dimensions)
                if vector is None:
                    raise RuntimeError(
                        "Embedding provider returned an incompatible vector dimension."
                    )
                row["vector"] = vector
                cache.put(content, vector)
                progress["embedded"] += 1
            cache.commit()
        arrow = pa.Table.from_pylist(batch, schema=schema)
        await table.add(arrow)
        progress["written"] += len(batch)
        if progress["written"] - progress.get("last_reported", 0) >= 1_000:
            print(
                "[partition-deploy] chunks "
                f"written={progress['written']} embedded={progress['embedded']} "
                f"cache_hits={progress['cache_hits']}",
                flush=True,
            )
            progress["last_reported"] = progress["written"]


async def _swap_tables(
    connection, staged: dict[str, str], live: dict[str, str], run_id: str
) -> list[str]:
    """Swap validated tables and restore every renamed live table on failure.

    Backup cleanup is deliberately best-effort after a successful swap. A failed
    cleanup must not trigger a rollback after another backup has already been
    deleted, because that could leave the three-table corpus inconsistent.
    """
    backups: dict[str, str] = {}
    swapped: list[str] = []
    table_names = set(await connection.table_names())
    try:
        for logical_name in ("chunks", "documents", "sources"):
            live_name = live[logical_name]
            staged_name = staged[logical_name]
            backup_name = f"{live_name}__backup_{run_id}"
            if live_name in table_names:
                await connection.rename_table(live_name, backup_name)
                backups[logical_name] = backup_name
            await connection.rename_table(staged_name, live_name)
            swapped.append(logical_name)
    except Exception as swap_error:
        rollback_errors: list[str] = []
        for logical_name in reversed(("chunks", "documents", "sources")):
            live_name = live[logical_name]
            failed_name = f"{live_name}__failed_{run_id}"
            if logical_name in swapped:
                try:
                    await connection.rename_table(live_name, failed_name)
                except Exception as error:  # pragma: no cover - provider failure
                    rollback_errors.append(f"move {live_name}: {error}")
            if logical_name in backups:
                try:
                    await connection.rename_table(backups[logical_name], live_name)
                except Exception as error:  # pragma: no cover - provider failure
                    rollback_errors.append(f"restore {live_name}: {error}")
        if rollback_errors:
            raise RuntimeError(
                "Corpus table swap failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from swap_error
        raise

    cleanup_warnings: list[str] = []
    for backup_name in backups.values():
        try:
            await connection.drop_table(backup_name, ignore_missing=True)
        except Exception as error:  # pragma: no cover - provider failure
            cleanup_warnings.append(f"Could not remove backup {backup_name}: {error}")
    return cleanup_warnings


def _governance_summary(document_rows: list[dict]) -> dict[str, Any]:
    classifications = Counter(
        str(row.get("data_classification") or "Unspecified") for row in document_rows
    )
    flag_counts: Counter[str] = Counter()
    flagged_documents = 0
    for row in document_rows:
        flags = {
            str(value).strip()
            for value in row.get("sensitive_content_flags") or []
            if str(value).strip()
        }
        if flags:
            flagged_documents += 1
            flag_counts.update(flags)
    non_public_documents = sum(
        count for label, count in classifications.items() if label != "Public"
    )
    return {
        "data_classifications": dict(sorted(classifications.items())),
        "non_public_documents": non_public_documents,
        "sensitive_flagged_documents": flagged_documents,
        "sensitive_content_flags": dict(sorted(flag_counts.items())),
        "external_embedding_approval_required": bool(
            non_public_documents or flagged_documents
        ),
    }


def _require_external_embedding_approval(
    governance: dict[str, Any], *, approved: bool
) -> None:
    if governance["external_embedding_approval_required"] and not approved:
        raise ValueError(
            "This manifest contains non-public or sensitive-flagged documents. "
            "Embedding can transmit chunk text to the configured external provider; "
            "rerun only after authorization with "
            "--allow-non-public-external-embedding."
        )


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).expanduser().resolve()
    profile, manifest, source_rows, document_rows, chunk_rows, chunk_paths = (
        _prepare_manifest_rows(
            manifest_path,
            assistant_id=args.assistant_id,
            chunks_jsonl=args.chunks_jsonl,
        )
    )
    replace_source_ids = _replace_sources(manifest, document_rows)
    incoming_document_counts = Counter(str(row["source_id"]) for row in document_rows)
    incoming_chunk_counts = Counter(str(row["source_id"]) for row in chunk_rows)
    governance = _governance_summary(document_rows)
    if set(incoming_chunk_counts) - replace_source_ids:
        raise ValueError(
            "Incoming chunks include a source outside the replacement partition."
        )
    incoming_chunk_ids = [str(row.get("chunk_id") or "") for row in chunk_rows]
    if not all(incoming_chunk_ids) or len(set(incoming_chunk_ids)) != len(
        incoming_chunk_ids
    ):
        raise ValueError("Incoming chunks must have unique, non-empty chunk_id values.")

    connection = await database.get_connection(profile=profile)
    run_id = uuid4().hex[:12]
    live = {
        logical_name: profile.table_names[logical_name]
        for logical_name in ("sources", "documents", "chunks")
    }
    staged = {
        name: f"{table_name}__staging_{run_id}" for name, table_name in live.items()
    }
    cache: EmbeddingCache | None = None
    try:
        existing_sources_table = await connection.open_table(live["sources"])
        existing_documents_table = await connection.open_table(live["documents"])
        existing_chunks_table = await connection.open_table(live["chunks"])
        existing_source_rows = await existing_sources_table.query().to_list()
        existing_document_rows = await existing_documents_table.query().to_list()
        known_sources = {
            str(row.get("source_id") or "")
            for row in existing_source_rows + source_rows
        } - {""}
        before_documents = await _source_counts(existing_documents_table, known_sources)
        before_chunks = await _source_counts(existing_chunks_table, known_sources)
        merged_sources = _merge_partition_rows(
            existing_source_rows,
            source_rows,
            replace_source_ids=replace_source_ids,
            key_field="source_id",
        )
        merged_documents = _merge_partition_rows(
            existing_document_rows,
            document_rows,
            replace_source_ids=replace_source_ids,
            key_field="document_id",
        )

        summary: dict[str, Any] = {
            "run_id": run_id,
            "generated_at": _utcnow(),
            "assistant_id": profile.assistant_id,
            "apply": bool(args.apply),
            "replace_source_ids": sorted(replace_source_ids),
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "chunk_files": [
                {"path": str(path), "sha256": _sha256(path)} for path in chunk_paths
            ],
            "incoming": {
                "sources": len(source_rows),
                "documents": len(document_rows),
                "chunks": len(chunk_rows),
                "documents_by_source": dict(sorted(incoming_document_counts.items())),
                "chunks_by_source": dict(sorted(incoming_chunk_counts.items())),
            },
            "before": {
                "documents_by_source": before_documents,
                "chunks_by_source": before_chunks,
            },
            "governance": governance,
        }
        if not args.apply:
            return summary

        _require_external_embedding_approval(
            governance,
            approved=args.allow_non_public_external_embedding,
        )

        embed_model = str(os.environ.get("AZURE_OPENAI_EMBED_MODEL") or "unknown")
        model_key = f"{embed_model}:dimensions={args.vector_dimensions}"
        cache_path = (
            Path(args.embedding_cache).expanduser().resolve()
            if args.embedding_cache
            else manifest_path.parent / "embeddings" / "embedding_cache.sqlite3"
        )
        cache = EmbeddingCache(
            cache_path,
            model_key=model_key,
            dimensions=args.vector_dimensions,
        )
        embedder = genai.get_embedding_client()
        chunk_schema = _chunk_schema(args.vector_dimensions)

        await connection.create_table(
            staged["sources"],
            data=pa.Table.from_pylist(
                merged_sources, schema=SourceRecord.to_arrow_schema()
            ),
            mode="create",
        )
        await connection.create_table(
            staged["documents"],
            data=pa.Table.from_pylist(
                merged_documents, schema=DocumentRecord.to_arrow_schema()
            ),
            mode="create",
        )
        await connection.create_table(
            staged["chunks"], schema=chunk_schema, mode="create"
        )
        staged_chunks_table = await connection.open_table(staged["chunks"])
        progress = {"written": 0, "embedded": 0, "cache_hits": 0, "last_reported": 0}
        incoming_ids = set(incoming_chunk_ids)
        retained_chunk_count = 0
        reader = await existing_chunks_table.query().to_batches(
            max_batch_length=args.read_batch_size
        )
        async for record_batch in reader:
            retained = []
            for row in record_batch.to_pylist():
                if str(row.get("source_id") or "") in replace_source_ids:
                    continue
                chunk_id = str(row.get("chunk_id") or "")
                if chunk_id in incoming_ids:
                    raise ValueError(
                        f"Incoming chunk_id collides with a retained source partition: {chunk_id}"
                    )
                retained.append(row)
            if retained:
                await _add_embedded_rows(
                    staged_chunks_table,
                    retained,
                    schema=chunk_schema,
                    embedder=embedder,
                    cache=cache,
                    dimensions=args.vector_dimensions,
                    embed_batch_size=args.embed_batch_size,
                    progress=progress,
                )
                retained_chunk_count += len(retained)
        await _add_embedded_rows(
            staged_chunks_table,
            chunk_rows,
            schema=chunk_schema,
            embedder=embedder,
            cache=cache,
            dimensions=args.vector_dimensions,
            embed_batch_size=args.embed_batch_size,
            progress=progress,
        )

        staged_documents_table = await connection.open_table(staged["documents"])
        staged_sources_table = await connection.open_table(staged["sources"])
        after_documents = await _source_counts(staged_documents_table, known_sources)
        after_chunks = await _source_counts(staged_chunks_table, known_sources)
        for source_id in replace_source_ids:
            if after_documents[source_id] != incoming_document_counts[source_id]:
                raise RuntimeError(f"Staged document count mismatch for {source_id}.")
            if after_chunks[source_id] != incoming_chunk_counts[source_id]:
                raise RuntimeError(f"Staged chunk count mismatch for {source_id}.")
        for source_id in known_sources - replace_source_ids:
            if after_documents[source_id] != before_documents[source_id]:
                raise RuntimeError(f"Retained document count changed for {source_id}.")
            if after_chunks[source_id] != before_chunks[source_id]:
                raise RuntimeError(f"Retained chunk count changed for {source_id}.")
        staged_schema = await staged_chunks_table.schema()
        vector_field = staged_schema.field("vector")
        if vector_field.type != pa.list_(pa.float32(), args.vector_dimensions):
            raise RuntimeError("Staged chunks table has an incompatible vector schema.")

        cleanup_warnings = await _swap_tables(connection, staged, live, run_id)
        summary.update(
            {
                "embedding": {
                    "model_key": model_key,
                    "vector_dimensions": args.vector_dimensions,
                    "cache": str(cache_path),
                    "written": progress["written"],
                    "embedded": progress["embedded"],
                    "cache_hits": progress["cache_hits"],
                },
                "retained_chunks": retained_chunk_count,
                "after": {
                    "documents_by_source": after_documents,
                    "chunks_by_source": after_chunks,
                },
                "tables": live,
                "swapped": True,
                "cleanup_warnings": cleanup_warnings,
            }
        )
        audit_path = (
            Path(args.audit_output).expanduser().resolve()
            if args.audit_output
            else manifest_path.parent / "deployment_audit.json"
        )
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary["audit"] = str(audit_path)
        return summary
    finally:
        if cache is not None:
            cache.close()
        close = getattr(connection, "close", None)
        if callable(close):
            result = close()
            if asyncio.iscoroutine(result):
                await result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", required=True, help="Assistant-kit corpus manifest."
    )
    parser.add_argument("--assistant-id", default=None)
    parser.add_argument("--chunks-jsonl", action="append", default=[])
    parser.add_argument("--embedding-cache", default="")
    parser.add_argument("--audit-output", default="")
    parser.add_argument(
        "--vector-dimensions", type=int, default=DEFAULT_VECTOR_DIMENSIONS
    )
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--read-batch-size", type=int, default=512)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Build, validate, and swap staging tables. Without this flag the command is read-only.",
    )
    parser.add_argument(
        "--allow-non-public-external-embedding",
        action="store_true",
        help=(
            "Confirm authorization to send non-public or sensitive-flagged chunk "
            "text to the configured external embedding provider."
        ),
    )
    args = parser.parse_args()
    if (
        args.vector_dimensions <= 0
        or args.embed_batch_size <= 0
        or args.read_batch_size <= 0
    ):
        parser.error("Vector dimensions and batch sizes must be positive.")
    print(json.dumps(asyncio.run(_run(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
