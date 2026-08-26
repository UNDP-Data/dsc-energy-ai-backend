from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "deploy_corpus_partition",
    ROOT / "scripts" / "deploy_corpus_partition.py",
)
assert SPEC and SPEC.loader
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)


def test_merge_partition_rows_replaces_only_target_source():
    rows = deploy._merge_partition_rows(
        [
            {"document_id": "library-1", "source_id": "gef_sgp_innovation_library"},
            {"document_id": "old-project", "source_id": "gef_sgp_intranet_projects"},
        ],
        [{"document_id": "new-project", "source_id": "gef_sgp_intranet_projects"}],
        replace_source_ids={"gef_sgp_intranet_projects"},
        key_field="document_id",
    )
    assert {row["document_id"] for row in rows} == {"library-1", "new-project"}


def test_non_public_manifest_requires_explicit_external_embedding_approval():
    governance = deploy._governance_summary(
        [
            {
                "data_classification": "Restricted",
                "sensitive_content_flags": ["PII", "signature"],
            },
            {"data_classification": "Public", "sensitive_content_flags": []},
        ]
    )
    assert governance["non_public_documents"] == 1
    assert governance["sensitive_flagged_documents"] == 1
    with pytest.raises(ValueError, match="allow-non-public-external-embedding"):
        deploy._require_external_embedding_approval(governance, approved=False)
    deploy._require_external_embedding_approval(governance, approved=True)


@pytest.mark.asyncio
async def test_swap_restores_every_live_table_when_staged_rename_fails():
    class FakeConnection:
        def __init__(self):
            self.names = {
                "chunks",
                "documents",
                "sources",
                "chunks-stage",
                "documents-stage",
                "sources-stage",
            }
            self.failed = False

        async def table_names(self):
            return list(self.names)

        async def rename_table(self, old, new):
            if old == "documents-stage" and not self.failed:
                self.failed = True
                raise RuntimeError("injected rename failure")
            assert old in self.names
            assert new not in self.names
            self.names.remove(old)
            self.names.add(new)

        async def drop_table(self, name, *, ignore_missing):
            self.names.discard(name)

    connection = FakeConnection()
    with pytest.raises(RuntimeError, match="injected rename failure"):
        await deploy._swap_tables(
            connection,
            {
                "chunks": "chunks-stage",
                "documents": "documents-stage",
                "sources": "sources-stage",
            },
            {"chunks": "chunks", "documents": "documents", "sources": "sources"},
            "test-run",
        )
    assert {"chunks", "documents", "sources"}.issubset(connection.names)
    assert "chunks__failed_test-run" in connection.names


def test_embedding_cache_round_trips_float32_vectors(tmp_path):
    cache = deploy.EmbeddingCache(
        tmp_path / "embeddings.sqlite3",
        model_key="test-model",
        dimensions=3,
    )
    vector = deploy.array("f", [0.1, 0.2, 0.3])
    cache.put("hello", vector)
    cache.commit()
    restored = cache.get("hello")
    cache.close()
    assert restored is not None
    assert list(restored) == pytest.approx([0.1, 0.2, 0.3])


@pytest.mark.asyncio
async def test_add_embedded_rows_uses_cache_on_repeat(tmp_path):
    class FakeEmbedder:
        calls = 0

        def embed_documents(self, texts):
            self.calls += 1
            return [[float(index + 1)] * 3 for index, _ in enumerate(texts)]

    class FakeTable:
        def __init__(self):
            self.rows = []

        async def add(self, table):
            self.rows.extend(table.to_pylist())

    cache = deploy.EmbeddingCache(
        tmp_path / "embeddings.sqlite3",
        model_key="test-model",
        dimensions=3,
    )
    embedder = FakeEmbedder()
    table = FakeTable()
    row = {
        "document_id": "doc-1",
        "title": "Example",
        "year": 2026,
        "language": "en",
        "url": "https://example.org/doc-1",
        "source_id": "gef_sgp_intranet_projects",
        "status": "approved",
        "content": "embedded content",
        "content_type": "text",
        "token_count": 2,
        "chunk_id": "chunk-1",
        "chunk_index": 0,
    }
    progress = {"written": 0, "embedded": 0, "cache_hits": 0}
    schema = deploy._chunk_schema(3)
    await deploy._add_embedded_rows(
        table,
        [dict(row)],
        schema=schema,
        embedder=embedder,
        cache=cache,
        dimensions=3,
        embed_batch_size=1,
        progress=progress,
    )
    await deploy._add_embedded_rows(
        table,
        [{**row, "chunk_id": "chunk-2"}],
        schema=schema,
        embedder=embedder,
        cache=cache,
        dimensions=3,
        embed_batch_size=1,
        progress=progress,
    )
    cache.close()
    assert embedder.calls == 1
    assert progress["written"] == 2
    assert progress["embedded"] == 1
    assert progress["cache_hits"] == 1
    assert len(table.rows[0]["vector"]) == 3
