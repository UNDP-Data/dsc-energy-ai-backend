"""Tests for the public SGP AI GitHub Pages proxy."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

import main as app_module
from src import database as database_module

ALLOWED_ORIGIN = "https://undp-data.github.io"


class FakeTable:
    async def count_rows(self, source_filter=None):
        counts = {
            None: 3080,
            "source_id = 'gef_sgp_innovation_library'": 1433,
            "source_id = 'gef_sgp_intranet_projects'": 1647,
            "source_id IN ('gef_sgp_innovation_library')": 1433,
            "source_id IN ('gef_sgp_intranet_projects')": 1647,
            "source_id IN ('gef_sgp_innovation_library', 'gef_sgp_intranet_projects')": 3080,
        }
        return counts[source_filter]


class FakeClient:
    def table_name(self, logical_name: str) -> str:
        return f"sgp_ai_{logical_name}"

    async def open_optional_table(self, _name: str):
        return FakeTable()

    async def retrieve_chunks(
        self, query: str, *, limit: int, debug=None, source_ids=None
    ):
        assert query == "coastal erosion"
        assert limit == 2
        assert source_ids == ("gef_sgp_innovation_library",)
        return [FakeDump({"content": "Evidence"})], [FakeDump({"title": "Doc"})]

    async def score_document_relevance_map(self, query: str, *, source_ids=None):
        assert query == "coastal erosion"
        self.source_ids = source_ids
        return [
            {
                "document_id": "doc-1",
                "title": "Coastal erosion grant lessons",
                "source": "gef_sgp_innovation_library",
                "year": 2024,
                "url": "https://example.org/doc-1",
                "document_type": "project profile",
                "topics": ["coastal erosion"],
                "geographies": ["Turkey"],
                "country_codes": ["Turkey"],
                "region_codes": [],
                "raw_score": 3.25,
                "relevance": 1.0,
                "score_explanation": {"signal_strength": 0.9},
            }
        ]


class FakeDump:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self):
        return self.payload


def test_pages_status_requires_allowed_origin(monkeypatch):
    @asynccontextmanager
    async def fake_profile_client(_profile):
        yield FakeClient()

    monkeypatch.setattr(app_module, "_profile_client", fake_profile_client)

    with TestClient(app_module.app) as client:
        response = client.get(
            "/pages/sgp-ai/status",
            headers={"Origin": ALLOWED_ORIGIN},
        )
        blocked = client.get(
            "/pages/sgp-ai/status",
            headers={"Origin": "https://example.org"},
        )

    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == ALLOWED_ORIGIN
    assert response.json()["corpus_ready"] is True
    assert response.json()["document_count"] == 1433
    assert response.json()["data_source"] == "innovation_library"
    assert blocked.status_code == 403


def test_pages_status_rejects_unapproved_data_sources(monkeypatch):
    @asynccontextmanager
    async def fake_profile_client(_profile):
        yield FakeClient()

    monkeypatch.setattr(app_module, "_profile_client", fake_profile_client)

    with TestClient(app_module.app) as client:
        projects = client.get(
            "/pages/sgp-ai/status?data_source=project_database",
            headers={"Origin": ALLOWED_ORIGIN},
        )
        all_sources = client.get(
            "/pages/sgp-ai/status?data_source=all",
            headers={"Origin": ALLOWED_ORIGIN},
        )

    assert projects.status_code == 422
    assert all_sources.status_code == 422


def test_internal_status_supports_source_partitions_with_separate_key(monkeypatch):
    @asynccontextmanager
    async def fake_profile_client(_profile):
        yield FakeClient()

    monkeypatch.setattr(app_module, "_profile_client", fake_profile_client)
    monkeypatch.setenv("SGP_PROJECT_RAG_API_KEY", "project-secret")

    with TestClient(app_module.app) as client:
        projects = client.get(
            "/internal/sgp-ai/status?data_source=projects",
            headers={"X-Api-Key": "project-secret"},
        )
        all_sources = client.get(
            "/internal/sgp-ai/status?data_source=all",
            headers={"X-Api-Key": "project-secret"},
        )
        public_key = client.get(
            "/internal/sgp-ai/status?data_source=projects",
            headers={"X-Api-Key": "not-the-project-key"},
        )

    assert projects.status_code == 200
    assert projects.json()["document_count"] == 1647
    assert projects.json()["source_ids"] == ["gef_sgp_intranet_projects"]
    assert all_sources.status_code == 200
    assert all_sources.json()["document_count"] == 3080
    assert public_key.status_code == 401


def test_internal_relevance_map_scopes_projects(monkeypatch):
    fake_client = FakeClient()

    @asynccontextmanager
    async def fake_profile_client(_profile):
        yield fake_client

    monkeypatch.setattr(app_module, "_profile_client", fake_profile_client)
    monkeypatch.setenv("SGP_PROJECT_RAG_API_KEY", "project-secret")

    with TestClient(app_module.app) as client:
        response = client.get(
            "/internal/sgp-ai/relevance-map?query=coastal%20erosion&data_source=project_database",
            headers={"X-Api-Key": "project-secret"},
        )

    assert response.status_code == 200
    assert response.json()["source_ids"] == ["gef_sgp_intranet_projects"]
    assert fake_client.source_ids == ("gef_sgp_intranet_projects",)


def test_pages_status_rejects_missing_origin(monkeypatch):
    @asynccontextmanager
    async def fake_profile_client(_profile):
        yield FakeClient()

    monkeypatch.setattr(app_module, "_profile_client", fake_profile_client)

    with TestClient(app_module.app) as client:
        response = client.get("/pages/sgp-ai/status")

    assert response.status_code == 403


def test_pages_retrieve_returns_documents_and_chunks(monkeypatch):
    @asynccontextmanager
    async def fake_profile_client(_profile):
        yield FakeClient()

    monkeypatch.setattr(app_module, "_profile_client", fake_profile_client)

    with TestClient(app_module.app) as client:
        response = client.get(
            "/pages/sgp-ai/debug/retrieve?query=coastal%20erosion&limit=2",
            headers={"Origin": ALLOWED_ORIGIN},
        )

    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == ALLOWED_ORIGIN
    assert response.json()["documents"] == [{"title": "Doc"}]
    assert response.json()["chunks"] == [{"content": "Evidence"}]


def test_pages_relevance_map_returns_cors_and_scores(monkeypatch):
    @asynccontextmanager
    async def fake_profile_client(_profile):
        yield FakeClient()

    monkeypatch.setattr(app_module, "_profile_client", fake_profile_client)

    with TestClient(app_module.app) as client:
        response = client.get(
            "/pages/sgp-ai/relevance-map?query=coastal%20erosion&data_source=innovation_library",
            headers={"Origin": ALLOWED_ORIGIN},
        )

    payload = response.json()
    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == ALLOWED_ORIGIN
    assert payload["data_source"] == "innovation_library"
    assert payload["document_count"] == 1
    assert payload["documents"][0]["relevance"] == 1.0
    assert payload["documents"][0]["source"] == "gef_sgp_innovation_library"


def test_pages_relevance_map_honors_empty_corpus(monkeypatch):
    class EmptyMapClient(FakeClient):
        async def score_document_relevance_map(self, query: str, *, source_ids=None):
            return []

    @asynccontextmanager
    async def fake_profile_client(_profile):
        yield EmptyMapClient()

    monkeypatch.setattr(app_module, "_profile_client", fake_profile_client)

    with TestClient(app_module.app) as client:
        response = client.get(
            "/pages/sgp-ai/relevance-map?query=coastal%20erosion",
            headers={"Origin": ALLOWED_ORIGIN},
        )

    assert response.status_code == 200
    assert response.json()["document_count"] == 0
    assert response.json()["documents"] == []


def test_relevance_feature_breakdown_matches_row_breakdown():
    row = {
        "document_id": "doc-1",
        "source_id": "gef_sgp_intranet_projects",
        "canonical_title": "Turkey coastal erosion coral reef project",
        "subtitle": "Community monitoring",
        "summary": "Women and local programme teams worked on coastal erosion and biodiversity.",
        "series_name": "SGP project database",
        "publisher": "GEF Small Grants Programme",
        "document_type": "project profile",
        "year": 2024,
        "status": "approved",
        "url": "https://example.org/project",
        "topic_tags": ["coastal erosion", "biodiversity", "women"],
        "audience_tags": ["programme teams", "women"],
        "country_codes": ["TUR"],
        "region_codes": ["europe_cis"],
        "source_priority": 1.0,
        "quality_score": 0.8,
        "authority_tier": "trusted",
        "is_flagship": False,
        "is_data_report": False,
    }
    profile = database_module._build_retrieval_profile("Turkey coastal erosion women")
    feature = database_module._build_document_relevance_features(row)

    row_breakdown = database_module._document_row_breakdown(row, profile)
    feature_breakdown = database_module._document_feature_breakdown(
        feature,
        profile,
        database_module._extract_focus_phrases(profile.query),
    )

    for key in (
        "title_overlap",
        "summary_overlap",
        "text_overlap",
        "topics_overlap",
        "signal_strength",
        "token_hits",
        "geography_match_class",
        "score",
    ):
        assert feature_breakdown[key] == row_breakdown[key]


def test_pages_preflight_allows_pages_origin():
    with TestClient(app_module.app) as client:
        response = client.options(
            "/pages/sgp-ai/model",
            headers={
                "Origin": ALLOWED_ORIGIN,
                "Access-Control-Request-Method": "POST",
            },
        )

    assert response.status_code == 204
    assert response.headers["Access-Control-Allow-Origin"] == ALLOWED_ORIGIN
    assert "POST" in response.headers["Access-Control-Allow-Methods"]


def test_pages_model_streams_without_browser_api_key(monkeypatch):
    async def fake_ask_assistant_model(_request, assistant_id, messages):
        assert assistant_id == "sgp_ai"
        assert messages[-1].content == "hello"
        assert _request.state.retrieval_source_ids == ("gef_sgp_innovation_library",)
        assert _request.state.ui_locale == "fr"
        return StreamingResponse(
            iter([b'{"role":"assistant","content":"Hello","graph":null}\n']),
            media_type="application/x-ndjson",
        )

    monkeypatch.setattr(app_module, "ask_assistant_model", fake_ask_assistant_model)

    with TestClient(app_module.app) as client:
        response = client.post(
            "/pages/sgp-ai/model?ui_locale=fr",
            headers={"Origin": ALLOWED_ORIGIN},
            json=[{"role": "human", "content": "hello"}],
        )
        blocked = client.post(
            "/pages/sgp-ai/model?data_source=project_database",
            headers={"Origin": ALLOWED_ORIGIN},
            json=[{"role": "human", "content": "hello"}],
        )
        unsupported_locale = client.post(
            "/pages/sgp-ai/model?ui_locale=de",
            headers={"Origin": ALLOWED_ORIGIN},
            json=[{"role": "human", "content": "hello"}],
        )

    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == ALLOWED_ORIGIN
    assert (
        response.text.strip() == '{"role":"assistant","content":"Hello","graph":null}'
    )
    assert blocked.status_code == 422
    assert unsupported_locale.status_code == 422
