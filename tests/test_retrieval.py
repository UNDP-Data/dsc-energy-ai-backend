"""
Tests for retrieval profiling and document selection heuristics.
"""

import pytest

from src.database import (
    Client,
    _build_retrieval_profile,
    _build_summary_fallback_chunks,
    _build_metadata_patterns,
    _classify_geography_match,
    _dedupe_document_rows,
    _document_has_min_signal,
    _score_document_row,
    _select_documents_and_chunks,
    _prioritize_retrieval_queries,
    _retrieval_queries_with_variants,
    build_retrieval_queries,
    should_defer_to_publications,
)
from src import corpus


def test_retrieval_profile_prefers_recent_for_data_queries():
    profile = _build_retrieval_profile(
        "What is the latest progress on access to electricity?"
    )
    assert profile.prefer_recent is True
    assert profile.explicit_years == []


def test_dedupe_document_rows_prefers_user_facing_url_identity():
    rows = [
        {
            "document_id": "download-a",
            "url": "https://sgp.undp.org/innovation-library/item/example.html",
            "canonical_title": "Example",
            "year": 2024,
        },
        {
            "document_id": "download-b",
            "url": "https://sgp.undp.org/innovation-library/item/example.html",
            "canonical_title": "Example",
            "year": 2024,
        },
        {
            "document_id": "download-c",
            "url": "https://sgp.undp.org/innovation-library/item/other.html",
            "canonical_title": "Other",
            "year": 2024,
        },
    ]

    deduped = _dedupe_document_rows(rows)

    assert [row["document_id"] for row in deduped] == ["download-a", "download-c"]


def test_retrieval_profile_does_not_force_recent_for_explanatory_queries():
    profile = _build_retrieval_profile("Tell me more about feed-in tariffs")
    assert profile.prefer_recent is False
    assert profile.explanatory is True
    assert profile.intent == "policy"


def test_retrieval_profile_classifies_data_queries():
    profile = _build_retrieval_profile("What is the latest progress on access to electricity in 2025?")
    assert profile.intent == "data"


def test_retrieval_profile_flags_unspecified_energy_access_count_as_current_data():
    profile = _build_retrieval_profile("How many people lack access to energy?")
    assert profile.intent == "data"
    assert profile.prefer_recent is True
    assert profile.current_data_query is True


def test_build_retrieval_queries_expands_unspecified_energy_access_count_query():
    queries = build_retrieval_queries("How many people lack access to energy?")
    assert "how many people lack access to electricity" in queries
    assert "tracking sdg7 access to electricity" in queries


def test_build_retrieval_queries_expands_sgp_operational_phase_shorthand():
    queries = build_retrieval_queries(
        "Who are the OP8 operational guidelines written for?"
    )
    prioritized = _prioritize_retrieval_queries(
        "Who are the OP8 operational guidelines written for?"
    )

    assert any("operational phase 8" in query.lower() for query in queries)
    assert "operational phase 8" in prioritized[0].lower()


def test_prioritized_retrieval_queries_keep_sdg7_variant_for_current_data_query():
    queries = _prioritize_retrieval_queries("How many people lack access to energy?")
    assert "tracking sdg7 access to electricity" in queries


def test_multilingual_retrieval_keeps_translation_and_original_query():
    queries = _retrieval_queries_with_variants(
        "How have SGP grants supported coastal resilience in Senegal?",
        ["Comment les subventions du SGP ont-elles soutenu la résilience côtière au Sénégal ?"],
    )

    assert queries[0].startswith("How have SGP grants")
    assert any(query.startswith("Comment les subventions") for query in queries)
    assert len(queries) == len(set(queries))


def test_should_defer_to_publications_for_current_energy_access_data_query():
    assert should_defer_to_publications("How many people lack access to energy?") is True
    assert should_defer_to_publications("Tell me more about feed-in tariffs") is False


def test_retrieval_profile_extracts_lac_scope():
    profile = _build_retrieval_profile(
        "What is the latest progress on access to electricity in Latin America and the Caribbean?"
    )
    assert profile.region_scopes == frozenset({"latin america"})
    assert profile.country_scopes == frozenset()
    assert profile.has_geographic_scope is True


def test_geography_matching_normalizes_imported_sgp_region_codes():
    profile = _build_retrieval_profile(
        "What are the lessons from integrated water management in the Caribbean?"
    )
    row = {"region_codes": ["latin_america_caribbean"]}

    match = _classify_geography_match(row, profile)

    assert match["match_class"] == "exact_region"


def test_retrieval_profile_extracts_country_scope_and_parent_region():
    profile = _build_retrieval_profile("Tell me more about access to electricity in Nigeria")
    assert profile.country_scopes == frozenset({"NGA"})
    assert profile.fallback_region_scopes == frozenset({"africa"})
    assert profile.has_geographic_scope is True


def test_retrieval_profile_has_no_geography_for_generic_query():
    profile = _build_retrieval_profile("Tell me more about access to electricity")
    assert profile.country_scopes == frozenset()
    assert profile.region_scopes == frozenset()
    assert profile.has_geographic_scope is False


def test_document_selection_prefers_topical_document_over_generic_recent_report():
    profile = _build_retrieval_profile("Tell me more about feed-in tariff")
    rows = [
        {
            "title": "2025 Tracking SDG7 Report",
            "year": 2025,
            "language": "en",
            "url": "https://trackingsdg7.esmap.org/downloads",
            "summary": "Annual tracking report on SDG 7 progress and indicators.",
            "content": "Electricity access, renewable share, energy efficiency trends.",
            "_distance": 0.15,
        },
        {
            "title": "Renewable Energy Policy Toolkit",
            "year": 2022,
            "language": "en",
            "url": "https://example.org/policy-toolkit",
            "summary": "Policy tools for renewable energy deployment.",
            "content": (
                "Feed-in tariff policy design, tariff setting, degression, "
                "power purchase arrangements, and implementation challenges."
            ),
            "_distance": 0.22,
        },
    ]

    chunks, documents = _select_documents_and_chunks(rows, profile, limit=4)

    assert chunks
    assert documents
    assert documents[0].title == "Renewable Energy Policy Toolkit"
    assert chunks[0].title == "Renewable Energy Policy Toolkit"


def test_document_scoring_prefers_latest_sgp_annual_monitoring_report():
    profile = _build_retrieval_profile(
        "What reporting period is covered by the latest SGP annual monitoring report?"
    )
    latest = {
        "canonical_title": "GEF SMALL GRANTS PROGRAMME ANNUAL MONITORING REPORT 2024 - 2025 (SUMMARY INFOGRAPHIC)",
        "year": 2025,
        "summary": "This is the infographic version of the GEF Small Grants Programme Results Report 2024-2025.",
        "topic_tags_text": "AMR | biodiversity | climate change mitigation",
        "audience_tags_text": "programme teams | indigenous peoples | women | youth | persons with disabilities",
        "document_type": "report",
        "status": "approved",
    }
    older = {
        "canonical_title": "GEF SMALL GRANTS PROGRAMME ANNUAL MONITORING REPORT 2023 - 2024 (Full Version)",
        "year": 2024,
        "summary": "The GEF Small Grants Programme annual monitoring report presents results for 2023-2024.",
        "topic_tags_text": "AMR | biodiversity | climate change mitigation",
        "audience_tags_text": "programme teams | indigenous peoples | women | youth | persons with disabilities",
        "document_type": "report",
        "status": "approved",
    }

    assert _score_document_row(latest, profile) > _score_document_row(older, profile)


def test_document_signal_uses_sgp_priority_group_audience_tags():
    profile = _build_retrieval_profile("How are youth engaged in SGP projects?")
    row = {
        "canonical_title": "Evaluation of project Fotae-Na - SGP Ecuador",
        "year": 2018,
        "summary": "",
        "topic_tags_text": "report/publication | case study",
        "audience_tags_text": "programme teams | indigenous peoples | women | youth",
        "document_type": "publication",
        "status": "approved",
    }

    assert _document_has_min_signal(row, profile) is True


def test_document_signal_accepts_strong_sgp_title_match_without_exact_focus_phrase():
    profile = _build_retrieval_profile(
        "Who are the OP8 operational guidelines written for, and what implementation issues do they cover?"
    )
    row = {
        "canonical_title": "GEF Small Grants Programme UNDP Operational Guidelines for Operational Phase 8",
        "year": 2025,
        "summary": "Operational Phase 8 implementation guidance for SGP country programme teams.",
        "topic_tags_text": "report/publication | global publication",
        "audience_tags_text": "programme teams",
        "document_type": "policy",
        "status": "approved",
    }

    expanded_profile = _build_retrieval_profile(_prioritize_retrieval_queries(profile.query)[0])

    assert _document_has_min_signal(row, expanded_profile) is True


def test_document_selection_penalizes_glossary_for_concept_query():
    profile = _build_retrieval_profile(
        "What is the connection between sustainable energy and climate change mitigation"
    )
    rows = [
        {
            "title": "The Climate Dictionary",
            "year": 2023,
            "language": "en",
            "url": "https://www.undp.org/publications/climate-dictionary",
            "summary": "A glossary of climate change terms for the general public.",
            "content": "Definitions of climate adaptation, mitigation, resilience, and greenhouse gases.",
            "_distance": 0.08,
        },
        {
            "title": "Sustainable Energy and Climate Mitigation Pathways",
            "year": 2022,
            "language": "en",
            "url": "https://example.org/mitigation-pathways",
            "summary": "How sustainable energy transitions reduce greenhouse gas emissions and support climate mitigation.",
            "content": "Sustainable energy, renewable deployment, and climate change mitigation pathways.",
            "_distance": 0.18,
        },
    ]

    _chunks, documents = _select_documents_and_chunks(rows, profile, limit=4)

    assert documents
    assert documents[0].title == "Sustainable Energy and Climate Mitigation Pathways"


def test_build_retrieval_queries_strips_explanatory_prefix():
    queries = build_retrieval_queries("Tell me more about feed in tariff")
    assert "Tell me more about feed in tariff" in queries
    assert "feed in tariff" in queries
    assert "feed-in tariff" in queries


def test_build_retrieval_queries_compacts_relationship_query():
    queries = build_retrieval_queries(
        "What is the connection between sustainable energy and climate change mitigation"
    )
    assert (
        "sustainable energy and climate change mitigation" in queries
        or "sustainable energy climate change mitigation" in queries
    )


def test_build_metadata_patterns_prefers_multi_word_phrases():
    patterns = _build_metadata_patterns("Tell me more about access to electricity")
    assert any("access[-\\s]+to[-\\s]+electricity" in pattern for pattern in patterns)

    patterns = _build_metadata_patterns("Tell me more about feed in tariff")
    assert any("feed[-\\s]+in[-\\s]+tariffs?" in pattern for pattern in patterns)
    assert not any("fits?" in pattern for pattern in patterns)


def test_document_selection_keeps_title_only_documents():
    profile = _build_retrieval_profile("Tell me more about grid infrastructure")
    rows = [
        {
            "title": "Grid Modernization Handbook",
            "year": 2024,
            "language": "en",
            "url": "",
            "summary": "Grid planning and modernization guidance.",
            "content": "Grid infrastructure planning, resilience, and integration of renewables.",
            "_distance": 0.12,
        }
    ]

    chunks, documents = _select_documents_and_chunks(rows, profile, limit=4)

    assert chunks
    assert documents
    assert documents[0].title == "Grid Modernization Handbook"


def test_document_selection_deduplicates_same_url_reports():
    profile = _build_retrieval_profile("Tell me more about access to electricity")
    rows = [
        {
            "title": "2013 Progress toward Sustainable Energy",
            "year": 2013,
            "language": "en",
            "url": "https://trackingsdg7.esmap.org/downloads",
            "summary": "Electricity access report.",
            "content": "Access to electricity trends in 2013.",
            "_distance": 0.12,
        },
        {
            "title": "2017 Progress toward Sustainable Energy",
            "year": 2017,
            "language": "en",
            "url": "https://trackingsdg7.esmap.org/downloads",
            "summary": "Electricity access report.",
            "content": "Access to electricity trends in 2017.",
            "_distance": 0.11,
        },
        {
            "title": "Energy Access Investment Case",
            "year": 2021,
            "language": "en",
            "url": "https://example.org/access-case",
            "summary": "Improving access to electricity in underserved regions.",
            "content": "Electricity access, rural electrification, and mini-grid investment.",
            "_distance": 0.18,
        },
    ]

    _chunks, documents = _select_documents_and_chunks(rows, profile, limit=4)

    assert len([doc for doc in documents if doc.url == "https://trackingsdg7.esmap.org/downloads"]) == 1


def test_document_selection_prefers_flagship_data_report_for_data_queries():
    profile = _build_retrieval_profile("What is the latest data on access to electricity?")
    rows = [
        {
            "title": "2025 Tracking SDG7 Report",
            "year": 2025,
            "language": "en",
            "url": "https://trackingsdg7.esmap.org/downloads",
            "summary": "Latest global electricity access progress and indicators.",
            "content": "Access to electricity, rural deficit, regional trends, and SDG7 indicators.",
            "_distance": 0.14,
        },
        {
            "title": "Energy Access Investment Case",
            "year": 2024,
            "language": "en",
            "url": "https://example.org/access-case",
            "summary": "Investment opportunities for improving access to electricity.",
            "content": "Electricity access finance, off-grid deployment, and mini-grid growth.",
            "_distance": 0.13,
        },
    ]

    _chunks, documents = _select_documents_and_chunks(rows, profile, limit=4)

    assert documents
    assert documents[0].title == "2025 Tracking SDG7 Report"


def test_document_selection_prefers_latest_sdg7_report_for_unspecified_energy_access_count():
    profile = _build_retrieval_profile("How many people lack access to energy?")
    rows = [
        {
            "title": "2025 Tracking SDG7 Report",
            "year": 2025,
            "language": "en",
            "url": "https://trackingsdg7.esmap.org/downloads",
            "summary": "Latest global electricity access progress and indicators.",
            "content": "In 2023, 666 million people remained without access to electricity worldwide.",
            "_distance": 0.16,
        },
        {
            "title": "2024 Energy Access Update",
            "year": 2024,
            "language": "en",
            "url": "https://example.org/older-access-update",
            "summary": "Older energy access estimates and trends.",
            "content": "About 770 million people lacked access to electricity worldwide.",
            "_distance": 0.12,
        },
    ]

    _chunks, documents = _select_documents_and_chunks(rows, profile, limit=4)

    assert documents
    assert documents[0].title == "2025 Tracking SDG7 Report"


def test_document_selection_returns_sdg7_report_as_top_resource_for_energy_access_count_query():
    profile = _build_retrieval_profile("How many people lack access to energy?")
    rows = [
        {
            "title": "2023 Regional Energy Access Brief",
            "year": 2023,
            "language": "en",
            "url": "https://example.org/regional-access-brief",
            "summary": "Regional energy access trends from an older brief.",
            "content": "About 770 million people lacked access to electricity worldwide.",
            "_distance": 0.09,
        },
        {
            "title": "2025 Tracking SDG7 Report",
            "year": 2025,
            "language": "en",
            "url": "https://trackingsdg7.esmap.org/downloads",
            "summary": "Latest global electricity access progress and indicators.",
            "content": "In 2023, 666 million people remained without access to electricity worldwide.",
            "_distance": 0.15,
        },
        {
            "title": "Energy Access Investment Case",
            "year": 2024,
            "language": "en",
            "url": "https://example.org/access-investment",
            "summary": "Investment opportunities for improving access to electricity.",
            "content": "Electricity access finance and off-grid deployment pathways.",
            "_distance": 0.08,
        },
    ]

    chunks, documents = _select_documents_and_chunks(rows, profile, limit=4)

    assert documents
    assert chunks
    assert documents[0].title == "2025 Tracking SDG7 Report"
    assert chunks[0].title == "2025 Tracking SDG7 Report"


def test_document_selection_strongly_prefers_2025_sdg7_report_for_recent_data_queries():
    profile = _build_retrieval_profile("What is the latest data on access to electricity?")
    rows = [
        {
            "title": "2024 Global Energy Access Update",
            "year": 2024,
            "language": "en",
            "url": "https://example.org/2024-global-access-update",
            "summary": "Recent electricity access progress and global estimates.",
            "content": "Electricity access trends, regional deficits, and global progress indicators.",
            "_distance": 0.06,
        },
        {
            "title": "2025 Tracking SDG7 Report",
            "year": 2025,
            "language": "en",
            "url": "https://trackingsdg7.esmap.org/downloads",
            "summary": "Latest global electricity access progress and indicators.",
            "content": "Access to electricity, rural deficit, regional trends, and SDG7 indicators.",
            "_distance": 0.17,
        },
    ]

    chunks, documents = _select_documents_and_chunks(rows, profile, limit=4)

    assert documents
    assert chunks
    assert documents[0].title == "2025 Tracking SDG7 Report"
    assert chunks[0].title == "2025 Tracking SDG7 Report"


def test_document_selection_top_three_titles_stay_in_scope_for_energy_access_count():
    profile = _build_retrieval_profile("How many people lack access to energy?")
    rows = [
        {
            "title": "Forests, Energy and Livelihoods",
            "year": 2023,
            "language": "en",
            "url": "https://example.org/forests-energy-livelihoods",
            "summary": "Woodfuel use, forests, biomass reliance, and household livelihoods.",
            "content": (
                "More than 2.4 billion people rely on polluting cooking systems, "
                "and woodfuels remain a major household energy source."
            ),
            "_distance": 0.08,
        },
        {
            "title": "2025 Tracking SDG7 Report",
            "year": 2025,
            "language": "en",
            "url": "https://trackingsdg7.esmap.org/downloads",
            "summary": "Latest global electricity access progress and indicators.",
            "content": "In 2023, 666 million people remained without access to electricity worldwide.",
            "_distance": 0.18,
        },
        {
            "title": "Rural Electrification Progress Brief",
            "year": 2024,
            "language": "en",
            "url": "https://example.org/rural-electrification-brief",
            "summary": "Electricity access and rural electrification progress.",
            "content": "Recent electricity access expansion through grids and mini-grids.",
            "_distance": 0.12,
        },
        {
            "title": "Energy Access Investment Case",
            "year": 2024,
            "language": "en",
            "url": "https://example.org/energy-access-investment-case",
            "summary": "Investment opportunities for improving access to electricity.",
            "content": "Electricity access finance and off-grid deployment pathways.",
            "_distance": 0.1,
        },
        {
            "title": "Climate Dictionary",
            "year": 2023,
            "language": "en",
            "url": "https://example.org/climate-dictionary",
            "summary": "Glossary of climate and environment terms.",
            "content": "Definitions for climate adaptation, mitigation, and resilience.",
            "_distance": 0.07,
        },
    ]

    chunks, documents = _select_documents_and_chunks(rows, profile, limit=4)

    assert documents
    assert chunks
    assert documents[0].title == "2025 Tracking SDG7 Report"
    off_topic_titles = {
        "Forests, Energy and Livelihoods",
        "Climate Dictionary",
    }
    top_three_titles = [document.title for document in documents[:3]]
    assert sum(title in off_topic_titles for title in top_three_titles) == 0


def test_document_row_requires_real_signal_not_only_priors():
    profile = _build_retrieval_profile("Tell me more about feed in tariff")
    row = {
        "document_id": "doc-1",
        "source_id": "undp",
        "canonical_title": "Air quality monitoring data for analysis of the pace and intensity of the coronavirus spread",
        "summary": "Air quality observations and pandemic analysis in Central and Eastern Europe.",
        "document_type": "policy",
        "status": "approved",
        "quality_score": 1.0,
        "source_priority": 1.0,
        "year": 2021,
        "topic_tags": [
            "energy efficiency",
            "climate mitigation",
            "grid infrastructure",
        ],
        "region_codes": ["europe"],
    }

    assert _document_has_min_signal(row, profile) is False
    assert _score_document_row(row, profile) < 0.8


def test_geography_classifier_rejects_sibling_country_for_country_query():
    profile = _build_retrieval_profile("Tell me more about access to electricity in Nigeria")
    row = {
        "document_id": "doc-nga-miss",
        "canonical_title": "Kenya Electricity Access Brief",
        "summary": "Electricity access trends in Kenya.",
        "country_codes": ["KEN"],
        "region_codes": ["africa"],
    }

    geo = _classify_geography_match(row, profile)

    assert geo["match_class"] == "out_of_scope"
    assert _document_has_min_signal(row, profile) is False


def test_geography_classifier_accepts_country_and_parent_region_fallback():
    profile = _build_retrieval_profile("Tell me more about access to electricity in Nigeria")
    nigeria_row = {
        "document_id": "doc-nga",
        "canonical_title": "Nigeria Electricity Access Brief",
        "summary": "Electricity access trends in Nigeria.",
        "country_codes": ["NGA"],
        "region_codes": ["africa"],
    }
    africa_row = {
        "document_id": "doc-africa",
        "canonical_title": "Africa Energy Access Outlook",
        "summary": "Regional electricity access trends across Africa.",
        "region_codes": ["africa"],
    }

    assert _classify_geography_match(nigeria_row, profile)["match_class"] == "exact_country"
    assert _classify_geography_match(africa_row, profile)["match_class"] == "parent_region"


def test_document_selection_rejects_africa_for_lac_query():
    profile = _build_retrieval_profile(
        "What is the latest progress on access to electricity in Latin America and the Caribbean?"
    )
    rows = [
        {
            "title": "Africa Energy Access Outlook",
            "year": 2025,
            "language": "en",
            "url": "https://example.org/africa-access",
            "summary": "Regional electricity access trends across Africa.",
            "content": "Access to electricity trends across Africa.",
            "region_codes": ["africa"],
            "_distance": 0.08,
        },
        {
            "title": "LAC Energy Access Outlook",
            "year": 2025,
            "language": "en",
            "url": "https://example.org/lac-access",
            "summary": "Regional electricity access trends across Latin America and the Caribbean.",
            "content": "Access to electricity trends across Latin America and the Caribbean.",
            "region_codes": ["latin america"],
            "_distance": 0.11,
        },
        {
            "title": "Global Electricity Access Update",
            "year": 2025,
            "language": "en",
            "url": "https://example.org/global-access",
            "summary": "Global electricity access update.",
            "content": "Worldwide access to electricity update.",
            "region_codes": ["global"],
            "_distance": 0.09,
        },
    ]

    chunks, documents = _select_documents_and_chunks(rows, profile, limit=4)

    assert documents
    assert all("Africa Energy Access Outlook" != document.title for document in documents)
    assert any(document.title == "LAC Energy Access Outlook" for document in documents)
    assert any(document.title == "Global Electricity Access Update" for document in documents)


def test_summary_fallback_respects_geographic_scope():
    profile = _build_retrieval_profile("Tell me more about access to electricity in Nigeria")
    rows = [
        {
            "document_id": "doc-ken",
            "canonical_title": "Kenya Electricity Access Brief",
            "summary": "Electricity access trends in Kenya.",
            "country_codes": ["KEN"],
            "region_codes": ["africa"],
            "year": 2025,
            "language": "en",
            "url": "https://example.org/kenya-access",
        },
        {
            "document_id": "doc-africa",
            "canonical_title": "Africa Energy Access Outlook",
            "summary": "Regional electricity access trends across Africa.",
            "region_codes": ["africa"],
            "year": 2025,
            "language": "en",
            "url": "https://example.org/africa-access",
        },
    ]

    chunks = _build_summary_fallback_chunks(rows, profile, limit=3)

    assert len(chunks) == 1
    assert chunks[0].title == "Africa Energy Access Outlook"


def test_document_row_accepts_topic_backed_infrastructure_match():
    profile = _build_retrieval_profile("Tell me more about grid infrastructure")
    row = {
        "document_id": "doc-2",
        "source_id": "undp",
        "canonical_title": "Resilient power systems planning note",
        "summary": "Transmission resilience and power system modernization guidance.",
        "document_type": "policy",
        "status": "approved",
        "quality_score": 0.9,
        "source_priority": 1.0,
        "year": 2024,
        "topic_tags": ["grid infrastructure", "renewable integration"],
        "region_codes": ["global"],
    }

    assert _document_has_min_signal(row, profile) is True
    assert _score_document_row(row, profile) >= 0.8


def test_document_row_requires_focus_phrase_for_feed_in_tariff():
    profile = _build_retrieval_profile("Tell me more about feed in tariff")
    row = {
        "document_id": "doc-3",
        "source_id": "undp",
        "canonical_title": "BIOFIN Synthesis Report for Thailand",
        "summary": (
            "Findings from the three assessments feed into the formulation of the country's "
            "Biodiversity Finance Plan."
        ),
        "document_type": "policy",
        "status": "approved",
        "quality_score": 1.0,
        "source_priority": 1.0,
        "year": 2020,
        "topic_tags": ["energy finance"],
        "region_codes": ["asia"],
    }

    assert _document_has_min_signal(row, profile) is False


def test_summary_fallback_chunks_use_document_summaries():
    profile = _build_retrieval_profile("Tell me more about access to electricity")
    rows = [
        {
            "document_id": "doc-4",
            "source_id": "undp",
            "canonical_title": "Energy Access Brief",
            "summary": "Access to electricity remains uneven across rural areas.",
            "year": 2025,
            "language": "English",
            "url": "https://example.org/energy-access-brief",
        }
    ]

    chunks = _build_summary_fallback_chunks(rows, profile, limit=3)

    assert len(chunks) == 1
    assert chunks[0].title == "Energy Access Brief"
    assert "Access to electricity" in chunks[0].content


@pytest.mark.asyncio
async def test_retrieve_chunks_uses_lexical_matches_before_vector_search():
    rows = [
        {
            "title": "Feed-in Tariff Policy Toolkit",
            "year": 2024,
            "language": "en",
            "url": "https://example.org/fit-toolkit",
            "summary": "Guidance on feed-in tariff design and reform.",
            "content": "Feed-in tariff structures, degression, procurement, and market integration.",
        },
        {
            "title": "Renewable Tariff Design Handbook",
            "year": 2023,
            "language": "en",
            "url": "https://example.org/fit-handbook",
            "summary": "Practical guidance for feed-in tariffs and related renewable pricing policies.",
            "content": "Feed in tariff implementation, payment design, and project bankability.",
        },
    ]

    class FakeQuery:
        def __init__(self, rows):
            self.rows = rows

        def select(self, _fields):
            return self

        def where(self, _clause):
            return self

        def limit(self, _limit):
            return self

        async def to_list(self):
            return self.rows

    class FakeTable:
        def __init__(self, rows):
            self.rows = rows

        def query(self):
            return FakeQuery(self.rows)

        def vector_search(self, _vector):
            raise AssertionError("vector search should not run when lexical matches exist")

    class FakeConnection:
        def __init__(self, rows):
            self.rows = rows

        async def open_table(self, name):
            if name == "chunks":
                return FakeTable(self.rows)
            raise ValueError(f"Table {name} was not found")

    class FailingEmbedder:
        async def aembed_query(self, _query):
            raise AssertionError("embedder should not run when lexical matches exist")

    client = Client(FakeConnection(rows))
    client.embedder = FailingEmbedder()

    chunks, documents = await client.retrieve_chunks("Tell me more about feed in tariff")

    assert len(chunks) >= 2
    assert len(documents) >= 2
    assert documents[0].title == "Feed-in Tariff Policy Toolkit"


def test_build_document_record_enriches_metadata():
    record = corpus.build_document_record(
        [
            {
                "title": "2025 Tracking SDG7 Report: Access to Electricity in Africa",
                "year": 2025,
                "language": "en",
                "url": "https://trackingsdg7.esmap.org/downloads/report.pdf",
                "summary": "Latest progress on electricity access and renewable energy in Sub-Saharan Africa.",
                "content": "Electricity access, renewable energy, SDG 7 indicators, and Africa regional trends.",
            }
        ]
    )

    assert record.source_id == "tracking_sdg7"
    assert record.document_type in {"flagship_report", "report"}
    assert record.is_flagship is True
    assert "energy access" in (record.topic_tags or [])
    assert "africa" in (record.region_codes or [])
    assert "SDG7" in (record.sdg_tags or [])
    assert record.status == "approved"


@pytest.mark.asyncio
async def test_retrieve_chunks_prefers_documents_table_when_available():
    document_rows = [
        {
            "document_id": "doc-1",
            "source_id": "undp",
            "canonical_title": "Feed-in Tariff Policy Toolkit",
            "url": "https://example.org/fit-toolkit",
            "language": "en",
            "document_type": "policy",
            "publication_date": "2024-01-01",
            "year": 2024,
            "summary": "Guidance on feed-in tariff design and renewable energy tariff reform.",
            "status": "approved",
            "publisher": "UNDP",
            "series_name": None,
            "topic_tags": ["energy finance", "renewable energy"],
            "topic_tags_text": "energy finance | renewable energy",
            "region_codes": ["global"],
            "geography_tags_text": "global",
            "audience_tags": ["policy-makers"],
            "audience_tags_text": "policy-makers",
            "authority_tier": "trusted",
            "source_priority": 1.0,
            "quality_score": 0.9,
            "is_flagship": False,
            "is_data_report": False,
        },
        {
            "document_id": "doc-2",
            "source_id": "world_bank",
            "canonical_title": "Renewable Tariff Design Handbook",
            "url": "https://example.org/fit-handbook",
            "language": "en",
            "document_type": "policy",
            "publication_date": "2023-01-01",
            "year": 2023,
            "summary": "Practical guidance for feed-in tariffs and procurement rules.",
            "status": "approved",
            "publisher": "World Bank",
            "series_name": None,
            "topic_tags": ["energy finance"],
            "topic_tags_text": "energy finance",
            "region_codes": ["global"],
            "geography_tags_text": "global",
            "audience_tags": ["policy-makers"],
            "audience_tags_text": "policy-makers",
            "authority_tier": "trusted",
            "source_priority": 1.0,
            "quality_score": 0.85,
            "is_flagship": False,
            "is_data_report": False,
        },
    ]
    chunk_rows = [
        {
            "document_id": "doc-1",
            "title": "Feed-in Tariff Policy Toolkit",
            "year": 2024,
            "language": "en",
            "url": "https://example.org/fit-toolkit",
            "summary": "Guidance on feed-in tariff design and renewable energy tariff reform.",
            "content": "Feed-in tariff degression, tariff setting, policy sequencing, and market integration.",
            "chunk_id": "chunk-1",
            "chunk_index": 0,
        },
        {
            "document_id": "doc-2",
            "title": "Renewable Tariff Design Handbook",
            "year": 2023,
            "language": "en",
            "url": "https://example.org/fit-handbook",
            "summary": "Practical guidance for feed-in tariffs and procurement rules.",
            "content": "Feed in tariff implementation and procurement design for renewable markets.",
            "chunk_id": "chunk-2",
            "chunk_index": 0,
        },
    ]

    class FakeSchemaField:
        def __init__(self, name):
            self.name = name

    class FakeQuery:
        def __init__(self, rows):
            self.rows = rows

        def select(self, _fields=None):
            return self

        def where(self, _clause):
            return self

        def limit(self, _limit):
            return self

        async def to_list(self):
            return self.rows

    class FakeVectorSearch(FakeQuery):
        pass

    class FakeTable:
        def __init__(self, rows):
            self.rows = rows

        def query(self):
            return FakeQuery(self.rows)

        def vector_search(self, _vector):
            return FakeVectorSearch(self.rows)

        async def schema(self):
            if self.rows:
                return [FakeSchemaField(name) for name in self.rows[0].keys()]
            return []

    class FakeConnection:
        async def table_names(self):
            return ["chunks", "documents"]

        async def open_table(self, name):
            if name == "chunks":
                return FakeTable(chunk_rows)
            if name == "documents":
                return FakeTable(document_rows)
            raise ValueError(name)

    class FakeEmbedder:
        async def aembed_query(self, _query):
            return [0.1]

    client = Client(FakeConnection())
    client.embedder = FakeEmbedder()

    chunks, documents = await client.retrieve_chunks("Tell me more about feed in tariff")

    assert len(documents) >= 2
    assert documents[0].document_id == "doc-1"
    assert documents[0].document_type == "policy"
    assert len(chunks) >= 2
    assert all(chunk.document_id for chunk in chunks)


@pytest.mark.asyncio
async def test_retrieve_chunks_enforces_country_scope_before_document_selection():
    document_rows = [
        {
            "document_id": "doc-nga",
            "source_id": "undp",
            "canonical_title": "Nigeria Electricity Access Brief",
            "url": "https://example.org/nigeria-access",
            "language": "en",
            "document_type": "report",
            "publication_date": "2024-01-01",
            "year": 2024,
            "summary": "Electricity access trends in Nigeria.",
            "status": "approved",
            "publisher": "UNDP",
            "region_codes": ["africa"],
            "country_codes": ["NGA"],
            "geography_tags_text": "africa | NGA",
            "authority_tier": "trusted",
            "source_priority": 1.0,
            "quality_score": 0.9,
            "is_flagship": False,
            "is_data_report": False,
        },
        {
            "document_id": "doc-ken",
            "source_id": "undp",
            "canonical_title": "Kenya Electricity Access Brief",
            "url": "https://example.org/kenya-access",
            "language": "en",
            "document_type": "report",
            "publication_date": "2024-01-01",
            "year": 2024,
            "summary": "Electricity access trends in Kenya.",
            "status": "approved",
            "publisher": "UNDP",
            "region_codes": ["africa"],
            "country_codes": ["KEN"],
            "geography_tags_text": "africa | KEN",
            "authority_tier": "trusted",
            "source_priority": 1.0,
            "quality_score": 0.9,
            "is_flagship": False,
            "is_data_report": False,
        },
        {
            "document_id": "doc-global",
            "source_id": "world_bank",
            "canonical_title": "Global Electricity Access Update",
            "url": "https://example.org/global-access",
            "language": "en",
            "document_type": "report",
            "publication_date": "2025-01-01",
            "year": 2025,
            "summary": "Global electricity access update.",
            "status": "approved",
            "publisher": "World Bank",
            "region_codes": ["global"],
            "geography_tags_text": "global",
            "authority_tier": "trusted",
            "source_priority": 1.0,
            "quality_score": 0.8,
            "is_flagship": False,
            "is_data_report": True,
        },
    ]
    chunk_rows = [
        {
            "document_id": "doc-nga",
            "title": "Nigeria Electricity Access Brief",
            "year": 2024,
            "language": "en",
            "url": "https://example.org/nigeria-access",
            "summary": "Electricity access trends in Nigeria.",
            "content": "Nigeria electricity access trends and off-grid deployment.",
            "country_codes": ["NGA"],
            "region_codes": ["africa"],
            "chunk_id": "chunk-nga",
            "chunk_index": 0,
        },
        {
            "document_id": "doc-ken",
            "title": "Kenya Electricity Access Brief",
            "year": 2024,
            "language": "en",
            "url": "https://example.org/kenya-access",
            "summary": "Electricity access trends in Kenya.",
            "content": "Kenya electricity access trends and mini-grid deployment.",
            "country_codes": ["KEN"],
            "region_codes": ["africa"],
            "chunk_id": "chunk-ken",
            "chunk_index": 0,
        },
        {
            "document_id": "doc-global",
            "title": "Global Electricity Access Update",
            "year": 2025,
            "language": "en",
            "url": "https://example.org/global-access",
            "summary": "Global electricity access update.",
            "content": "Worldwide electricity access update and SDG 7 progress.",
            "region_codes": ["global"],
            "chunk_id": "chunk-global",
            "chunk_index": 0,
        },
    ]

    class FakeSchemaField:
        def __init__(self, name):
            self.name = name

    class FakeQuery:
        def __init__(self, rows):
            self.rows = rows

        def select(self, _fields=None):
            return self

        def where(self, _clause):
            return self

        def limit(self, _limit):
            return self

        async def to_list(self):
            return self.rows

    class FakeVectorSearch(FakeQuery):
        pass

    class FakeTable:
        def __init__(self, rows):
            self.rows = rows

        def query(self):
            return FakeQuery(self.rows)

        def vector_search(self, _vector):
            return FakeVectorSearch(self.rows)

        async def schema(self):
            if self.rows:
                return [FakeSchemaField(name) for name in self.rows[0].keys()]
            return []

    class FakeConnection:
        async def table_names(self):
            return ["chunks", "documents"]

        async def open_table(self, name):
            if name == "chunks":
                return FakeTable(chunk_rows)
            if name == "documents":
                return FakeTable(document_rows)
            raise ValueError(name)

    class FakeEmbedder:
        async def aembed_query(self, _query):
            return [0.1]

    client = Client(FakeConnection())
    client.embedder = FakeEmbedder()

    chunks, documents = await client.retrieve_chunks(
        "Tell me more about access to electricity in Nigeria"
    )

    assert any(document.document_id == "doc-nga" for document in documents)
    assert any(document.document_id == "doc-global" for document in documents)
    assert all(document.document_id != "doc-ken" for document in documents)
    assert all(chunk.document_id != "doc-ken" for chunk in chunks)


@pytest.mark.asyncio
async def test_retrieve_chunks_seeds_latest_sdg7_report_for_current_data_query():
    document_rows = [
        {
            "document_id": "doc-forest",
            "source_id": "undp",
            "canonical_title": "Forests, Energy and Livelihoods",
            "url": "https://example.org/forests-energy-livelihoods",
            "language": "en",
            "document_type": "report",
            "publication_date": "2023-01-01",
            "year": 2023,
            "summary": "Woodfuel use, forests, biomass reliance, and household livelihoods.",
            "status": "approved",
            "publisher": "UNDP",
            "region_codes": ["global"],
            "geography_tags_text": "global",
            "authority_tier": "partner",
            "source_priority": 1.0,
            "quality_score": 1.0,
            "is_flagship": False,
            "is_data_report": True,
        },
        {
            "document_id": "doc-sdg7",
            "source_id": "tracking_sdg7",
            "canonical_title": "2025 Tracking SDG7 Report",
            "url": "https://trackingsdg7.esmap.org/downloads",
            "language": "en",
            "document_type": "flagship_report",
            "publication_date": "2025-01-01",
            "year": 2025,
            "summary": "Annual SDG7 tracking report.",
            "status": "approved",
            "publisher": "Tracking SDG7 / ESMAP",
            "region_codes": ["global"],
            "geography_tags_text": "global",
            "authority_tier": "trusted",
            "source_priority": 1.0,
            "quality_score": 1.0,
            "is_flagship": True,
            "is_data_report": True,
        },
    ]
    chunk_rows = [
        {
            "document_id": "doc-forest",
            "title": "Forests, Energy and Livelihoods",
            "year": 2023,
            "language": "en",
            "url": "https://example.org/forests-energy-livelihoods",
            "summary": "Woodfuel use, forests, biomass reliance, and household livelihoods.",
            "content": "More than 2.4 billion people rely on polluting cooking systems.",
            "region_codes": ["global"],
            "chunk_id": "chunk-forest",
            "chunk_index": 0,
        },
        {
            "document_id": "doc-sdg7",
            "title": "2025 Tracking SDG7 Report",
            "year": 2025,
            "language": "en",
            "url": "https://trackingsdg7.esmap.org/downloads",
            "summary": "Annual SDG7 tracking report.",
            "content": "In 2023, 666 million people remained without access to electricity worldwide.",
            "region_codes": ["global"],
            "chunk_id": "chunk-sdg7",
            "chunk_index": 0,
        },
        {
            "document_id": "doc-sdg7",
            "title": "2025 Tracking SDG7 Report",
            "year": 2025,
            "language": "en",
            "url": "https://trackingsdg7.esmap.org/downloads",
            "summary": "Annual SDG7 tracking report.",
            "content": (
                "Access to clean cooking remains part of SDG7 tracking; in 2023, "
                "2.1 billion people relied on polluting fuels and technologies."
            ),
            "region_codes": ["global"],
            "chunk_id": "chunk-sdg7-clean-cooking",
            "chunk_index": 1,
        },
    ]

    class FakeSchemaField:
        def __init__(self, name):
            self.name = name

    class FakeQuery:
        def __init__(self, rows):
            self.rows = rows

        def select(self, _fields=None):
            return self

        def where(self, _clause):
            return self

        def limit(self, _limit):
            return self

        async def to_list(self):
            return self.rows

    class FakeVectorSearch(FakeQuery):
        pass

    class FakeTable:
        def __init__(self, rows):
            self.rows = rows

        def query(self):
            return FakeQuery(self.rows)

        def vector_search(self, _vector):
            return FakeVectorSearch(self.rows)

        async def schema(self):
            if self.rows:
                return [FakeSchemaField(name) for name in self.rows[0].keys()]
            return []

    class FakeConnection:
        async def table_names(self):
            return ["chunks", "documents"]

        async def open_table(self, name):
            if name == "chunks":
                return FakeTable(chunk_rows)
            if name == "documents":
                return FakeTable(document_rows)
            raise ValueError(name)

    class FakeEmbedder:
        async def aembed_query(self, _query):
            return [0.1]

    client = Client(FakeConnection())
    client.embedder = FakeEmbedder()

    chunks, documents = await client.retrieve_chunks("How many people lack access to energy?")

    assert documents
    assert chunks
    assert documents[0].document_id == "doc-sdg7"
    assert chunks[0].document_id == "doc-sdg7"
    assert len(documents) == 1
    assert any("clean cooking" in chunk.content.lower() for chunk in chunks)
    assert all(document.document_id == "doc-sdg7" for document in documents)


@pytest.mark.asyncio
async def test_retrieve_chunks_uses_trusted_sdg7_metric_when_chunks_are_unavailable():
    document_rows = [
        {
            "document_id": "doc-forest",
            "source_id": "undp",
            "canonical_title": "Forests, Energy and Livelihoods",
            "url": "https://example.org/forests-energy-livelihoods",
            "language": "en",
            "document_type": "report",
            "publication_date": "2023-01-01",
            "year": 2023,
            "summary": "Woodfuel use, forests, biomass reliance, and household livelihoods.",
            "status": "approved",
            "publisher": "UNDP",
            "region_codes": ["global"],
            "geography_tags_text": "global",
            "authority_tier": "partner",
            "source_priority": 1.0,
            "quality_score": 1.0,
            "is_flagship": False,
            "is_data_report": True,
        },
        {
            "document_id": "doc-sdg7",
            "source_id": "tracking_sdg7",
            "canonical_title": "2025 Tracking SDG7 Report",
            "url": "https://trackingsdg7.esmap.org/downloads",
            "language": "en",
            "document_type": "flagship_report",
            "publication_date": "2025-01-01",
            "year": 2025,
            "summary": "Annual SDG7 tracking report.",
            "status": "approved",
            "publisher": "Tracking SDG7 / ESMAP",
            "region_codes": ["global"],
            "geography_tags_text": "global",
            "authority_tier": "trusted",
            "source_priority": 1.0,
            "quality_score": 1.0,
            "is_flagship": True,
            "is_data_report": True,
        },
    ]

    class FakeSchemaField:
        def __init__(self, name):
            self.name = name

    class FakeQuery:
        def __init__(self, rows):
            self.rows = rows

        def select(self, _fields=None):
            return self

        def where(self, _clause):
            return self

        def limit(self, _limit):
            return self

        async def to_list(self):
            return self.rows

    class FakeVectorSearch(FakeQuery):
        pass

    class FakeTable:
        def __init__(self, rows, fields=None):
            self.rows = rows
            self.fields = fields

        def query(self):
            return FakeQuery(self.rows)

        def vector_search(self, _vector):
            return FakeVectorSearch(self.rows)

        async def schema(self):
            fields = self.fields or (self.rows[0].keys() if self.rows else [])
            return [FakeSchemaField(name) for name in fields]

    class FakeConnection:
        async def table_names(self):
            return ["chunks", "documents"]

        async def open_table(self, name):
            if name == "chunks":
                return FakeTable(
                    [],
                    fields=[
                        "document_id",
                        "title",
                        "year",
                        "language",
                        "url",
                        "summary",
                        "content",
                        "chunk_id",
                    ],
                )
            if name == "documents":
                return FakeTable(document_rows)
            raise ValueError(name)

    class FakeEmbedder:
        async def aembed_query(self, _query):
            return [0.1]

    client = Client(FakeConnection())
    client.embedder = FakeEmbedder()

    debug = {}
    chunks, documents = await client.retrieve_chunks(
        "How many people lack access to energy?",
        debug=debug,
    )

    assert documents[0].document_id == "doc-sdg7"
    assert chunks[0].document_id == "doc-sdg7"
    assert chunks[0].content_type == "trusted_metric_fallback"
    assert "666 million" in chunks[0].content
    assert "770 million" not in chunks[0].content
    assert any(chunk.content_type == "trusted_context_fallback" for chunk in chunks)
    assert any("clean cooking" in chunk.content.lower() for chunk in chunks)
    assert any(
        branch.get("stage") == "trusted_context_fallback"
        for branch in debug.get("branches", [])
    )
    assert debug.get("selected", {}).get("path") == "current_data_metric_plus_context"
