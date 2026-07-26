"""Multilingual behavior for SGP assistant answers, ideas, and retrieval."""

import pytest

from src import genai
from src.entities import Message


@pytest.mark.parametrize(
    ("question", "fallback", "expected"),
    [
        ("How have SGP grants supported coastal resilience?", "fr", "en"),
        ("Como as subvenções apoiaram a resiliência costeira?", "en", "pt"),
        ("Comment les subventions ont-elles soutenu la résilience côtière ?", "en", "fr"),
        ("¿Cómo han apoyado las subvenciones la resiliencia costera?", "en", "es"),
        ("Как гранты поддержали устойчивость прибрежных районов?", "en", "ru"),
        ("这些赠款如何支持沿海韧性？", "en", "zh"),
        ("كيف دعمت المنح القدرة على الصمود الساحلي؟", "en", "ar"),
        ("OP8?", "fr", "fr"),
    ],
)
def test_question_locale_inference_for_fallbacks(question, fallback, expected):
    assert genai.infer_question_locale(question, fallback) == expected


def test_answer_instruction_uses_question_language_and_ui_fallback():
    instruction = genai.answer_language_instruction("pt")

    assert "language of the latest human question" in instruction
    assert "Portuguese" in instruction
    assert "selected interface language" in instruction


def test_localized_scope_content_uses_answer_and_ui_languages_separately():
    refusal = genai.localized_assistant_text("outside_sgp_scope", "fr")
    ideas = genai.localized_scope_ideas("ar")

    assert "publications approuvées du SGP" in refusal
    assert len(ideas) == 3
    assert all(any("\u0600" <= character <= "\u06ff" for character in idea) for idea in ideas)


@pytest.mark.asyncio
async def test_suggested_questions_follow_ui_locale(monkeypatch):
    captured = {}

    async def fake_generate_response(*, system_message, schema, **_kwargs):
        captured["system_message"] = system_message
        return schema(ideas=["¿Qué muestran los informes?"])

    monkeypatch.setattr(genai, "generate_response", fake_generate_response)

    ideas = await genai.generate_query_ideas(
        [Message(role="human", content="What have SGP grants achieved?")],
        ui_locale="es",
    )

    assert ideas == ["¿Qué muestran los informes?"]
    assert "entirely in Spanish" in captured["system_message"]
    assert "regardless of the language used in the conversation" in captured["system_message"]


@pytest.mark.asyncio
async def test_retrieval_translation_preserves_search_scope(monkeypatch):
    captured = {}

    async def fake_generate_response(*, prompt, system_message, schema, **_kwargs):
        captured["prompt"] = prompt
        captured["system_message"] = system_message
        return schema(
            english_query="How have SGP grants supported coastal resilience in Senegal in 2024?"
        )

    monkeypatch.setattr(genai, "generate_response", fake_generate_response)

    translated = await genai.translate_query_for_retrieval(
        "Comment les subventions du SGP ont-elles soutenu la résilience côtière au Sénégal en 2024 ?"
    )

    assert translated == "How have SGP grants supported coastal resilience in Senegal in 2024?"
    assert "Sénégal" in captured["prompt"]
    assert "named place" in captured["system_message"]
    assert "date" in captured["system_message"]
    assert "Do not answer" in captured["system_message"]
