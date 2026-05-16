import pytest

from src.gemini_client import (
    InvalidGeminiResponseError,
    MissingGeminiApiKeyError,
    _extract_json_object,
    check_gemini_connection,
    generate_json,
    generate_text,
)


def test_generate_json_uses_mock_mode_without_api_key(monkeypatch):
    monkeypatch.setenv("BUYERLAB_MOCK_MODE", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "")

    result = generate_json("Return a buyer decision as JSON.")
    status = check_gemini_connection()

    assert result["mock_mode"] is True
    assert status["ok"] is True
    assert status["mode"] == "mock"


def test_live_mode_without_api_key_raises_friendly_error(monkeypatch):
    monkeypatch.setenv("BUYERLAB_MOCK_MODE", "false")
    monkeypatch.setenv("GEMINI_API_KEY", "")

    with pytest.raises(MissingGeminiApiKeyError, match="GEMINI_API_KEY is missing"):
        generate_text("Hello Gemini")

    status = check_gemini_connection()
    assert status["ok"] is False
    assert status["mode"] == "live"
    assert "GEMINI_API_KEY is missing" in status["message"]


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ('{"ok": true}', {"ok": True}),
        ('```json\n{"decision": "buy"}\n```', {"decision": "buy"}),
        ('Extra text before {"score": 87} and after.', {"score": 87}),
    ],
)
def test_extract_json_object_supported_formats(raw_text, expected):
    assert _extract_json_object(raw_text) == expected


def test_extract_json_object_rejects_invalid_or_empty_response():
    with pytest.raises(InvalidGeminiResponseError):
        _extract_json_object("")

    with pytest.raises(InvalidGeminiResponseError):
        _extract_json_object("No JSON here.")
