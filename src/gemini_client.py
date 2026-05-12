"""Gemini client and JSON parsing helpers for BuyerLab AI."""

from __future__ import annotations

import json
import os
import re
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is listed for local runs.
    load_dotenv = None


DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
MOCK_JSON_RESPONSE: dict[str, Any] = {
    "mock_mode": True,
    "decision": "hesitate",
    "confidence": 72,
    "purchase_intent": 58,
    "main_reason": "Deterministic mock response for demos and tests.",
    "objections": ["Needs stronger proof", "Value is not fully clear"],
    "missing_information": ["Warranty details", "Shipping cost"],
    "suggested_fix": "Add clearer trust signals, shipping details, and purchase proof.",
}


class GeminiClientError(RuntimeError):
    """Base error for Gemini client failures."""


class MissingGeminiApiKeyError(GeminiClientError):
    """Raised when Gemini is called without a configured API key."""


class InvalidGeminiResponseError(GeminiClientError):
    """Raised when Gemini returns text that cannot be parsed as JSON."""


def generate_text(prompt: str) -> str:
    """Generate raw text from Gemini for a non-empty prompt."""
    prompt = _validate_prompt(prompt)

    if _mock_mode_enabled():
        return json.dumps(MOCK_JSON_RESPONSE)

    client = _create_client()
    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    response = client.models.generate_content(model=model, contents=prompt)
    text = getattr(response, "text", None)

    if not text:
        raise GeminiClientError("Gemini returned an empty response.")

    return text


def generate_json(prompt: str) -> dict[str, Any]:
    """Generate structured JSON from Gemini and return it as a dictionary."""
    prompt = _validate_prompt(prompt)

    if _mock_mode_enabled():
        return dict(MOCK_JSON_RESPONSE)

    return _extract_json_object(generate_text(prompt))


def _create_client() -> Any:
    """Create a Gemini client from the GEMINI_API_KEY environment variable."""
    _load_env()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise MissingGeminiApiKeyError(
            "GEMINI_API_KEY is missing. Set it in your environment or enable "
            "BUYERLAB_MOCK_MODE=true for demos and tests."
        )

    try:
        from google import genai
    except ImportError as exc:
        raise GeminiClientError(
            "google-genai is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    return genai.Client(api_key=api_key)


def _extract_json_object(response_text: str) -> dict[str, Any]:
    """Extract a JSON object from raw model text, including markdown fences."""
    if not response_text or not response_text.strip():
        raise InvalidGeminiResponseError("Gemini returned an empty JSON response.")

    candidates = _json_candidates(response_text)
    for candidate in candidates:
        parsed = _try_parse_json_object(candidate)
        if parsed is not None:
            return parsed

    raise InvalidGeminiResponseError(
        "Gemini response did not contain a valid JSON object."
    )


def _json_candidates(response_text: str) -> list[str]:
    """Return likely JSON snippets from raw text in priority order."""
    text = response_text.strip()
    fenced_blocks = re.findall(
        r"```(?:json)?\s*(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidates = [text, *[block.strip() for block in fenced_blocks]]

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            _, end_index = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        candidates.append(text[index : index + end_index])

    return candidates


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse text as a JSON object, returning None when parsing fails."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        raise InvalidGeminiResponseError("Gemini JSON response must be an object.")

    return parsed


def _validate_prompt(prompt: str) -> str:
    """Validate and normalize a prompt before sending it to Gemini."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")
    return prompt.strip()


def _mock_mode_enabled() -> bool:
    """Return True when deterministic local mock responses should be used."""
    _load_env()
    return os.getenv("BUYERLAB_MOCK_MODE", "").strip().lower() == "true"


def _load_env() -> None:
    """Load local environment variables when python-dotenv is available."""
    if load_dotenv is not None:
        load_dotenv()
