"""Gemini client and JSON parsing helpers for BuyerLab AI."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is listed for local runs.
    load_dotenv = None


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
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

    model = _configured_model()

    try:
        response_text = _request_gemini_text(prompt, model)
    except Exception as exc:  # pragma: no cover - depends on remote API behavior.
        if isinstance(exc, GeminiClientError):
            raise
        raise GeminiClientError(
            f"Gemini request failed safely: {_safe_error_message(exc)}"
        ) from exc

    if not response_text:
        raise GeminiClientError("Gemini returned an empty response.")

    return response_text


def generate_json(prompt: str) -> dict[str, Any]:
    """Generate structured JSON from Gemini and return it as a dictionary."""
    prompt = _validate_prompt(prompt)

    if _mock_mode_enabled():
        return dict(MOCK_JSON_RESPONSE)

    return _extract_json_object(generate_text(prompt))


def check_gemini_connection() -> dict[str, Any]:
    """Return safe Gemini readiness status without exposing secrets."""
    if _mock_mode_enabled():
        return {
            "ok": True,
            "mode": "mock",
            "model": _configured_model(),
            "message": "Mock mode is enabled; Gemini API will not be called.",
        }

    try:
        _create_client()
    except MissingGeminiApiKeyError as exc:
        return {
            "ok": False,
            "mode": "live",
            "model": _configured_model(),
            "message": str(exc),
        }
    except GeminiClientError as exc:
        return {
            "ok": False,
            "mode": "live",
            "model": _configured_model(),
            "message": str(exc),
        }

    return {
        "ok": True,
        "mode": "live",
        "model": _configured_model(),
        "message": "Gemini client is ready.",
    }


def _create_client() -> Any:
    """Validate Gemini configuration without importing a provider SDK."""
    _load_env()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise MissingGeminiApiKeyError(
            "GEMINI_API_KEY is missing. Set it in your environment or enable "
            "BUYERLAB_MOCK_MODE=true for demos and tests."
        )

    return {"api_key_configured": True}


def _request_gemini_text(prompt: str, model: str) -> str:
    """Call Gemini through REST so cPanel does not need Google SDK packages."""
    _load_env()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise MissingGeminiApiKeyError(
            "GEMINI_API_KEY is missing. Set it in your environment or enable "
            "BUYERLAB_MOCK_MODE=true for demos and tests."
        )

    model_name = model[7:] if model.startswith("models/") else model
    encoded_model = urllib.parse.quote(model_name, safe="")
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{encoded_model}:generateContent?key={urllib.parse.quote(api_key, safe='')}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # pragma: no cover - remote API behavior.
        error_body = exc.read().decode("utf-8", errors="replace")[:300]
        raise GeminiClientError(
            f"Gemini request failed with HTTP {exc.code}: {_safe_error_message(Exception(error_body))}"
        ) from exc
    except urllib.error.URLError as exc:  # pragma: no cover - remote API behavior.
        raise GeminiClientError(
            f"Gemini request could not reach the API: {_safe_error_message(exc)}"
        ) from exc

    try:
        parsed = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise GeminiClientError("Gemini returned a non-JSON API response.") from exc

    candidates = parsed.get("candidates") or []
    if not candidates:
        raise GeminiClientError("Gemini returned no candidates.")

    parts = (
        candidates[0]
        .get("content", {})
        .get("parts", [])
    )
    texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    return "\n".join(text for text in texts if text).strip()


def _configured_model() -> str:
    """Read the configured Gemini model, using a fast demo-friendly default."""
    _load_env()
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL


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


def _try_parse_json_object(text: str) -> Optional[dict[str, Any]]:
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
    return os.getenv("BUYERLAB_MOCK_MODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _load_env() -> None:
    """Load local .env values without printing or exposing environment secrets."""
    if load_dotenv is not None:
        load_dotenv()


def _safe_error_message(exc: Exception) -> str:
    """Format remote/client errors without leaking configured secrets."""
    message = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key:
        message = message.replace(api_key, "[redacted]")
    return message[:180]
