"""Placeholder agent functions for BuyerLab AI."""

from __future__ import annotations

import os
from typing import Any

from src.prompts import build_buyer_prompt
from src.state import SimulationState


def get_gemini_api_key() -> str | None:
    """Read the Gemini API key from the environment."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()

    return os.getenv("GEMINI_API_KEY")


def create_gemini_client() -> Any | None:
    """Create a Gemini client when configuration and dependencies are available."""
    api_key = get_gemini_api_key()
    if not api_key:
        return None

    try:
        from google import genai
    except ImportError:
        return None

    return genai.Client(api_key=api_key)


def simulate_buyer_response(state: SimulationState) -> SimulationState:
    """Generate or stub a buyer response."""
    client = create_gemini_client()
    prompt = build_buyer_prompt(state)

    if client is None:
        state["buyer_response"] = (
            "Placeholder buyer response: the product sounds useful, but the buyer "
            "would need clearer proof, reviews, and a stronger reason to buy now."
        )
        return state

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    state["buyer_response"] = response.text or "No response generated."
    return state
