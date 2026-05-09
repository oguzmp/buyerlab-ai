"""Prompt templates for buyer simulation and judging."""

from __future__ import annotations

from src.state import SimulationState


BUYER_SYSTEM_PROMPT = (
    "You are a realistic e-commerce buyer. React honestly to the product, "
    "including what feels compelling, unclear, or risky."
)

JUDGE_SYSTEM_PROMPT = (
    "You evaluate buyer feedback for product-market fit signals, purchase "
    "intent, objections, and next experiment ideas."
)


def build_buyer_prompt(state: SimulationState) -> str:
    """Build a buyer-facing prompt from the current simulation state."""
    product = state["product"]
    persona = state["persona"]
    return (
        f"Persona: {persona['name']} ({persona['segment']})\n"
        f"Motivation: {persona['motivation']}\n"
        f"Objection: {persona['objection']}\n\n"
        f"Product: {product['name']}\n"
        f"Category: {product['category']}\n"
        f"Price: ${product['price']:.2f}\n"
        f"Description: {product['description']}\n"
        f"Target audience: {product['target_audience']}\n\n"
        "Respond in 3 concise bullets."
    )


def build_judge_prompt(state: SimulationState) -> str:
    """Build an evaluator prompt from buyer feedback."""
    return (
        f"Buyer response:\n{state.get('buyer_response', 'No buyer response yet.')}\n\n"
        "Return a concise summary and a purchase intent score from 1 to 10."
    )
