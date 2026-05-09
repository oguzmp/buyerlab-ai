"""Shared state types for BuyerLab AI simulations."""

from __future__ import annotations

from typing import NotRequired, TypedDict


class Product(TypedDict):
    id: str
    name: str
    category: str
    price: float
    description: str
    target_audience: str


class BuyerPersona(TypedDict):
    name: str
    segment: str
    motivation: str
    objection: str


class SimulationState(TypedDict):
    product: Product
    persona: BuyerPersona
    buyer_response: NotRequired[str]
    judge_summary: NotRequired[str]
    purchase_intent_score: NotRequired[int]


DEFAULT_PERSONA: BuyerPersona = {
    "name": "Practical Priya",
    "segment": "Value-conscious online shopper",
    "motivation": "Finds products that solve a real daily problem.",
    "objection": "Needs clear proof before trying a new brand.",
}


def create_initial_state(product: Product, persona: BuyerPersona | None = None) -> SimulationState:
    """Create the initial simulation state for one product and buyer persona."""
    return {
        "product": product,
        "persona": persona or DEFAULT_PERSONA,
    }
