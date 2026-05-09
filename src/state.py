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
    """Temporary persona shape used by the placeholder simulation."""

    name: str
    segment: str
    motivation: str
    objection: str


class SimulationState(TypedDict):
    """State passed between BuyerLab AI simulation graph nodes."""

    product_title: str
    product_price: float
    product_description: str
    product_category: str
    target_audience: str
    debate_history: list[str]
    conversion_score: float
    final_report: str

    product: NotRequired[Product]
    persona: NotRequired[BuyerPersona]
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
        "product_title": product["name"],
        "product_price": product["price"],
        "product_description": product["description"],
        "product_category": product["category"],
        "target_audience": product["target_audience"],
        "debate_history": [],
        "conversion_score": 0.0,
        "final_report": "",
        "product": product,
        "persona": persona or DEFAULT_PERSONA,
    }
