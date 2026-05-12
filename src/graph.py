"""Core simulation orchestration for BuyerLab AI."""

from __future__ import annotations

from typing import Any

from src.agents import run_debate_round, run_initial_buyer_round
from src.judge import run_judge_report
from src.state import ProductInput, SimulationState, create_empty_state


def run_simulation(product: ProductInput) -> SimulationState:
    """Run the complete buyer simulation engine for one product input."""
    state = create_empty_state(product)
    state.first_round_responses = run_initial_buyer_round(
        product=state.product,
        personas=state.personas,
    )
    state.debate_history = run_debate_round(
        product=state.product,
        personas=state.personas,
        first_round_responses=state.first_round_responses,
    )
    state.final_report = run_judge_report(state)
    state.before_score = state.final_report.simulated_conversion_score
    return state


def run_sample_simulation(product: ProductInput | dict[str, Any]) -> SimulationState:
    """Backward-compatible wrapper for early local smoke tests."""
    return run_simulation(_coerce_product_input(product))


def _coerce_product_input(product: ProductInput | dict[str, Any]) -> ProductInput:
    """Convert old sample product dictionaries into ProductInput objects."""
    if isinstance(product, ProductInput):
        return product

    return ProductInput(
        title=str(product.get("title", product.get("name", ""))),
        category=str(product.get("category", "")),
        price=float(product.get("price", 0.0)),
        currency=str(product.get("currency", "USD")),
        description=str(product.get("description", "")),
        target_audience=str(product.get("target_audience", "")),
        value_proposition=str(product.get("value_proposition", "")),
        warranty_or_return_policy=str(product.get("warranty_or_return_policy", "")),
        shipping_info=str(product.get("shipping_info", "")),
        trust_signals=_coerce_string_list(product.get("trust_signals", [])),
        reviews_or_social_proof=str(product.get("reviews_or_social_proof", "")),
        call_to_action=str(product.get("call_to_action", "")),
        image_notes=product.get("image_notes"),
    )


def _coerce_string_list(value: Any) -> list[str]:
    """Normalize optional list-like sample product fields."""
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []
