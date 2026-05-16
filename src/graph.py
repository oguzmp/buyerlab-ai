"""Core simulation orchestration for BuyerLab AI."""

from __future__ import annotations

from typing import Any

from src.agents import run_debate_round, run_initial_buyer_round
from src.judge import run_judge_report
from src.price_intelligence import apply_category_persona_weights
from src.state import CompetitorContext, ProductInput, SimulationState, create_empty_state


def run_simulation(product: ProductInput) -> SimulationState:
    """Run the complete buyer simulation engine for one product input."""
    state = create_empty_state(product)
    state.personas = apply_category_persona_weights(state.personas, product.category)
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
        competitor_context=_coerce_competitor_context(product.get("competitor_context")),
    )


def _coerce_string_list(value: Any) -> list[str]:
    """Normalize optional list-like sample product fields."""
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _coerce_competitor_context(value: Any) -> CompetitorContext | None:
    """Normalize optional competitor context dictionaries from sample data."""
    if isinstance(value, CompetitorContext):
        return value
    if not isinstance(value, dict):
        return None

    return CompetitorContext(
        competitor_name=str(value.get("competitor_name", "")),
        competitor_price=_coerce_optional_float(value.get("competitor_price")),
        competitor_strengths=_coerce_string_list(value.get("competitor_strengths", [])),
        competitor_weaknesses=_coerce_string_list(value.get("competitor_weaknesses", [])),
        our_differentiator=str(value.get("our_differentiator", "")),
    )


def _coerce_optional_float(value: Any) -> float | None:
    """Convert optional numeric fields without turning missing values into zero."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
