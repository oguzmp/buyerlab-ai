"""Placeholder judge logic for interpreting buyer responses."""

from __future__ import annotations

from src.state import SimulationState


def judge_buyer_response(state: SimulationState) -> SimulationState:
    """Add a simple deterministic judgment to the simulation state."""
    buyer_response = state.get("buyer_response", "")
    has_objection = any(word in buyer_response.lower() for word in ["proof", "review", "unclear"])

    state["purchase_intent_score"] = 6 if has_objection else 7
    state["judge_summary"] = (
        "Early signal: useful positioning, with trust and urgency as the first "
        "objections to test."
    )
    return state
