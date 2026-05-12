"""Judge agent for BuyerLab AI simulation reports."""

from __future__ import annotations

from dataclasses import asdict
from statistics import mean
from typing import Any

from src.gemini_client import generate_json
from src.prompts import JUDGE_PROMPT
from src.state import AgentResponse, SimulationReport, SimulationState


def run_judge_report(state: SimulationState) -> SimulationReport:
    """Generate the final judge report, with a deterministic fallback."""
    prompt = _build_judge_prompt(state)

    try:
        raw_report = generate_json(prompt)
        return _simulation_report_from_json(raw_report, state)
    except Exception as exc:
        return _fallback_report(state, error=exc)


def _build_judge_prompt(state: SimulationState) -> str:
    """Build a compact judge prompt from responses and debate history."""
    first_round = [asdict(response) for response in state.first_round_responses]
    debate = [asdict(turn) for turn in state.debate_history]

    return f"""
{JUDGE_PROMPT}

Product:
- title: {state.product.title}
- category: {state.product.category}
- price: {state.product.price} {state.product.currency}
- target_audience: {state.product.target_audience}
- value_proposition: {state.product.value_proposition}

First-round buyer responses:
{first_round}

Debate history:
{debate}

Return only valid JSON for this exact report schema:
{{
  "simulated_conversion_score": 0,
  "buyer_loss_reasons": [],
  "winning_personas": [],
  "lost_personas": [],
  "trust_risk_score": 0,
  "price_resistance_score": 0,
  "clarity_score": 0,
  "return_risk_score": 0,
  "top_action_items": [],
  "summary": ""
}}

Keep all arrays to 5 short dashboard-ready items or fewer.
""".strip()


def _simulation_report_from_json(
    raw_report: dict[str, Any],
    state: SimulationState,
) -> SimulationReport:
    """Convert Gemini JSON into a SimulationReport with safe defaults."""
    fallback_score = _estimate_conversion_score(state.first_round_responses)
    conversion_score = _safe_score(
        raw_report.get("simulated_conversion_score", raw_report.get("conversion_score")),
        default=fallback_score,
    )

    return SimulationReport(
        simulated_conversion_score=conversion_score,
        buyer_loss_reasons=_short_list(
            raw_report.get("buyer_loss_reasons"),
            raw_report.get("lost_customer_reasons"),
            default=_fallback_loss_reasons(state.first_round_responses),
        ),
        winning_personas=_short_list(
            raw_report.get("winning_personas"),
            default=_personas_by_decision(state.first_round_responses, "buy"),
        ),
        lost_personas=_short_list(
            raw_report.get("lost_personas"),
            default=_personas_by_decision(state.first_round_responses, "reject"),
        ),
        trust_risk_score=_safe_score(raw_report.get("trust_risk_score"), default=50),
        price_resistance_score=_safe_score(raw_report.get("price_resistance_score"), default=50),
        clarity_score=_safe_score(raw_report.get("clarity_score"), default=50),
        return_risk_score=_safe_score(raw_report.get("return_risk_score"), default=50),
        top_action_items=_short_list(
            raw_report.get("top_action_items"),
            raw_report.get("optimization_action_plan"),
            default=["Clarify value, proof, shipping, and return details."],
        ),
        summary=_first_text(
            raw_report.get("summary"),
            raw_report.get("dashboard_summary"),
            "Simulation completed with mixed buyer intent.",
        ),
    )


def _fallback_report(state: SimulationState, error: Exception | None = None) -> SimulationReport:
    """Create a final report when the judge agent fails or returns invalid JSON."""
    responses = state.first_round_responses
    conversion_score = _estimate_conversion_score(responses)
    buyer_loss_reasons = _fallback_loss_reasons(responses)

    if error is not None:
        buyer_loss_reasons = [*buyer_loss_reasons, f"Judge fallback used: {_short_error(error)}"]

    return SimulationReport(
        simulated_conversion_score=conversion_score,
        buyer_loss_reasons=buyer_loss_reasons[:5],
        winning_personas=_personas_by_decision(responses, "buy"),
        lost_personas=_personas_by_decision(responses, "reject"),
        trust_risk_score=_risk_score_from_keywords(responses, ["trust", "review", "proof"]),
        price_resistance_score=_risk_score_from_keywords(responses, ["price", "value", "cost"]),
        clarity_score=_estimate_clarity_score(responses),
        return_risk_score=_risk_score_from_keywords(responses, ["return", "warranty", "refund"]),
        top_action_items=_fallback_action_items(responses),
        summary="Buyer simulation completed with fallback scoring.",
    )


def _estimate_conversion_score(responses: list[AgentResponse]) -> int:
    """Estimate conversion score from persona purchase intent and weights."""
    if not responses:
        return 0
    return _safe_score(round(mean(response.purchase_intent for response in responses)))


def _fallback_loss_reasons(responses: list[AgentResponse]) -> list[str]:
    """Summarize concise loss reasons from rejected and hesitant personas."""
    reasons = [
        response.main_reason
        for response in responses
        if response.decision != "buy" and response.main_reason
    ]
    return reasons[:5] or ["No major loss reason identified."]


def _fallback_action_items(responses: list[AgentResponse]) -> list[str]:
    """Collect suggested fixes from buyer responses."""
    actions = [response.suggested_fix for response in responses if response.suggested_fix]
    return actions[:5] or ["Add clearer product proof and purchase risk reducers."]


def _personas_by_decision(responses: list[AgentResponse], decision: str) -> list[str]:
    """Return persona ids that made a specific purchase decision."""
    return [response.persona_id for response in responses if response.decision == decision]


def _risk_score_from_keywords(responses: list[AgentResponse], keywords: list[str]) -> int:
    """Estimate a risk score from objections and missing information."""
    response_texts = [
        " ".join(
            [
                response.main_reason,
                *response.objections,
                *response.missing_information,
            ]
        )
        for response in responses
    ]
    text = " ".join(response_texts).lower()
    matches = sum(1 for keyword in keywords if keyword in text)
    return min(100, matches * 30)


def _estimate_clarity_score(responses: list[AgentResponse]) -> int:
    """Estimate clarity from the volume of objections and missing details."""
    issue_count = sum(
        len(response.objections) + len(response.missing_information)
        for response in responses
    )
    return max(0, 100 - issue_count * 10)


def _short_list(*values: Any, default: list[str] | None = None, limit: int = 5) -> list[str]:
    """Return a compact list from JSON values or a default."""
    items: list[str] = []
    for value in values:
        if isinstance(value, list):
            items.extend(str(item).strip() for item in value if str(item).strip())
        elif isinstance(value, str) and value.strip():
            items.append(value.strip())

    if not items and default is not None:
        items = default

    return items[:limit]


def _first_text(*values: Any) -> str:
    """Return the first non-empty string value."""
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "No summary provided."


def _safe_score(value: Any, default: int = 0) -> int:
    """Normalize a score into an integer from 0 to 100."""
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = default
    return max(0, min(100, score))


def _short_error(exc: Exception) -> str:
    """Format an exception as a short dashboard-safe message."""
    return str(exc).splitlines()[0][:120] or exc.__class__.__name__
