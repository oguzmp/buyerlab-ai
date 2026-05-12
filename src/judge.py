"""Judge agent for BuyerLab AI simulation reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from statistics import mean
from typing import Any

from src.gemini_client import generate_json
from src.prompts import JUDGE_PROMPT
from src.state import AgentResponse, SimulationReport, SimulationState


OBJECTION_CATEGORIES = {
    "trust_gap": [
        "trust",
        "credibility",
        "secure",
        "proof",
        "guarantee",
        "legit",
        "professional",
    ],
    "price_resistance": [
        "price",
        "cost",
        "expensive",
        "discount",
        "value",
        "worth",
    ],
    "missing_product_information": [
        "missing",
        "detail",
        "technical",
        "spec",
        "ingredient",
        "material",
        "size",
        "unclear",
    ],
    "weak_emotional_appeal": [
        "boring",
        "desire",
        "emotion",
        "exciting",
        "urgency",
        "fomo",
        "visual",
    ],
    "unclear_value_proposition": [
        "benefit",
        "proposition",
        "why",
        "differentiation",
        "useful",
        "problem",
    ],
    "shipping_or_return_concern": [
        "shipping",
        "return",
        "refund",
        "warranty",
        "delivery",
        "policy",
    ],
    "social_proof_gap": [
        "review",
        "testimonial",
        "rating",
        "social proof",
        "customer",
        "case study",
    ],
    "other": [],
}


def run_judge_report(state: SimulationState) -> SimulationReport:
    """Generate the final judge report, with a deterministic fallback."""
    prompt = _build_judge_prompt(state)

    try:
        raw_report = generate_json(prompt)
        return _simulation_report_from_json(raw_report, state)
    except Exception as exc:
        return _fallback_report(state, error=exc)


def analyze_buyer_losses(agent_responses: list[AgentResponse]) -> list[dict[str, Any]]:
    """Build dashboard-ready buyer loss analysis for each persona response."""
    return [_buyer_loss_row(response) for response in agent_responses]


def cluster_objections(agent_responses: list[AgentResponse]) -> dict[str, list[str]]:
    """Group buyer objections into business-readable loss categories."""
    clusters: dict[str, list[str]] = {category: [] for category in OBJECTION_CATEGORIES}

    for response in agent_responses:
        for objection in _response_objection_texts(response):
            category = _categorize_objection(objection)
            _append_unique(clusters[category], _short_text(objection))

    return clusters


def calculate_risk_scores(agent_responses: list[AgentResponse]) -> dict[str, int]:
    """Calculate simulated report scores from buyer response patterns."""
    clusters = cluster_objections(agent_responses)
    trust_risk = _category_risk(clusters, ["trust_gap", "social_proof_gap"])
    price_risk = _category_risk(clusters, ["price_resistance"])
    return_risk = _category_risk(clusters, ["shipping_or_return_concern"])
    clarity_penalty = _category_risk(
        clusters,
        ["missing_product_information", "unclear_value_proposition"],
    )

    return {
        "trust_risk_score": trust_risk,
        "price_resistance_score": price_risk,
        "clarity_score": max(0, 100 - clarity_penalty),
        "return_risk_score": return_risk,
    }


def prioritize_action_items(
    agent_responses: list[AgentResponse],
    clustered_objections: dict[str, list[str]],
) -> list[str]:
    """Order action items by expected simulated business impact."""
    actions: list[str] = []

    if (
        clustered_objections.get("trust_gap")
        or clustered_objections.get("social_proof_gap")
        or clustered_objections.get("shipping_or_return_concern")
    ):
        actions.append("Add stronger proof, reviews, guarantees, shipping, and return signals.")

    if clustered_objections.get("missing_product_information"):
        actions.append("Clarify critical product details, specs, claims, and usage information.")

    if (
        clustered_objections.get("price_resistance")
        or clustered_objections.get("unclear_value_proposition")
    ):
        actions.append("Explain value for money with sharper benefits and price justification.")

    if clustered_objections.get("weak_emotional_appeal"):
        actions.append("Improve product copy with stronger desire, urgency, and visual appeal.")

    for response in agent_responses:
        if _safe_decision(_response_value(response, "decision")) == "buy":
            continue
        suggested_fix = _safe_text(_response_value(response, "suggested_fix"))
        if suggested_fix:
            _append_unique(actions, _short_text(suggested_fix))

    return actions[:5] or ["No urgent action identified from simulated buyer feedback."]


def build_enhanced_judge_context(state: SimulationState) -> dict[str, Any]:
    """Build structured judge context for Gemini and deterministic fallback reports."""
    persona_names = {persona.id: persona.name for persona in state.personas}
    buyer_losses = analyze_buyer_losses(state.first_round_responses)
    for loss in buyer_losses:
        loss["persona_name"] = persona_names.get(
            loss["persona_id"],
            loss["persona_name"],
        )

    clustered_objections = cluster_objections(state.first_round_responses)
    risk_scores = calculate_risk_scores(state.first_round_responses)

    return {
        "important_note": (
            "Scores are AI-simulated testing signals, not real market predictions."
        ),
        "product": {
            "title": state.product.title,
            "category": state.product.category,
            "price": state.product.price,
            "currency": state.product.currency,
            "target_audience": state.product.target_audience,
            "value_proposition": state.product.value_proposition,
        },
        "first_round_responses": [
            _response_as_dict(response) for response in state.first_round_responses
        ],
        "debate_history": [asdict(turn) for turn in state.debate_history],
        "buyer_loss_analysis": buyer_losses,
        "clustered_objections": clustered_objections,
        "risk_scores": risk_scores,
        "prioritized_action_items": prioritize_action_items(
            state.first_round_responses,
            clustered_objections,
        ),
    }


def _build_judge_prompt(state: SimulationState) -> str:
    """Build a compact judge prompt from responses and debate history."""
    judge_context = build_enhanced_judge_context(state)
    context_json = json.dumps(judge_context, ensure_ascii=True, indent=2)

    return f"""
{JUDGE_PROMPT}

Enhanced judge context:
{context_json}

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

Rules:
- Use the phrase "simulated conversion score" in the summary.
- Do not claim the score is a real market prediction.
- Keep all arrays to 5 short dashboard-ready items or fewer.
- Order top_action_items by business impact: trust blockers, missing critical
  product information, price/value objections, then emotional/copy improvements.
""".strip()


def _simulation_report_from_json(
    raw_report: dict[str, Any],
    state: SimulationState,
) -> SimulationReport:
    """Convert Gemini JSON into a SimulationReport with safe defaults."""
    enhanced_context = build_enhanced_judge_context(state)
    risk_scores = enhanced_context["risk_scores"]
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
        trust_risk_score=_safe_score(
            raw_report.get("trust_risk_score"),
            default=risk_scores["trust_risk_score"],
        ),
        price_resistance_score=_safe_score(
            raw_report.get("price_resistance_score"),
            default=risk_scores["price_resistance_score"],
        ),
        clarity_score=_safe_score(
            raw_report.get("clarity_score"),
            default=risk_scores["clarity_score"],
        ),
        return_risk_score=_safe_score(
            raw_report.get("return_risk_score"),
            default=risk_scores["return_risk_score"],
        ),
        top_action_items=_short_list(
            raw_report.get("top_action_items"),
            raw_report.get("optimization_action_plan"),
            default=enhanced_context["prioritized_action_items"],
        ),
        summary=_simulated_summary(
            _first_text(
                raw_report.get("summary"),
                raw_report.get("dashboard_summary"),
                "Simulation completed with mixed buyer intent.",
            )
        ),
    )


def _fallback_report(state: SimulationState, error: Exception | None = None) -> SimulationReport:
    """Create a final report when the judge agent fails or returns invalid JSON."""
    responses = state.first_round_responses
    enhanced_context = build_enhanced_judge_context(state)
    risk_scores = enhanced_context["risk_scores"]
    conversion_score = _estimate_conversion_score(responses)
    buyer_loss_reasons = _fallback_loss_reasons(responses)

    if error is not None:
        buyer_loss_reasons = [*buyer_loss_reasons, f"Judge fallback used: {_short_error(error)}"]

    return SimulationReport(
        simulated_conversion_score=conversion_score,
        buyer_loss_reasons=buyer_loss_reasons[:5],
        winning_personas=_personas_by_decision(responses, "buy"),
        lost_personas=_personas_by_decision(responses, "reject"),
        trust_risk_score=risk_scores["trust_risk_score"],
        price_resistance_score=risk_scores["price_resistance_score"],
        clarity_score=risk_scores["clarity_score"],
        return_risk_score=risk_scores["return_risk_score"],
        top_action_items=enhanced_context["prioritized_action_items"],
        summary="Simulated conversion score uses fallback buyer-loss analysis.",
    )


def _buyer_loss_row(response: AgentResponse) -> dict[str, Any]:
    """Create one safe buyer-loss row from an agent response."""
    persona_id = _safe_text(_response_value(response, "persona_id")) or "unknown_persona"
    decision = _safe_decision(_response_value(response, "decision"))
    purchase_intent = _safe_score(_response_value(response, "purchase_intent"))
    objections = _short_list(_response_value(response, "objections"), limit=3)
    missing_information = _short_list(
        _response_value(response, "missing_information"),
        limit=3,
    )

    return {
        "persona_id": persona_id,
        "persona_name": _persona_name_from_id(persona_id),
        "final_decision": decision,
        "purchase_intent": purchase_intent,
        "main_loss_reason": _main_loss_reason(response),
        "objections": objections,
        "missing_information": missing_information,
        "suggested_fix": _safe_text(_response_value(response, "suggested_fix"))
        or "No fix suggested.",
        "business_impact": _business_impact(decision, purchase_intent, objections),
    }


def _response_objection_texts(response: AgentResponse) -> list[str]:
    """Collect objection-like fields from a response for clustering."""
    texts = [
        *_short_list(_response_value(response, "objections"), limit=5),
        *_short_list(_response_value(response, "missing_information"), limit=5),
    ]
    decision = _safe_decision(_response_value(response, "decision"))
    main_reason = _safe_text(_response_value(response, "main_reason"))
    if decision != "buy" and main_reason:
        texts.append(main_reason)
    return texts


def _categorize_objection(objection: str) -> str:
    """Assign one objection to the first matching business category."""
    text = objection.lower()
    for category, keywords in OBJECTION_CATEGORIES.items():
        if category == "other":
            continue
        if any(keyword in text for keyword in keywords):
            return category
    return "other"


def _category_risk(clusters: dict[str, list[str]], categories: list[str]) -> int:
    """Convert clustered objection volume into a compact 0-100 risk score."""
    issue_count = sum(len(clusters.get(category, [])) for category in categories)
    return min(100, issue_count * 25)


def _estimate_conversion_score(responses: list[AgentResponse]) -> int:
    """Estimate conversion score from persona purchase intent and weights."""
    if not responses:
        return 0
    intents = [
        _safe_score(_response_value(response, "purchase_intent"))
        for response in responses
    ]
    return _safe_score(round(mean(intents)))


def _fallback_loss_reasons(responses: list[AgentResponse]) -> list[str]:
    """Summarize concise loss reasons from rejected and hesitant personas."""
    reasons = [
        loss["main_loss_reason"]
        for loss in analyze_buyer_losses(responses)
        if loss["final_decision"] != "buy" and loss["main_loss_reason"]
    ]
    return reasons[:5] or ["No major loss reason identified."]


def _personas_by_decision(responses: list[AgentResponse], decision: str) -> list[str]:
    """Return persona ids that made a specific purchase decision."""
    return [
        _safe_text(_response_value(response, "persona_id")) or "unknown_persona"
        for response in responses
        if _safe_decision(_response_value(response, "decision")) == decision
    ]


def _short_list(*values: Any, default: list[str] | None = None, limit: int = 5) -> list[str]:
    """Return a compact list from JSON values or a default."""
    items: list[str] = []
    for value in values:
        if isinstance(value, list):
            items.extend(_short_text(str(item)) for item in value if str(item).strip())
        elif isinstance(value, str) and value.strip():
            items.append(_short_text(value))

    if not items and default is not None:
        items = default

    return items[:limit]


def _main_loss_reason(response: AgentResponse) -> str:
    """Return a short reason focused on why a buyer was lost or hesitant."""
    decision = _safe_decision(_response_value(response, "decision"))
    main_reason = _safe_text(_response_value(response, "main_reason"))
    if decision == "buy":
        return main_reason or "Persona is likely to buy."
    return main_reason or "Persona needs stronger purchase confidence."


def _business_impact(decision: str, purchase_intent: int, objections: list[str]) -> str:
    """Estimate expected business impact for a persona's blocker."""
    if decision == "reject":
        return "high"
    if decision == "hesitate" and purchase_intent >= 50:
        return "high"
    if decision == "hesitate" or objections:
        return "medium"
    return "low"


def _persona_name_from_id(persona_id: str) -> str:
    """Create a readable fallback persona name from an id."""
    if not persona_id or persona_id == "unknown_persona":
        return "Unknown Persona"
    return persona_id.replace("_", " ").title()


def _first_text(*values: Any) -> str:
    """Return the first non-empty string value."""
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "No summary provided."


def _simulated_summary(summary: str) -> str:
    """Ensure report summaries describe the score as simulated."""
    if "simulated conversion score" in summary.lower():
        return _short_text(summary, limit=220)
    return _short_text(f"Simulated conversion score: {summary}", limit=220)


def _safe_decision(value: Any) -> str:
    """Normalize a decision label without trusting malformed data."""
    if isinstance(value, str):
        decision = value.strip().lower()
        if decision in {"buy", "reject", "hesitate"}:
            return decision
        if "reject" in decision:
            return "reject"
        if "buy" in decision:
            return "buy"
    return "hesitate"


def _safe_score(value: Any, default: int = 0) -> int:
    """Normalize a score into an integer from 0 to 100."""
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = default
    return max(0, min(100, score))


def _response_value(response: Any, field_name: str) -> Any:
    """Safely read a field from dataclass or dict response objects."""
    if isinstance(response, dict):
        return response.get(field_name)
    return getattr(response, field_name, None)


def _response_as_dict(response: Any) -> dict[str, Any]:
    """Serialize response-like objects without trusting their shape."""
    if isinstance(response, dict):
        return dict(response)
    try:
        return asdict(response)
    except TypeError:
        return {
            "persona_id": _response_value(response, "persona_id"),
            "decision": _response_value(response, "decision"),
            "purchase_intent": _response_value(response, "purchase_intent"),
            "main_reason": _response_value(response, "main_reason"),
        }


def _safe_text(value: Any) -> str:
    """Normalize optional text into a short string."""
    if value is None:
        return ""
    return _short_text(str(value).strip())


def _short_text(value: str, limit: int = 140) -> str:
    """Keep dashboard text compact and single-line."""
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _append_unique(items: list[str], value: str) -> None:
    """Append a value once while preserving order."""
    if value and value not in items:
        items.append(value)


def _short_error(exc: Exception) -> str:
    """Format an exception as a short dashboard-safe message."""
    return str(exc).splitlines()[0][:120] or exc.__class__.__name__
