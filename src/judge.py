"""Judge agent for BuyerLab AI simulation reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from statistics import mean
from typing import Any

from src.gemini_client import generate_json
from src.price_intelligence import (
    analyze_competitor_gap,
    analyze_competitor_context,
    analyze_local_price_perception,
    build_structured_product_brief,
)
from src.launch_readiness import (
    build_launch_readiness_report,
    build_launch_readiness_summary as _build_launch_readiness_summary,
)
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
    price_perception_report = analyze_local_price_perception(state.product)
    competitor_gap_report = analyze_competitor_gap(state.product)
    competitor_analysis = analyze_competitor_context(state.product, price_perception_report)
    risk_scores["price_resistance_score"] = max(
        risk_scores["price_resistance_score"],
        price_perception_report.perceived_value_risk,
    )
    price_perception = asdict(price_perception_report)
    structured_product_brief = build_structured_product_brief(state.product)
    launch_readiness_report = build_launch_readiness_report(
        state=state,
        risk_scores=risk_scores,
        price_report=price_perception_report,
        competitor_gap=competitor_gap_report,
        buyer_loss_analysis=buyer_losses,
        clustered_objections=clustered_objections,
    )
    launch_readiness = asdict(launch_readiness_report)

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
        "structured_product_brief": structured_product_brief,
        "price_perception": price_perception,
        "competitor_gap": asdict(competitor_gap_report),
        "competitor_analysis": competitor_analysis,
        "launch_readiness": launch_readiness,
        "prioritized_action_items": prioritize_action_items(
            state.first_round_responses,
            clustered_objections,
        ),
    }


def build_launch_readiness_summary(report_inputs: dict[str, Any]) -> dict[str, Any]:
    """Build a 10-second launch readiness summary from simulated diagnostics."""
    return _build_launch_readiness_summary(report_inputs)


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
  "price_positioning_verdict": "",
  "competitor_gap_summary": "",
  "required_price_proofs": [],
  "launch_readiness_score": 0,
  "launch_status": "ready | needs_fixes | not_ready",
  "executive_verdict": "",
  "main_blocker": "",
  "category_expectation_check": [
    {{
      "field_name": "",
      "status": "present | missing | weak",
      "impact": "high | medium | low",
      "explanation": "",
      "suggested_fix": ""
    }}
  ],
  "local_price_perception_summary": "",
  "buyer_persona_verdicts": [],
  "buyer_loss_summary": "",
  "top_conversion_blockers": [],
  "required_fix_before_launch": [],
  "next_best_actions": [],
  "price_justification_verdict": "",
  "competitor_gap_verdict": "",
  "launch_decision_summary": "",
  "summary": ""
}}

Rules:
- Write user-facing report text in Turkish for Turkish e-commerce sellers.
- Use the phrase "simulated conversion score" in the summary.
- Do not claim the score is a real market prediction.
- Keep all arrays to 5 short dashboard-ready items or fewer.
- Order top_action_items by business impact: trust blockers, missing critical
  product information, price/value objections, then emotional/copy improvements.
- Include price_positioning_verdict, competitor_gap_summary, and
  required_price_proofs from the heuristic price and competitor context.
- Include launch readiness fields. Use launch_status values only:
  ready, needs_fixes, or not_ready.
- Treat launch readiness as an AI-assisted diagnostic, not a real market prediction.
- Make required_fix_before_launch concrete, such as exact specs, warranty,
  return policy, proof assets, or competitor comparison fixes.
""".strip()


def _simulation_report_from_json(
    raw_report: dict[str, Any],
    state: SimulationState,
) -> SimulationReport:
    """Convert Gemini JSON into a SimulationReport with safe defaults."""
    enhanced_context = build_enhanced_judge_context(state)
    risk_scores = enhanced_context["risk_scores"]
    competitor_analysis = enhanced_context["competitor_analysis"]
    launch_readiness = enhanced_context["launch_readiness"]
    fallback_score = _estimate_conversion_score(state.first_round_responses)
    conversion_score = _safe_score(
        raw_report.get("simulated_conversion_score", raw_report.get("conversion_score")),
        default=fallback_score,
    )

    return SimulationReport(
        simulated_conversion_score=conversion_score,
        launch_readiness_score=launch_readiness["launch_readiness_score"],
        launch_status=launch_readiness["launch_status"],
        executive_verdict=_first_text(
            raw_report.get("executive_verdict"),
            launch_readiness["executive_verdict"],
        ),
        main_blocker=_first_text(
            raw_report.get("main_blocker"),
            launch_readiness["main_blocker"],
        ),
        category_expectation_check=_safe_dict_list(
            raw_report.get("category_expectation_check"),
            default=launch_readiness["category_expectation_check"],
        ),
        local_price_perception_summary=_first_text(
            raw_report.get("local_price_perception_summary"),
            launch_readiness["local_price_perception_summary"],
        ),
        competitor_gap_summary=_first_text(
            raw_report.get("competitor_gap_summary"),
            launch_readiness["competitor_gap_summary"],
        ),
        buyer_persona_verdicts=_safe_dict_list(
            raw_report.get("buyer_persona_verdicts"),
            default=launch_readiness["buyer_persona_verdicts"],
        ),
        buyer_loss_summary=launch_readiness["buyer_loss_summary"],
        top_conversion_blockers=_short_list(
            raw_report.get("top_conversion_blockers"),
            default=launch_readiness["top_conversion_blockers"],
        ),
        required_fix_before_launch=_short_list(
            raw_report.get("required_fix_before_launch"),
            default=launch_readiness["required_fix_before_launch"],
        ),
        next_best_actions=_short_list(
            raw_report.get("next_best_actions"),
            default=launch_readiness["next_best_actions"],
        ),
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
        price_positioning_verdict=_first_text(
            raw_report.get("price_positioning_verdict"),
            competitor_analysis["price_positioning_verdict"],
        ),
        price_justification_verdict=launch_readiness["price_justification_verdict"],
        competitor_gap_verdict=launch_readiness["competitor_gap_verdict"],
        launch_decision_summary=launch_readiness["launch_decision_summary"],
        required_price_proofs=_short_list(
            raw_report.get("required_price_proofs"),
            default=competitor_analysis["required_price_proofs"],
        ),
        summary=_simulated_summary(
            _first_text(
                raw_report.get("summary"),
                raw_report.get("dashboard_summary"),
                launch_readiness["summary"],
            )
        ),
    )


def _fallback_report(state: SimulationState, error: Exception | None = None) -> SimulationReport:
    """Create a final report when the judge agent fails or returns invalid JSON."""
    responses = state.first_round_responses
    enhanced_context = build_enhanced_judge_context(state)
    risk_scores = enhanced_context["risk_scores"]
    competitor_analysis = enhanced_context["competitor_analysis"]
    launch_readiness = enhanced_context["launch_readiness"]
    conversion_score = _estimate_conversion_score(responses)
    buyer_loss_reasons = _fallback_loss_reasons(responses)

    if error is not None:
        buyer_loss_reasons = [*buyer_loss_reasons, f"Judge fallback used: {_short_error(error)}"]

    return SimulationReport(
        simulated_conversion_score=conversion_score,
        launch_readiness_score=launch_readiness["launch_readiness_score"],
        launch_status=launch_readiness["launch_status"],
        executive_verdict=launch_readiness["executive_verdict"],
        main_blocker=launch_readiness["main_blocker"],
        category_expectation_check=launch_readiness["category_expectation_check"],
        local_price_perception_summary=launch_readiness["local_price_perception_summary"],
        competitor_gap_summary=launch_readiness["competitor_gap_summary"],
        buyer_persona_verdicts=launch_readiness["buyer_persona_verdicts"],
        buyer_loss_summary=launch_readiness["buyer_loss_summary"],
        top_conversion_blockers=launch_readiness["top_conversion_blockers"],
        required_fix_before_launch=launch_readiness["required_fix_before_launch"],
        next_best_actions=launch_readiness["next_best_actions"],
        buyer_loss_reasons=buyer_loss_reasons[:5],
        winning_personas=_personas_by_decision(responses, "buy"),
        lost_personas=_personas_by_decision(responses, "reject"),
        trust_risk_score=risk_scores["trust_risk_score"],
        price_resistance_score=risk_scores["price_resistance_score"],
        clarity_score=risk_scores["clarity_score"],
        return_risk_score=risk_scores["return_risk_score"],
        top_action_items=enhanced_context["prioritized_action_items"],
        price_positioning_verdict=competitor_analysis["price_positioning_verdict"],
        price_justification_verdict=launch_readiness["price_justification_verdict"],
        competitor_gap_verdict=launch_readiness["competitor_gap_verdict"],
        launch_decision_summary=launch_readiness["launch_decision_summary"],
        required_price_proofs=competitor_analysis["required_price_proofs"],
        summary=launch_readiness["summary"],
    )


def _buyer_persona_verdicts(responses: list[AgentResponse]) -> list[dict[str, Any]]:
    """Return compact persona verdicts for launch readiness cards."""
    return [
        {
            "persona_id": _safe_text(_response_value(response, "persona_id")),
            "decision": _safe_decision(_response_value(response, "decision")),
            "purchase_intent": _safe_score(_response_value(response, "purchase_intent")),
            "reason": _safe_text(_response_value(response, "main_reason")),
        }
        for response in responses
    ]


def _launch_status(
    launch_score: int,
    missing_count: int,
    risk_scores: dict[str, int],
) -> str:
    """Map launch readiness signals into a simple launch status."""
    if launch_score >= 75 and missing_count <= 1 and risk_scores["trust_risk_score"] < 50:
        return "ready"
    if launch_score < 45 or risk_scores["trust_risk_score"] >= 75:
        return "not_ready"
    return "needs_fixes"


def _top_conversion_blockers(
    clustered_objections: dict[str, list[str]],
    category_check: list[dict[str, str]],
    price_perception: dict[str, Any],
    competitor_gap: dict[str, Any],
) -> list[str]:
    """Collect the clearest blockers for a 10-second launch verdict."""
    blockers: list[str] = []
    missing_fields = [
        item["field"] for item in category_check if item.get("status") == "missing"
    ]
    if missing_fields:
        blockers.append(f"Missing category-required information: {', '.join(missing_fields[:3])}.")
    if clustered_objections.get("trust_gap") or clustered_objections.get("social_proof_gap"):
        blockers.append("Trust proof is weak for launch.")
    if price_perception.get("perceived_value_risk", 0) >= 65:
        blockers.append("Price justification is weak for the detected local price band.")
    if competitor_gap.get("our_unproven_claims"):
        blockers.append("Competitor differentiator is not backed by proof assets.")
    if clustered_objections.get("shipping_or_return_concern"):
        blockers.append("Shipping, warranty, or return terms need clearer risk reduction.")
    return blockers[:5] or ["No major launch blocker identified in the simulated buyer assessment."]


def _required_launch_fixes(
    blockers: list[str],
    competitor_gap: dict[str, Any],
) -> list[str]:
    """Turn blockers into required fixes before launch."""
    fixes: list[str] = []
    for blocker in blockers:
        if "Missing category-required" in blocker:
            fixes.append("Add the missing category-required details before launch.")
        elif "Trust proof" in blocker:
            fixes.append("Add authentic proof assets, warranty, support, and trust signals.")
        elif "Price justification" in blocker:
            fixes.append("Defend the price with benefits, proof, and clear total cost.")
        elif "Competitor differentiator" in blocker:
            fixes.append("Add evidence for the claimed competitor differentiator.")
        elif "Shipping" in blocker:
            fixes.append("Clarify shipping, warranty, return, or refund terms.")

    for proof in competitor_gap.get("required_proofs_to_win", []):
        _append_unique(fixes, _short_text(proof))

    return fixes[:6] or ["No required fix identified before launch."]


def _next_best_actions(required_fixes: list[str]) -> list[str]:
    """Create concise next actions from required fixes."""
    actions = []
    for fix in required_fixes:
        if "missing category" in fix.lower():
            actions.append("Complete the structured product brief with category-specific details.")
        elif "proof" in fix.lower():
            actions.append("Upload or describe real proof assets before re-running the audit.")
        elif "price" in fix.lower():
            actions.append("Rewrite the value proposition around price defense and outcomes.")
        elif "shipping" in fix.lower() or "return" in fix.lower():
            actions.append("Make shipping, warranty, and returns visible near the CTA.")
    return _short_list(actions, default=required_fixes, limit=5)


def _executive_verdict(launch_status: str, main_blocker: str) -> str:
    """Write a short executive launch verdict."""
    if launch_status == "ready":
        return "Launch readiness: Ready. Simulated buyer assessment found no major blocker."
    if launch_status == "not_ready":
        return f"Launch readiness: Not Ready. Main blocker: {main_blocker}"
    return f"Launch readiness: Needs Fixes. Main blocker: {main_blocker}"


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


def _safe_launch_status(value: Any, default: str = "needs_fixes") -> str:
    """Normalize launch status labels from Gemini or fallback logic."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"ready", "needs_fixes", "not_ready"}:
            return normalized
        if "not" in normalized:
            return "not_ready"
        if "ready" in normalized and "need" not in normalized:
            return "ready"
    return default if default in {"ready", "needs_fixes", "not_ready"} else "needs_fixes"


def _safe_score(value: Any, default: int = 0) -> int:
    """Normalize a score into an integer from 0 to 100."""
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = default
    return max(0, min(100, score))


def _safe_dict_list(value: Any, default: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Return a list of compact dictionaries from raw JSON or fallback."""
    if isinstance(value, list):
        rows = [dict(item) for item in value if isinstance(item, dict)]
        if rows:
            return rows[:8]
    return (default or [])[:8]


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
