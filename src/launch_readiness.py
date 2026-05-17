"""Launch readiness reporting for BuyerLab AI pre-launch audits."""

from __future__ import annotations

from dataclasses import asdict
from statistics import mean
from typing import Any

from src.category_intelligence import (
    build_category_expectation_check as build_basic_category_expectation_check,
    get_category_profile,
    normalize_category,
)
from src.price_intelligence import (
    analyze_competitor_gap,
    analyze_local_price_perception,
)
from src.state import (
    AgentResponse,
    BuyerPersona,
    CompetitorGapReport,
    LaunchReadinessReport,
    PricePerceptionReport,
    ProductInput,
    SimulationState,
)


BLOCKER_PRIORITY = [
    "trust",
    "category_information",
    "price_value",
    "competitor_gap",
    "return_risk",
    "clarity",
    "emotional_appeal",
]

HIGH_IMPACT_FIELDS = {
    "battery life",
    "warranty",
    "technical specifications",
    "compatibility",
    "return policy",
    "return/exchange policy",
    "refund policy",
    "real usage proof",
    "portfolio proof",
    "student proof",
    "size guide",
    "material",
    "scope",
    "delivery time",
    "revision policy",
    "instructor credibility",
    "learning outcomes",
}


def build_launch_readiness_report(
    state: SimulationState,
    risk_scores: dict[str, int] | None = None,
    price_report: PricePerceptionReport | None = None,
    competitor_gap: CompetitorGapReport | None = None,
    buyer_loss_analysis: list[dict[str, Any]] | None = None,
    clustered_objections: dict[str, list[str]] | None = None,
) -> LaunchReadinessReport:
    """Build an AI-assisted launch readiness diagnostic, not a market prediction."""
    product = state.product
    responses = state.first_round_responses
    price_report = price_report or analyze_local_price_perception(product)
    competitor_gap = competitor_gap or analyze_competitor_gap(product)
    category_check = build_category_expectation_check(product)
    persona_verdicts = build_buyer_persona_verdicts(responses, state.personas)
    simulated_conversion_score = _estimate_conversion_score(responses)
    risk_scores = risk_scores or _calculate_risk_scores(
        product,
        responses,
        category_check,
        price_report,
    )

    blockers = prioritize_conversion_blockers(
        product=product,
        agent_responses=responses,
        category_expectation_check=category_check,
        price_report=price_report,
        competitor_gap=competitor_gap,
        risk_scores=risk_scores,
        clustered_objections=clustered_objections,
    )
    required_fixes = build_required_fix_list(
        product=product,
        blockers=blockers,
        category_expectation_check=category_check,
        price_report=price_report,
        competitor_gap=competitor_gap,
    )
    launch_score = calculate_launch_readiness_score(
        simulated_conversion_score=simulated_conversion_score,
        risk_scores=risk_scores,
        category_expectation_check=category_check,
        price_report=price_report,
        competitor_gap=competitor_gap,
        buyer_persona_verdicts=persona_verdicts,
    )
    launch_status = determine_launch_status(
        launch_readiness_score=launch_score,
        trust_risk_score=risk_scores["trust_risk_score"],
        return_risk_score=risk_scores["return_risk_score"],
        simulated_conversion_score=simulated_conversion_score,
        category_expectation_check=category_check,
        price_report=price_report,
        competitor_gap=competitor_gap,
    )
    main_blocker = blockers[0] if blockers else "No major blocker identified."
    buyer_loss_summary = _buyer_loss_summary(persona_verdicts)
    price_verdict = _price_justification_verdict(
        product,
        price_report,
        category_check,
        competitor_gap,
    )
    competitor_verdict = _competitor_gap_verdict(product, competitor_gap)
    launch_decision = _launch_decision_summary(launch_status, required_fixes)

    return LaunchReadinessReport(
        launch_readiness_score=launch_score,
        launch_status=launch_status,
        executive_verdict=_executive_verdict(
            launch_status,
            main_blocker,
            persona_verdicts,
            required_fixes,
        ),
        main_blocker=main_blocker,
        simulated_conversion_score=simulated_conversion_score,
        trust_risk_score=risk_scores["trust_risk_score"],
        price_resistance_score=risk_scores["price_resistance_score"],
        clarity_score=risk_scores["clarity_score"],
        return_risk_score=risk_scores["return_risk_score"],
        category_expectation_check=category_check,
        local_price_perception_summary=price_report.pricing_comment,
        competitor_gap_summary=competitor_gap.value_gap_summary,
        buyer_persona_verdicts=persona_verdicts,
        buyer_loss_summary=buyer_loss_summary,
        top_conversion_blockers=blockers,
        required_fix_before_launch=required_fixes,
        next_best_actions=_next_best_actions(required_fixes),
        price_justification_verdict=price_verdict,
        competitor_gap_verdict=competitor_verdict,
        launch_decision_summary=launch_decision,
        summary=_summary(
            launch_status,
            launch_score,
            simulated_conversion_score,
            launch_decision,
        ),
    )


def build_launch_readiness_summary(report_inputs: dict[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper returning a dict launch readiness summary."""
    state = report_inputs["state"]
    report = build_launch_readiness_report(
        state=state,
        risk_scores=report_inputs.get("risk_scores"),
        price_report=_coerce_price_report(report_inputs.get("price_perception")),
        competitor_gap=_coerce_competitor_gap(report_inputs.get("competitor_gap")),
        buyer_loss_analysis=report_inputs.get("buyer_losses"),
        clustered_objections=report_inputs.get("clustered_objections"),
    )
    return asdict(report)


def calculate_launch_readiness_score(
    simulated_conversion_score: int,
    risk_scores: dict[str, int],
    category_expectation_check: list[dict[str, Any]],
    price_report: PricePerceptionReport,
    competitor_gap: CompetitorGapReport,
    buyer_persona_verdicts: list[dict[str, Any]],
) -> int:
    """Calculate a 0-100 AI-assisted diagnostic score, not a market prediction."""
    category_penalty = _category_penalty(category_expectation_check)
    trust_penalty = round(risk_scores.get("trust_risk_score", 0) * 0.14)
    price_penalty = round(risk_scores.get("price_resistance_score", 0) * 0.12)
    return_penalty = round(risk_scores.get("return_risk_score", 0) * 0.1)
    clarity_bonus = round((risk_scores.get("clarity_score", 0) - 50) * 0.08)
    premium_penalty = 8 if _premium_price_needs_proof(price_report) else 0
    competitor_penalty = 8 if _competitor_gap_is_unjustified(competitor_gap) else 0
    persona_penalty = sum(
        4
        for verdict in buyer_persona_verdicts
        if verdict.get("business_impact") == "high"
        and verdict.get("decision") != "buy"
    )

    score = (
        simulated_conversion_score
        - category_penalty
        - trust_penalty
        - price_penalty
        - return_penalty
        - premium_penalty
        - competitor_penalty
        - persona_penalty
        + clarity_bonus
    )
    if simulated_conversion_score >= 40:
        score = max(score, min(35, round(simulated_conversion_score * 0.35)))
    return _safe_score(score)


def determine_launch_status(
    launch_readiness_score: int,
    trust_risk_score: int = 0,
    return_risk_score: int = 0,
    simulated_conversion_score: int = 100,
    category_expectation_check: list[dict[str, Any]] | None = None,
    price_report: PricePerceptionReport | None = None,
    competitor_gap: CompetitorGapReport | None = None,
) -> str:
    """Determine launch status from thresholds and critical blocker overrides."""
    if launch_readiness_score >= 80:
        status = "ready"
    elif launch_readiness_score >= 55:
        status = "needs_fixes"
    else:
        status = "not_ready"

    critical_missing = _critical_missing_count(category_expectation_check or [])
    if status == "ready" and (
        trust_risk_score >= 65
        or critical_missing > 0
        or (price_report is not None and _premium_price_needs_proof(price_report))
        or (competitor_gap is not None and _competitor_gap_is_unjustified(competitor_gap))
        or return_risk_score >= 60
        or simulated_conversion_score < 65
    ):
        status = "needs_fixes"

    if (
        trust_risk_score >= 85
        or critical_missing >= 3
        or return_risk_score >= 80
        or simulated_conversion_score < 45
        or (price_report is not None and price_report.price_band == "irrational")
        or (
            competitor_gap is not None
            and _competitor_gap_is_unjustified(competitor_gap)
            and launch_readiness_score < 70
        )
    ):
        status = "not_ready"

    return status


def build_category_expectation_check(product: ProductInput) -> list[dict[str, Any]]:
    """Check required category information using heuristic category profiles."""
    profile = get_category_profile(product.normalized_category or product.category)
    basic_rows = {
        row["field"]: row for row in build_basic_category_expectation_check(product)
    }
    rows: list[dict[str, Any]] = []

    for field_name in profile.required_information_fields:
        status = basic_rows.get(field_name, {}).get("status", "missing")
        if _field_is_declared_missing(product, field_name):
            status = "missing"
        rows.append(
            {
                "field_name": field_name,
                "field": field_name,
                "status": status,
                "impact": _field_impact(field_name, status),
                "explanation": _field_explanation(profile.display_name, field_name, status),
                "suggested_fix": _field_suggested_fix(product, field_name, status),
            }
        )
    return rows


def build_buyer_persona_verdicts(
    agent_responses: list[AgentResponse],
    personas: list[BuyerPersona] | None = None,
) -> list[dict[str, Any]]:
    """Normalize buyer persona decisions into dashboard-ready verdict objects."""
    persona_names = {persona.id: persona.name for persona in personas or []}
    verdicts: list[dict[str, Any]] = []

    for response in agent_responses:
        decision = _safe_decision(_response_value(response, "decision"))
        purchase_intent = _safe_score(_response_value(response, "purchase_intent"))
        objections = _short_list(_response_value(response, "objections"), limit=3)
        top_objection = objections[0] if objections else "No major objection."
        persona_id = _safe_text(_response_value(response, "persona_id")) or "unknown_persona"

        verdicts.append(
            {
                "persona_name": persona_names.get(persona_id, _persona_name_from_id(persona_id)),
                "decision": decision,
                "purchase_intent": purchase_intent,
                "confidence": _safe_score(_response_value(response, "confidence")),
                "main_reason": _safe_text(_response_value(response, "main_reason")),
                "top_objection": top_objection,
                "suggested_fix": _safe_text(_response_value(response, "suggested_fix"))
                or "No fix suggested.",
                "business_impact": _business_impact(decision, purchase_intent, objections),
            }
        )
    return verdicts


def prioritize_conversion_blockers(
    product: ProductInput,
    agent_responses: list[AgentResponse],
    category_expectation_check: list[dict[str, Any]],
    price_report: PricePerceptionReport,
    competitor_gap: CompetitorGapReport,
    risk_scores: dict[str, int] | None = None,
    clustered_objections: dict[str, list[str]] | None = None,
) -> list[str]:
    """Prioritize conversion blockers by expected business impact."""
    risk_scores = risk_scores or _calculate_risk_scores(
        product,
        agent_responses,
        category_expectation_check,
        price_report,
    )
    blockers: dict[str, list[str]] = {key: [] for key in BLOCKER_PRIORITY}
    objection_text = " ".join(_all_response_objections(agent_responses)).lower()

    if risk_scores["trust_risk_score"] >= 55 or _trust_is_weak(product):
        blockers["trust"].append(
            "Trust proof is weak: buyers need real proof assets, warranty clarity, and visible support."
        )

    missing_fields = [
        row["field_name"]
        for row in category_expectation_check
        if row["status"] in {"missing", "weak"} and row["impact"] in {"high", "medium"}
    ]
    if missing_fields:
        blockers["category_information"].append(
            "Missing category-critical information: "
            f"{', '.join(missing_fields[:4])}."
        )

    if risk_scores["price_resistance_score"] >= 60 or _premium_price_needs_proof(price_report):
        blockers["price_value"].append(
            "Price/value justification is weak for the heuristic local price perception band."
        )

    if _competitor_gap_is_unjustified(competitor_gap):
        blockers["competitor_gap"].append(
            "Competitor gap is not justified with proof or a clear comparison."
        )

    if risk_scores["return_risk_score"] >= 50:
        blockers["return_risk"].append(
            "Return, warranty, shipping, or delivery terms do not reduce buyer risk enough."
        )

    if risk_scores["clarity_score"] < 55 or "unclear" in objection_text:
        blockers["clarity"].append(
            "Product value and key claims are not clear enough for a fast launch decision."
        )

    if _has_emotional_gap(agent_responses):
        blockers["emotional_appeal"].append(
            "Emotional appeal or CTA strength is not compelling enough to create urgency."
        )

    ordered: list[str] = []
    for priority in BLOCKER_PRIORITY:
        for blocker in blockers[priority]:
            _append_unique(ordered, blocker)
    return ordered[:6] or ["No major blocker found in the simulated buyer assessment."]


def build_required_fix_list(
    product: ProductInput,
    blockers: list[str],
    category_expectation_check: list[dict[str, Any]],
    price_report: PricePerceptionReport,
    competitor_gap: CompetitorGapReport,
) -> list[str]:
    """Turn prioritized blockers into concrete fixes before launch."""
    fixes: list[str] = []
    missing_fields = [
        row["field_name"]
        for row in category_expectation_check
        if row["status"] in {"missing", "weak"} and row["impact"] in {"high", "medium"}
    ]

    if any("Trust proof" in blocker for blocker in blockers):
        fixes.append(
            "Add real trust signals: exact warranty duration, return conditions, support contact, proof assets, and visible purchase safety."
        )

    if missing_fields:
        fixes.append(_category_specific_fix(product, missing_fields))

    if any("Price/value" in blocker for blocker in blockers):
        fixes.append(
            "Explain what buyers get for this price and support it with specs, materials, outcomes, or proof assets."
        )
        if price_report.price_band in {"upper_mid", "premium", "irrational"}:
            fixes.append(
                f"Add proof that defends the {price_report.price_band} price band before launch."
            )

    if any("Competitor gap" in blocker for blocker in blockers):
        price_gap = competitor_gap.price_gap
        competitor_name = _competitor_name(product)
        if price_gap and price_gap > 0:
            fixes.append(
                f"Explain why this product is worth {price_gap:g} {price_report.currency} more than {competitor_name}."
            )
        fixes.append("Add a short comparison table against the alternative product.")

    if any("Return" in blocker for blocker in blockers):
        fixes.append("Make shipping cost, delivery time, warranty period, and return rules visible near the CTA.")

    if any("clear" in blocker.lower() for blocker in blockers):
        fixes.append("Rewrite the first screen around the product type, main benefit, proof, price, and CTA.")

    if any("Emotional" in blocker for blocker in blockers):
        fixes.append("Sharpen the CTA and add a concrete use-case promise without exaggerating claims.")

    for proof in competitor_gap.required_proofs_to_win[:3]:
        if "Clear benefit explanation" not in proof and "Shipping and return terms" not in proof:
            _append_unique(fixes, _proof_to_fix(proof))

    return fixes[:7] or ["No required fix identified before launch."]


def _calculate_risk_scores(
    product: ProductInput,
    responses: list[AgentResponse],
    category_check: list[dict[str, Any]],
    price_report: PricePerceptionReport,
) -> dict[str, int]:
    """Calculate risk scores from buyer responses and structured product brief gaps."""
    objection_text = " ".join(_all_response_objections(responses)).lower()
    missing_high = sum(
        1 for row in category_check if row["status"] == "missing" and row["impact"] == "high"
    )
    weak_or_missing = sum(1 for row in category_check if row["status"] != "present")

    trust_risk = 0
    if _trust_is_weak(product):
        trust_risk += 55
    if any(word in objection_text for word in ["trust", "proof", "review", "guarantee"]):
        trust_risk += 25

    price_risk = price_report.perceived_value_risk
    if any(word in objection_text for word in ["price", "expensive", "value", "worth"]):
        price_risk += 15

    return_risk = 0
    if not product.warranty_or_return_policy.strip():
        return_risk += 45
    if not product.shipping_info.strip():
        return_risk += 30
    if any(word in objection_text for word in ["return", "refund", "shipping", "delivery"]):
        return_risk += 20

    clarity_penalty = min(100, weak_or_missing * 9 + missing_high * 8)
    if len((product.description or "").strip()) < 80:
        clarity_penalty += 12
    if len((product.value_proposition or "").strip()) < 40:
        clarity_penalty += 10

    return {
        "trust_risk_score": _safe_score(trust_risk),
        "price_resistance_score": _safe_score(price_risk),
        "clarity_score": _safe_score(100 - clarity_penalty),
        "return_risk_score": _safe_score(return_risk),
    }


def _category_penalty(category_check: list[dict[str, Any]]) -> int:
    """Convert category expectation gaps into a score penalty."""
    penalty = 0
    for row in category_check:
        status = row.get("status")
        impact = row.get("impact")
        if status == "missing":
            penalty += 8 if impact == "high" else 5 if impact == "medium" else 2
        elif status == "weak":
            penalty += 4 if impact == "high" else 2 if impact == "medium" else 1
    return min(40, penalty)


def _field_impact(field_name: str, status: str) -> str:
    """Estimate business impact for one category requirement."""
    if status == "present":
        return "low"
    if field_name in HIGH_IMPACT_FIELDS:
        return "high"
    return "medium" if status == "missing" else "low"


def _field_explanation(category_name: str, field_name: str, status: str) -> str:
    """Explain why a category requirement affects launch readiness."""
    if status == "present":
        return f"{field_name} is covered enough for a first {category_name} audit pass."
    if status == "weak":
        return f"{field_name} is only partially covered; buyers may still hesitate."
    return f"{category_name} buyers expect {field_name} before launch."


def _field_suggested_fix(product: ProductInput, field_name: str, status: str) -> str:
    """Create a concrete suggested fix for one missing or weak field."""
    if status == "present":
        return "Keep this information visible near the relevant product section."

    category = normalize_category(product.normalized_category or product.category)
    if category == "electronics_accessory" and field_name in {
        "battery life",
        "technical specifications",
        "compatibility",
        "real usage proof",
    }:
        return "Add exact battery life, compatibility, core specs, and real usage proof."
    if category == "fashion_shoes" and field_name in {
        "size guide",
        "fit information",
        "material",
        "real product photos",
    }:
        return "Add size guide, fit notes, material details, and real product photos."
    if category == "small_home_appliance":
        return "Add specs, usage details, warranty period, service terms, and delivery clarity."
    if category == "digital_service":
        return "Add scope, delivery time, revision policy, portfolio proof, and exclusions."
    if category == "online_course":
        return "Add curriculum, instructor proof, outcomes, time commitment, and refund terms."
    if category == "handmade_bag":
        return "Add material, dimensions, craftsmanship proof, care notes, and return terms."
    return f"Add clear {field_name} with concrete proof before launch."


def _field_is_declared_missing(product: ProductInput, field_name: str) -> bool:
    """Treat seller-declared limitations as missing, even if the field keyword appears."""
    limitation_text = " ".join(product.known_limitations).lower()
    field_text = field_name.lower()
    if field_text not in limitation_text:
        return False
    return any(
        marker in limitation_text
        for marker in ["missing", "not specified", "unclear", "no ", "none"]
    )


def _category_specific_fix(product: ProductInput, missing_fields: list[str]) -> str:
    """Create a concrete category-critical information fix."""
    category = normalize_category(product.normalized_category or product.category)
    if category == "electronics_accessory":
        fix = "Add exact battery life, warranty period, Bluetooth/version compatibility, technical specs, return policy, and real usage proof before launch."
        if _product_mentions_audio(product):
            fix += " Include microphone or sound quality proof for this model."
        return fix
    if category == "fashion_shoes":
        return "Add a size guide, fit information, material details, return/exchange terms, comfort use case, and real product photos."
    if category == "small_home_appliance":
        return "Add warranty, technical specifications, capacity, power usage, cleaning/maintenance details, safety information, and delivery terms."
    if category == "digital_service":
        return "Add scope, delivery time, revision policy, portfolio proof, support terms, and what is included or excluded."
    if category == "online_course":
        return "Add instructor credibility, curriculum clarity, learning outcomes, target student level, sample lesson/proof, and certificate or refund policy only if real."
    if category == "handmade_bag":
        return "Add material proof, dimensions, craftsmanship details, real product photos, return policy, and authenticity/trust proof."
    return f"Add category-critical details before launch: {', '.join(missing_fields[:5])}."


def _next_best_actions(required_fixes: list[str]) -> list[str]:
    """Convert required fixes into short seller-facing next actions."""
    actions: list[str] = []
    for fix in required_fixes:
        lowered = fix.lower()
        if "trust" in lowered or "proof" in lowered:
            _append_unique(actions, "Collect real proof assets and place them near price and CTA.")
        elif "battery" in lowered or "size guide" in lowered or "technical" in lowered:
            _append_unique(actions, "Complete the category-critical product information.")
        elif "worth" in lowered or "price" in lowered:
            _append_unique(actions, "Rewrite price justification with concrete value proof.")
        elif "comparison" in lowered or "competitor" in lowered:
            _append_unique(actions, "Add a short competitor comparison using seller-provided facts.")
        elif "shipping" in lowered or "return" in lowered or "warranty" in lowered:
            _append_unique(actions, "Make shipping, warranty, and return terms visible.")
    return actions[:5] or required_fixes[:5]


def _summary(
    status: str,
    launch_score: int,
    conversion_score: int,
    launch_decision: str,
) -> str:
    """Write a short report summary with careful non-predictive wording."""
    return (
        f"Launch readiness is {_status_label(status)} with a {launch_score}/100 AI-assisted "
        f"diagnostic score and {conversion_score}/100 simulated conversion score; "
        f"not a real market prediction. {launch_decision}"
    )


def _executive_verdict(
    status: str,
    main_blocker: str,
    persona_verdicts: list[dict[str, Any]],
    required_fixes: list[str],
) -> str:
    """Write a 10-second executive verdict."""
    business_impact = _business_impact_summary(persona_verdicts)
    recommendation = _launch_recommendation(status, required_fixes)
    if status == "ready":
        return (
            "This product page is ready to launch in the simulated buyer "
            f"assessment. Business impact: {business_impact} Recommendation: {recommendation}"
        )
    if status == "not_ready":
        return (
            "This product page is not ready to launch. "
            f"Main blocker: {main_blocker} Business impact: {business_impact} "
            f"Recommendation: {recommendation}"
        )
    return (
        "This product page needs fixes before launch. "
        f"Main blocker: {main_blocker} Business impact: {business_impact} "
        f"Recommendation: {recommendation}"
    )


def _buyer_loss_summary(persona_verdicts: list[dict[str, Any]]) -> str:
    """Summarize which simulated buyer personas are lost and why."""
    lost = [
        verdict
        for verdict in persona_verdicts
        if verdict.get("decision") in {"reject", "hesitate"}
    ]
    if not lost:
        return "No buyer persona was lost in this simulated buyer assessment."

    names = ", ".join(str(verdict.get("persona_name", "Unknown Persona")) for verdict in lost[:3])
    reasons = [
        str(verdict.get("top_objection") or verdict.get("main_reason") or "").strip()
        for verdict in lost
    ]
    reasons = [reason for reason in reasons if reason and reason != "No major objection."]
    reason_text = "; ".join(reasons[:3]) or "the page does not create enough purchase confidence"
    return (
        f"BuyerLab lost the {names} persona(s). This creates business risk because "
        f"buyers are likely to hesitate over: {reason_text}."
    )


def _price_justification_verdict(
    product: ProductInput,
    price_report: PricePerceptionReport,
    category_check: list[dict[str, Any]],
    competitor_gap: CompetitorGapReport,
) -> str:
    """Explain whether the page proves the local price without claiming market data."""
    category_name = get_category_profile(
        price_report.normalized_category
    ).display_name.lower()
    price_label = f"{price_report.price:g} {price_report.currency}"
    proof_gaps = [
        row["field_name"]
        for row in category_check
        if row["status"] in {"missing", "weak"} and row["impact"] in {"high", "medium"}
    ]
    competitor_gap_text = ""
    if competitor_gap.price_gap is not None and competitor_gap.price_gap > 0:
        competitor_gap_text = (
            f" It is {competitor_gap.price_gap:g} {price_report.currency} above the competitor, "
            "so the difference must be explained."
        )

    if price_report.currency == "TRY":
        prefix = (
            f"{price_label} places this product in a {price_report.price_band} "
            f"perception band for {category_name}."
        )
    else:
        prefix = (
            f"{price_label} is assessed with a generic {price_report.price_band} "
            "price perception fallback."
        )

    if price_report.perceived_value_risk >= 60 or proof_gaps:
        gaps = ", ".join(proof_gaps[:4]) or "value proof"
        return (
            f"{prefix} The page does not yet justify this price because {gaps} "
            f"are missing or weak.{competitor_gap_text}"
        )
    return f"{prefix} The page has enough proof for a first simulated price pass.{competitor_gap_text}"


def _competitor_gap_verdict(
    product: ProductInput,
    competitor_gap: CompetitorGapReport,
) -> str:
    """Explain seller-provided competitor gap without live competitor research."""
    competitor = product.competitor_context
    if competitor is None or not (
        competitor.competitor_name
        or competitor.competitor_price
        or competitor.our_differentiator
        or competitor.competitor_strengths
    ):
        return (
            "Competitor context was not provided, so BuyerLab cannot evaluate "
            "price positioning against alternatives."
        )

    name = competitor.competitor_name or "the provided competitor"
    if competitor_gap.price_gap is None:
        price_text = "No same-currency competitor price gap could be calculated."
    elif competitor_gap.price_gap > 0:
        price_text = f"The product is {competitor_gap.price_gap:g} above {name}."
    elif competitor_gap.price_gap < 0:
        price_text = f"The product is {abs(competitor_gap.price_gap):g} below {name}."
    else:
        price_text = f"The product is priced at parity with {name}."

    proof_text = (
        "The differentiator is not proven yet."
        if competitor_gap.our_unproven_claims
        else "The differentiator has some provided support, but should stay visible."
    )
    return (
        f"{price_text} {proof_text} Position against the competitor with a short "
        "comparison table and seller-provided proof."
    )


def _launch_decision_summary(status: str, required_fixes: list[str]) -> str:
    """Create the final short launch decision."""
    first_fix = required_fixes[0] if required_fixes else "keep the current proof visible"
    if status == "not_ready":
        return (
            "Decision: Do not launch yet. Fix trust proof, price justification, "
            f"and category-critical details first. Start with: {first_fix}"
        )
    if status == "needs_fixes":
        return (
            "Decision: Launch is possible only after required fixes before launch "
            f"are completed. Start with: {first_fix}"
        )
    return "Decision: Launch is possible. Keep proof, pricing, and risk reducers visible."


def _status_label(status: str) -> str:
    """Convert internal status labels into user-facing labels."""
    return {
        "ready": "Ready",
        "needs_fixes": "Needs Fixes",
        "not_ready": "Not Ready",
    }.get(status, "Needs Fixes")


def _business_impact_summary(persona_verdicts: list[dict[str, Any]]) -> str:
    """Summarize business impact from lost personas."""
    high_impact_losses = [
        verdict
        for verdict in persona_verdicts
        if verdict.get("business_impact") == "high"
        and verdict.get("decision") != "buy"
    ]
    if high_impact_losses:
        names = ", ".join(str(verdict.get("persona_name")) for verdict in high_impact_losses[:2])
        return f"high, because {names} are rejecting or strongly hesitating."
    if any(verdict.get("decision") != "buy" for verdict in persona_verdicts):
        return "medium, because some buyer personas still hesitate."
    return "low, because no major simulated buyer segment is blocked."


def _launch_recommendation(status: str, required_fixes: list[str]) -> str:
    """Return a concise launch recommendation."""
    if status == "ready":
        return "Launch can proceed, but keep monitoring real buyer behavior."
    first_fix = required_fixes[0] if required_fixes else "complete the required fixes before launch"
    if status == "not_ready":
        return f"Do not launch yet; {first_fix}"
    return f"Launch only after this fix is completed: {first_fix}"


def _estimate_conversion_score(responses: list[AgentResponse]) -> int:
    """Estimate simulated conversion score from buyer persona purchase intent."""
    if not responses:
        return 0
    return _safe_score(round(mean(_safe_score(_response_value(r, "purchase_intent")) for r in responses)))


def _critical_missing_count(category_check: list[dict[str, Any]]) -> int:
    """Count high-impact missing or weak category fields."""
    return sum(
        1
        for row in category_check
        if row.get("impact") == "high" and row.get("status") in {"missing", "weak"}
    )


def _premium_price_needs_proof(price_report: PricePerceptionReport) -> bool:
    """Return True when upper price bands require stronger proof before launch."""
    return (
        price_report.price_band in {"upper_mid", "premium", "irrational"}
        and price_report.perceived_value_risk >= 60
    )


def _competitor_gap_is_unjustified(competitor_gap: CompetitorGapReport) -> bool:
    """Return True when seller-provided competitor context reveals an unproven gap."""
    if competitor_gap.price_gap is not None and competitor_gap.price_gap > 0:
        return True
    return bool(competitor_gap.our_unproven_claims)


def _trust_is_weak(product: ProductInput) -> bool:
    """Detect weak trust proof from explicit seller inputs."""
    return not (
        product.trust_signals
        and (product.proof_assets or product.reviews_or_social_proof.strip())
        and product.warranty_or_return_policy.strip()
    )


def _has_emotional_gap(responses: list[AgentResponse]) -> bool:
    """Detect emotional appeal or CTA objections in simulated buyer responses."""
    text = " ".join(_all_response_objections(responses)).lower()
    return any(
        word in text
        for word in ["boring", "desire", "emotion", "urgency", "fomo", "visual", "cta"]
    )


def _all_response_objections(responses: list[AgentResponse]) -> list[str]:
    """Collect all objection-like response fields."""
    texts: list[str] = []
    for response in responses:
        texts.extend(_short_list(_response_value(response, "objections"), limit=5))
        texts.extend(_short_list(_response_value(response, "missing_information"), limit=5))
        main_reason = _safe_text(_response_value(response, "main_reason"))
        if main_reason:
            texts.append(main_reason)
    return texts


def _business_impact(decision: str, purchase_intent: int, objections: list[str]) -> str:
    """Estimate the business impact of one persona verdict."""
    if decision == "reject" or purchase_intent < 35:
        return "high"
    if decision == "hesitate" or objections:
        return "medium"
    return "low"


def _proof_to_fix(proof: str) -> str:
    """Convert a required proof label into an action item."""
    return f"Add proof for: {proof}."


def _competitor_name(product: ProductInput) -> str:
    """Return a safe competitor name for fix copy."""
    if product.competitor_context and product.competitor_context.competitor_name:
        return product.competitor_context.competitor_name
    return "the competitor"


def _product_mentions_audio(product: ProductInput) -> bool:
    """Detect when electronics copy should ask for microphone or sound proof."""
    text = " ".join(
        [
            product.product_type,
            product.title,
            product.description,
            product.value_proposition,
            product.intended_use_case,
            " ".join(product.known_limitations),
        ]
    ).lower()
    return any(word in text for word in ["earbud", "headphone", "audio", "mic", "microphone", "sound"])


def _coerce_price_report(value: Any) -> PricePerceptionReport | None:
    """Coerce a dict price report when available; otherwise let callers recalculate."""
    if isinstance(value, PricePerceptionReport):
        return value
    if isinstance(value, dict):
        try:
            return PricePerceptionReport(**value)
        except (TypeError, ValueError):
            return None
    return None


def _coerce_competitor_gap(value: Any) -> CompetitorGapReport | None:
    """Coerce a dict competitor report when available; otherwise recalculate."""
    if isinstance(value, CompetitorGapReport):
        return value
    if isinstance(value, dict):
        try:
            return CompetitorGapReport(**value)
        except TypeError:
            return None
    return None


def _persona_name_from_id(persona_id: str) -> str:
    """Create a readable fallback persona name from an id."""
    if not persona_id or persona_id == "unknown_persona":
        return "Unknown Persona"
    return persona_id.replace("_", " ").title()


def _response_value(response: Any, field_name: str) -> Any:
    """Safely read a field from dataclass or dict response objects."""
    if isinstance(response, dict):
        return response.get(field_name)
    return getattr(response, field_name, None)


def _safe_decision(value: Any) -> str:
    """Normalize buyer decision labels."""
    if isinstance(value, str):
        decision = value.strip().lower()
        if decision in {"buy", "reject", "hesitate"}:
            return decision
        if "reject" in decision:
            return "reject"
        if "buy" in decision:
            return "buy"
    return "hesitate"


def _safe_score(value: Any) -> int:
    """Normalize a score into 0-100."""
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = 0
    return max(0, min(100, score))


def _safe_text(value: Any, limit: int = 140) -> str:
    """Normalize optional dashboard text."""
    if value is None:
        return ""
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _short_list(value: Any, limit: int = 5) -> list[str]:
    """Return a compact list of strings."""
    if isinstance(value, list):
        return [_safe_text(item) for item in value if _safe_text(item)][:limit]
    if isinstance(value, str) and value.strip():
        return [_safe_text(value)]
    return []


def _append_unique(items: list[str], value: str) -> None:
    """Append a value once while preserving order."""
    text = _safe_text(value, limit=240)
    if text and text not in items:
        items.append(text)


__all__ = [
    "calculate_launch_readiness_score",
    "determine_launch_status",
    "build_category_expectation_check",
    "build_buyer_persona_verdicts",
    "prioritize_conversion_blockers",
    "build_required_fix_list",
    "build_launch_readiness_report",
    "build_launch_readiness_summary",
]
