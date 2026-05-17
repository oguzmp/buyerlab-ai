"""Before-after optimization helpers for BuyerLab AI simulations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from src.gemini_client import generate_json
from src.launch_readiness import build_category_expectation_check
from src.price_intelligence import analyze_competitor_gap, analyze_local_price_perception
from src.state import (
    AttentionMapReport,
    PAGE_SECTION_NAMES,
    ProductInput,
    SimulationReport,
    SimulationState,
)


@dataclass(slots=True)
class OptimizedProductSuggestion:
    """Dashboard-ready product page copy improvements for a second simulation."""

    title: str = ""
    description: str = ""
    value_proposition: str = ""
    warranty_or_return_policy: str = ""
    shipping_info: str = ""
    trust_signals: list[str] = field(default_factory=list)
    trust_proof_checklist: list[str] = field(default_factory=list)
    faq_items: list[str] = field(default_factory=list)
    competitor_comparison_suggestion: str = ""
    missing_information_checklist: list[str] = field(default_factory=list)
    call_to_action: str = ""
    change_summary: list[str] = field(default_factory=list)


def build_optimization_prompt(
    product: ProductInput,
    final_report: SimulationReport,
    attention_map: AttentionMapReport | None = None,
    buyer_loss_analysis: list[dict[str, Any]] | None = None,
) -> str:
    """Build a Gemini prompt for realistic product page optimization."""
    context = {
        "important_note": (
            "The current score is a simulated conversion score, not a real market "
            "prediction."
        ),
        "product": asdict(product),
        "final_report": asdict(final_report),
        "attention_map": asdict(attention_map) if attention_map else None,
        "buyer_loss_analysis": buyer_loss_analysis or [],
    }

    return f"""
You are optimizing a pre-launch e-commerce product page for BuyerLab AI.
Use the simulation report to propose realistic copy improvements that can be
tested in a second simulation.

Context:
{json.dumps(context, ensure_ascii=True, indent=2)}

Return only valid JSON with this exact shape:
{{
  "title": "",
  "description": "",
  "value_proposition": "",
  "warranty_or_return_policy": "",
  "shipping_info": "",
  "trust_signals": [],
  "trust_proof_checklist": [],
  "faq_items": [],
  "competitor_comparison_suggestion": "",
  "missing_information_checklist": [],
  "call_to_action": "",
  "change_summary": []
}}

Rules:
- Write all user-facing copy and checklist items in Turkish.
- Keep all text concise, practical, and seller-focused.
- Prioritize trust blockers first.
- Then address missing product information.
- Then improve price/value justification.
- Then improve emotional appeal and CTA.
- Do not exaggerate product claims.
- Do not invent fake reviews, fake certifications, fake awards, or fake guarantees.
- If trust signals are missing, suggest adding real trust signals the seller can verify.
- If proof is missing, say "add real proof" instead of fabricating facts.
- Include a missing_information_checklist with category-critical fields the seller
  must add before launch.
- Include competitor_comparison_suggestion only from seller-provided competitor data.
- Mention "simulated conversion score" only as a testing signal, not a prediction.
""".strip()


def generate_optimized_product(
    product: ProductInput,
    final_report: SimulationReport,
    attention_map: AttentionMapReport | None = None,
    buyer_loss_analysis: list[dict[str, Any]] | None = None,
) -> OptimizedProductSuggestion:
    """Generate optimized product copy, falling back safely on invalid Gemini output."""
    try:
        raw_suggestion = generate_json(
            build_optimization_prompt(
                product=product,
                final_report=final_report,
                attention_map=attention_map,
                buyer_loss_analysis=buyer_loss_analysis,
            )
        )
        return _suggestion_from_json(raw_suggestion, product, final_report, attention_map)
    except Exception:
        return _fallback_suggestion(product, final_report, attention_map)


def apply_optimization_to_product(
    product: ProductInput,
    suggestion: OptimizedProductSuggestion,
) -> ProductInput:
    """Apply optimized copy suggestions to a ProductInput for re-simulation."""
    return replace(
        product,
        title=suggestion.title or product.title,
        description=suggestion.description or product.description,
        value_proposition=suggestion.value_proposition or product.value_proposition,
        warranty_or_return_policy=(
            suggestion.warranty_or_return_policy
            or product.warranty_or_return_policy
        ),
        shipping_info=suggestion.shipping_info or product.shipping_info,
        trust_signals=suggestion.trust_signals or product.trust_signals,
        call_to_action=suggestion.call_to_action or product.call_to_action,
    )


def compare_before_after(
    before_state: SimulationState,
    after_state: SimulationState,
) -> dict[str, Any]:
    """Compare two simulation states using simulated conversion score signals."""
    before_score = _state_score(before_state)
    after_score = _state_score(after_state)
    score_delta = after_score - before_score

    return {
        "before_score": before_score,
        "after_score": after_score,
        "score_delta": score_delta,
        "improved_sections": _improved_sections(before_state, after_state),
        "remaining_risks": _remaining_risks(after_state),
        "summary": _comparison_summary(before_score, after_score, score_delta),
    }


def _suggestion_from_json(
    raw_suggestion: dict[str, Any],
    product: ProductInput,
    final_report: SimulationReport,
    attention_map: AttentionMapReport | None,
) -> OptimizedProductSuggestion:
    """Convert Gemini JSON into a safe OptimizedProductSuggestion."""
    suggestion = OptimizedProductSuggestion(
        title=_short_text(raw_suggestion.get("title")) or product.title,
        description=_short_text(raw_suggestion.get("description"), limit=320)
        or _fallback_description(product),
        value_proposition=_short_text(raw_suggestion.get("value_proposition"), limit=220)
        or _fallback_value_proposition(product),
        warranty_or_return_policy=_short_text(
            raw_suggestion.get("warranty_or_return_policy"),
            limit=180,
        )
        or _fallback_warranty(product),
        shipping_info=_short_text(raw_suggestion.get("shipping_info"), limit=180)
        or _fallback_shipping(product),
        trust_signals=_short_list(raw_suggestion.get("trust_signals"), limit=5)
        or _fallback_trust_signals(product),
        trust_proof_checklist=_short_list(
            raw_suggestion.get("trust_proof_checklist"),
            limit=6,
        )
        or _fallback_trust_proof_checklist(product, final_report),
        faq_items=_short_list(raw_suggestion.get("faq_items"), limit=5)
        or _fallback_faq_items(product, final_report),
        competitor_comparison_suggestion=_short_text(
            raw_suggestion.get("competitor_comparison_suggestion"),
            limit=240,
        )
        or _fallback_competitor_comparison(product),
        missing_information_checklist=_short_list(
            raw_suggestion.get("missing_information_checklist"),
            limit=8,
        )
        or _fallback_missing_information_checklist(product),
        call_to_action=_short_text(raw_suggestion.get("call_to_action"), limit=80)
        or _fallback_call_to_action(product),
        change_summary=_short_list(raw_suggestion.get("change_summary"), limit=5)
        or _fallback_change_summary(final_report, attention_map),
    )
    return suggestion


def _fallback_suggestion(
    product: ProductInput,
    final_report: SimulationReport,
    attention_map: AttentionMapReport | None,
) -> OptimizedProductSuggestion:
    """Create safe optimization suggestions when Gemini is unavailable or invalid."""
    return OptimizedProductSuggestion(
        title=_fallback_title(product),
        description=_fallback_description(product),
        value_proposition=_fallback_value_proposition(product),
        warranty_or_return_policy=_fallback_warranty(product),
        shipping_info=_fallback_shipping(product),
        trust_signals=_fallback_trust_signals(product),
        trust_proof_checklist=_fallback_trust_proof_checklist(product, final_report),
        faq_items=_fallback_faq_items(product, final_report),
        competitor_comparison_suggestion=_fallback_competitor_comparison(product),
        missing_information_checklist=_fallback_missing_information_checklist(product),
        call_to_action=_fallback_call_to_action(product),
        change_summary=_fallback_change_summary(final_report, attention_map),
    )


def _fallback_title(product: ProductInput) -> str:
    """Return a clearer title without inventing unsupported claims."""
    identity = " ".join(
        part
        for part in [product.brand, product.model, product.product_type]
        if part
    )
    if identity:
        return _short_text(identity)
    if product.title and product.category:
        return _short_text(f"{product.title} for {product.intended_use_case or product.category}")
    return product.title or "Product for your next launch"


def _fallback_description(product: ProductInput) -> str:
    """Return concise description copy grounded in the original product input."""
    base = product.description or "Describe what the product does and who it helps."
    audience = product.intended_use_case or product.target_audience or "the intended buyer"
    missing = _fallback_missing_information_checklist(product)
    missing_text = f" Add real proof for: {', '.join(missing[:3])}." if missing else ""
    return _short_text(
        f"{base} Built for {audience}. State the key benefit, exact proof, price justification, and next step clearly.{missing_text}",
        limit=320,
    )


def _fallback_value_proposition(product: ProductInput) -> str:
    """Return a sharper value proposition using existing product facts."""
    price_report = analyze_local_price_perception(product)
    if product.value_proposition:
        return _short_text(
            f"{product.value_proposition} Justify the {price_report.price_band} price with concrete proof, risk reduction, and category-specific details.",
            limit=220,
        )
    return "Explain the main benefit, why it matters, and what real proof justifies the price."


def _fallback_warranty(product: ProductInput) -> str:
    """Suggest real warranty or return policy content without fabricating terms."""
    if product.warranty_or_return_policy:
        return product.warranty_or_return_policy
    return "Add your real return window, refund conditions, and warranty coverage."


def _fallback_shipping(product: ProductInput) -> str:
    """Suggest realistic shipping information without inventing logistics."""
    if product.shipping_info:
        return product.shipping_info
    return "Add real shipping cost, delivery timing, regions served, and handling notes."


def _fallback_trust_signals(product: ProductInput) -> list[str]:
    """Suggest verifiable trust signals instead of fabricated social proof."""
    signals = list(product.trust_signals)
    suggestions = [
        "Add verified customer reviews when available.",
        "Show secure checkout and accepted payment methods.",
        "Link to real return, warranty, and support policies.",
    ]
    for suggestion in suggestions:
        if suggestion not in signals:
            signals.append(suggestion)
    return signals[:5]


def _fallback_trust_proof_checklist(
    product: ProductInput,
    final_report: SimulationReport,
) -> list[str]:
    """Create a concrete trust proof checklist without inventing fake claims."""
    checklist = [
        "Add exact warranty duration and return conditions.",
        "Add real support channel and service policy.",
        "Add real proof assets instead of generic quality claims.",
    ]
    if final_report.trust_risk_score >= 50:
        checklist.append("Place trust proof near price and CTA before launch.")
    if not product.reviews_or_social_proof.strip():
        checklist.append("Add verified reviews or previous work examples only when real.")
    return _dedupe_short_list(checklist, limit=6)


def _fallback_faq_items(
    product: ProductInput,
    final_report: SimulationReport,
) -> list[str]:
    """Create FAQ suggestions from report risks and product gaps."""
    faq_items: list[str] = []
    if final_report.return_risk_score >= 40 or not product.warranty_or_return_policy:
        faq_items.append("What is the real return or warranty policy?")
    if final_report.price_resistance_score >= 40:
        faq_items.append("Why is this product worth the price?")
    if final_report.price_justification_verdict:
        faq_items.append("What proof supports the price positioning?")
    if final_report.trust_risk_score >= 40:
        faq_items.append("What proof or real customer feedback is available?")
    if not product.shipping_info:
        faq_items.append("How much does shipping cost and how long does delivery take?")
    if not faq_items:
        faq_items.append("Who is this product best suited for?")
    return faq_items[:5]


def _fallback_competitor_comparison(product: ProductInput) -> str:
    """Suggest a competitor comparison from seller-provided context only."""
    competitor = product.competitor_context
    if competitor is None or not (
        competitor.competitor_name
        or competitor.competitor_price
        or competitor.our_differentiator
    ):
        return "Competitor context was not provided; add one alternative product before writing a comparison."

    gap = analyze_competitor_gap(product)
    name = competitor.competitor_name or "the competitor"
    if gap.price_gap and gap.price_gap > 0:
        return (
            f"Add a short comparison table against {name}: price gap, stated differentiator, "
            "proof assets, warranty/return terms, and why the higher price is justified."
        )
    return (
        f"Add a short comparison table against {name}: price, differentiator, proof assets, "
        "support terms, and buyer risk reducers."
    )


def _fallback_missing_information_checklist(product: ProductInput) -> list[str]:
    """Return category-critical missing information for the fix pack."""
    missing = [
        row["field_name"]
        for row in build_category_expectation_check(product)
        if row["status"] in {"missing", "weak"}
    ]
    missing.extend(product.known_limitations)
    return _dedupe_short_list(missing, limit=8)


def _fallback_call_to_action(product: ProductInput) -> str:
    """Return a clearer CTA grounded in the existing page."""
    if product.call_to_action:
        return product.call_to_action
    return "Check availability"


def _fallback_change_summary(
    final_report: SimulationReport,
    attention_map: AttentionMapReport | None,
) -> list[str]:
    """Summarize optimization priorities by expected simulated business impact."""
    changes: list[str] = []
    if final_report.trust_risk_score >= 40:
        changes.append("Prioritized real trust signals and purchase risk reducers.")
    if final_report.clarity_score < 70:
        changes.append("Clarified product details and value proposition.")
    if final_report.price_resistance_score >= 40:
        changes.append("Added stronger price and value justification.")
    if final_report.required_fix_before_launch:
        changes.append("Focused the fix pack on required fixes before launch.")
    if final_report.launch_decision_summary:
        changes.append(final_report.launch_decision_summary)
    if attention_map and attention_map.highest_friction_section:
        changes.append(f"Reduced friction in {attention_map.highest_friction_section}.")
    if not changes:
        changes.append("Tightened copy for clearer simulated conversion score testing.")
    return changes[:5]


def _state_score(state: SimulationState) -> int:
    """Read a state's simulated conversion score safely."""
    if state.final_report is not None:
        return _safe_score(state.final_report.simulated_conversion_score)
    if state.after_score is not None:
        return _safe_score(state.after_score)
    if state.before_score is not None:
        return _safe_score(state.before_score)
    return 0


def _improved_sections(
    before_state: SimulationState,
    after_state: SimulationState,
) -> list[str]:
    """Identify attention-map sections with improved simulated friction or attention."""
    if not before_state.attention_map or not after_state.attention_map:
        return []

    before_sections = {
        score.section_name: score for score in before_state.attention_map.section_scores
    }
    improved: list[str] = []
    for after_score in after_state.attention_map.section_scores:
        before_score = before_sections.get(after_score.section_name)
        if before_score is None:
            continue
        friction_improved = after_score.friction_score < before_score.friction_score
        attention_improved = after_score.attention_score > before_score.attention_score
        if friction_improved or attention_improved:
            improved.append(after_score.section_name)

    return [section for section in PAGE_SECTION_NAMES if section in improved]


def _remaining_risks(state: SimulationState) -> list[str]:
    """Summarize remaining dashboard-ready risks after optimization."""
    risks: list[str] = []
    if state.final_report:
        risks.extend(state.final_report.buyer_loss_reasons[:3])
        if state.final_report.trust_risk_score >= 50:
            risks.append("Trust risk remains elevated.")
        if state.final_report.price_resistance_score >= 50:
            risks.append("Price resistance remains elevated.")
        if state.final_report.return_risk_score >= 50:
            risks.append("Shipping or return concerns remain elevated.")

    if state.attention_map:
        risks.append(
            f"Highest simulated friction remains in {state.attention_map.highest_friction_section}."
        )

    return _dedupe_short_list(risks, limit=5) or ["No major remaining risk identified."]


def _comparison_summary(before_score: int, after_score: int, score_delta: int) -> str:
    """Create a concise before-after summary using simulated score language."""
    if score_delta > 0:
        return (
            "Simulated conversion score improved after optimization; rerun with "
            "real page constraints before launch."
        )
    if score_delta < 0:
        return (
            "Simulated conversion score decreased after optimization; review the "
            "changes before using this copy."
        )
    return (
        "Simulated conversion score was unchanged; focus on remaining buyer risks "
        "before another test."
    )


def _short_list(value: Any, limit: int = 5) -> list[str]:
    """Normalize JSON list fields into concise string lists."""
    if isinstance(value, list):
        return _dedupe_short_list([str(item) for item in value], limit=limit)
    if isinstance(value, str) and value.strip():
        return [_short_text(value)]
    return []


def _dedupe_short_list(values: list[str], limit: int = 5) -> list[str]:
    """Deduplicate dashboard text while preserving order."""
    items: list[str] = []
    for value in values:
        text = _short_text(value)
        if text and text not in items:
            items.append(text)
    return items[:limit]


def _safe_score(value: Any) -> int:
    """Normalize a score into an integer from 0 to 100."""
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, score))


def _short_text(value: Any, limit: int = 180) -> str:
    """Keep optimization text concise and dashboard-ready."""
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."
