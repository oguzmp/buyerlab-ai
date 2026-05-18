"""Buyer persona simulation agents for BuyerLab AI."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from src.gemini_client import generate_json
from src.category_intelligence import missing_required_information
from src.price_intelligence import (
    analyze_competitor_gap,
    analyze_local_price_perception,
    build_price_context_for_prompt,
    build_structured_product_brief,
)
from src.prompts import (
    BARGAIN_HUNTER_PROMPT,
    IMPULSIVE_BUYER_PROMPT,
    SKEPTIC_PROMPT,
    TRUST_SEEKER_PROMPT,
)
from src.state import AgentResponse, BuyerPersona, DebateTurn, ProductInput


PERSONA_PROMPTS = {
    "skeptic_buyer": SKEPTIC_PROMPT,
    "bargain_hunter": BARGAIN_HUNTER_PROMPT,
    "impulsive_buyer": IMPULSIVE_BUYER_PROMPT,
    "trust_seeker": TRUST_SEEKER_PROMPT,
}

VALID_DECISIONS = {"buy", "reject", "hesitate"}


def run_persona_evaluation(product: ProductInput, persona: BuyerPersona) -> AgentResponse:
    """Evaluate a product through one buyer persona and return a safe response."""
    prompt = _build_persona_prompt(product, persona)

    try:
        raw_response = generate_json(prompt)
        if raw_response.get("mock_mode") is True:
            return _mock_agent_response(product, persona)
        return _agent_response_from_json(persona, raw_response)
    except Exception:
        # Keep the seller-facing report clean even when one live JSON call fails.
        # The fallback is deterministic and product-aware, not a technical error dump.
        return _mock_agent_response(product, persona)


def run_initial_buyer_round(
    product: ProductInput,
    personas: list[BuyerPersona],
) -> list[AgentResponse]:
    """Run the first product evaluation round for every buyer persona."""
    return [run_persona_evaluation(product, persona) for persona in personas]


def run_debate_round(
    product: ProductInput,
    personas: list[BuyerPersona],
    first_round_responses: list[AgentResponse],
) -> list[DebateTurn]:
    """Create a concise dashboard-ready debate reaction from each persona."""
    responses_by_persona = {
        response.persona_id: response for response in first_round_responses
    }
    decisions = [response.decision for response in first_round_responses]
    buy_count = decisions.count("buy")
    reject_count = decisions.count("reject")
    hesitation_count = decisions.count("hesitate")

    debate_turns: list[DebateTurn] = []
    for persona in personas:
        response = responses_by_persona.get(persona.id)
        if response is None:
            debate_turns.append(
                DebateTurn(
                    speaker=persona.name,
                    message="No first-round response was available for this persona.",
                    stance="neutral",
                )
            )
            continue

        stance = _stance_from_decision(response.decision)
        message = _build_debate_message(
            product=product,
            response=response,
            buy_count=buy_count,
            reject_count=reject_count,
            hesitation_count=hesitation_count,
        )
        debate_turns.append(
            DebateTurn(
                speaker=persona.name,
                message=message,
                stance=stance,
            )
        )

    return debate_turns


def _build_persona_prompt(product: ProductInput, persona: BuyerPersona) -> str:
    """Build a persona-specific prompt with the product context and JSON contract."""
    base_prompt = PERSONA_PROMPTS.get(persona.id, SKEPTIC_PROMPT)
    product_context = _format_product_context(product)
    persona_context = _format_persona_context(persona)
    price_context = (
        f"\n\n{build_price_context_for_prompt(product)}"
        if persona.id == "bargain_hunter"
        else ""
    )

    return f"""
{base_prompt}

Persona context:
{persona_context}

Product page context:
{product_context}
{price_context}

Output language:
- Write all user-facing JSON string values in Turkish.
- Keep schema keys and enum values exactly as requested.

Return only valid JSON for this exact engine schema:
{{
  "decision": "buy | reject | hesitate",
  "confidence": 0,
  "purchase_intent": 0,
  "main_reason": "",
  "objections": [],
  "missing_information": [],
  "suggested_fix": ""
}}

Keep every string short and dashboard-ready.
""".strip()


def _format_product_context(product: ProductInput) -> str:
    """Format product input into compact prompt context."""
    data = asdict(product)
    data["trust_signals"] = ", ".join(product.trust_signals) or "Not provided"
    brief = build_structured_product_brief(product)
    product_lines = "\n".join(
        f"- {key}: {_display_value(value)}" for key, value in data.items()
    )
    return (
        f"{product_lines}\n\n"
        "Structured product brief:\n"
        f"{json.dumps(brief, ensure_ascii=True)}"
    )


def _format_persona_context(persona: BuyerPersona) -> str:
    """Format persona input into compact prompt context."""
    return "\n".join(
        [
            f"- id: {persona.id}",
            f"- name: {persona.name}",
            f"- role: {persona.role}",
            f"- decision_style: {persona.decision_style}",
            f"- priority_factors: {', '.join(persona.priority_factors)}",
            f"- rejection_triggers: {', '.join(persona.rejection_triggers)}",
            f"- weight: {persona.weight}",
        ]
    )


def _agent_response_from_json(
    persona: BuyerPersona,
    raw_response: dict[str, Any],
) -> AgentResponse:
    """Convert Gemini JSON into an AgentResponse with safe defaults."""
    decision = _safe_decision(raw_response.get("decision"))
    confidence = _safe_score(raw_response.get("confidence"), default=50)
    purchase_intent = _safe_score(
        raw_response.get("purchase_intent"),
        default=_purchase_intent_from_decision(decision),
    )

    return AgentResponse(
        persona_id=persona.id,
        decision=decision,
        confidence=confidence,
        purchase_intent=purchase_intent,
        main_reason=_first_text(
            raw_response.get("main_reason"),
            raw_response.get("dashboard_summary"),
            raw_response.get("reason"),
            _first_list_item(raw_response.get("key_reasons")),
            _first_list_item(raw_response.get("value_reasons")),
            _first_list_item(raw_response.get("desire_triggers")),
            _first_list_item(raw_response.get("trust_signals")),
            "No main reason provided.",
        ),
        objections=_short_list(
            raw_response.get("objections"),
            raw_response.get("price_objections"),
            raw_response.get("trust_gaps"),
            raw_response.get("friction_points"),
        ),
        missing_information=_short_list(raw_response.get("missing_information")),
        suggested_fix=_first_text(
            raw_response.get("suggested_fix"),
            "Clarify the product page with stronger proof and next-step guidance.",
        ),
    )


def _mock_agent_response(product: ProductInput, persona: BuyerPersona) -> AgentResponse:
    """Create deterministic, product-aware demo output when mock mode is enabled."""
    product_name = _product_name(product)
    missing_fields = missing_required_information(product)
    price_report = analyze_local_price_perception(product)
    competitor_gap = analyze_competitor_gap(product)
    trust_is_weak = not (
        product.trust_signals
        and product.warranty_or_return_policy.strip()
        and (product.proof_assets or product.reviews_or_social_proof.strip())
    )
    copy_is_weak = not (
        product.title.strip()
        and product.value_proposition.strip()
        and product.description.strip()
        and product.call_to_action.strip()
    )

    if persona.id == "skeptic_buyer":
        objections = _short_list(missing_fields[:2], product.known_limitations[:2], limit=3)
        decision = "reject" if missing_fields or product.known_limitations else "hesitate"
        return AgentResponse(
            persona_id=persona.id,
            decision=decision,
            confidence=84 if decision == "reject" else 66,
            purchase_intent=24 if decision == "reject" else 48,
            main_reason=(
                f"{product_name} için kategori-kritik kanıtlar eksik: "
                f"{', '.join(objections[:3]) or 'somut ürün kanıtı'}."
            ),
            objections=objections or ["Ürün iddiaları daha somut kanıt gerektiriyor"],
            missing_information=missing_fields[:3],
            suggested_fix=_category_fix(product, missing_fields),
        )

    if persona.id == "bargain_hunter":
        price_gap = competitor_gap.price_gap
        competitor_name = _competitor_name(product)
        price_risk = price_report.perceived_value_risk
        decision = "reject" if price_report.price_band == "irrational" else "hesitate"
        if price_risk < 45 and not (price_gap and price_gap > 0):
            decision = "buy"
        value_reason = (
            f"{price_report.price:g} {price_report.currency} fiyatı, "
            f"{price_report.price_band} bandı için daha net değer kanıtı gerektiriyor."
        )
        if price_gap and price_gap > 0:
            value_reason = (
                f"{product_name}, {competitor_name} ürününden {price_gap:g} "
                f"{price_report.currency} daha pahalı; sayfa bu farkı kanıtlamalı."
            )
        return AgentResponse(
            persona_id=persona.id,
            decision=decision,
            confidence=82 if decision != "buy" else 68,
            purchase_intent=18 if decision == "reject" else 46 if decision == "hesitate" else 72,
            main_reason=value_reason,
            objections=_short_list(
                price_report.required_value_proofs[:2],
                competitor_gap.required_proofs_to_win[:2],
                limit=3,
            )
            or ["Fiyat kanıtı yeterince net değil"],
            missing_information=price_report.required_value_proofs[:3],
            suggested_fix="Fiyatı somut kanıt, toplam maliyet netliği ve rakip karşılaştırmasıyla savun.",
        )

    if persona.id == "impulsive_buyer":
        decision = "reject" if copy_is_weak else "hesitate"
        purchase_intent = 30 if copy_is_weak else 62
        if not copy_is_weak and product.image_notes and product.call_to_action:
            decision = "buy"
            purchase_intent = 74
        return AgentResponse(
            persona_id=persona.id,
            decision=decision,
            confidence=70,
            purchase_intent=purchase_intent,
            main_reason=(
                f"{product_name} daha net bir duygusal çekim ve CTA sunuyor."
                if decision == "buy"
                else f"{product_name} yeterli anlık istek veya görsel güven oluşturmuyor."
            ),
            objections=["Duygusal çekicilik zayıf"] if decision != "buy" else [],
            missing_information=[] if decision == "buy" else ["Daha güçlü görsel vaat", "Daha net CTA"],
            suggested_fix="İlk ekranı iddiaları abartmadan daha somut, görsel ve istek uyandıran hale getir.",
        )

    decision = "reject" if trust_is_weak else "hesitate"
    if not trust_is_weak and price_report.perceived_value_risk < 55:
        decision = "buy"
    return AgentResponse(
        persona_id=persona.id,
        decision=decision,
        confidence=86 if decision == "reject" else 72,
        purchase_intent=22 if decision == "reject" else 52 if decision == "hesitate" else 76,
        main_reason=(
            f"{product_name} henüz yeterli güven kanıtı, garanti netliği veya sosyal kanıt göstermiyor."
            if decision != "buy"
            else f"{product_name} ilk simüle değerlendirme için yeterli güven sinyali içeriyor."
        ),
        objections=["Güven kanıtı zayıf", "Garanti veya iade politikası netleşmeli"]
        if decision != "buy"
        else [],
        missing_information=["Gerçek güven sinyalleri", "Garanti ve destek detayları"]
        if decision != "buy"
        else [],
        suggested_fix="Gerçek güven sinyalleri, net garanti/iade koşulları, destek bilgisi ve kanıt varlıkları ekle.",
    )


def _build_debate_message(
    product: ProductInput,
    response: AgentResponse,
    buy_count: int,
    reject_count: int,
    hesitation_count: int,
) -> str:
    """Create a short debate message from one persona's first-round decision."""
    market_signal = (
        f"Grup sinyali: {buy_count} satın alır, {reject_count} reddeder, "
        f"{hesitation_count} kararsız."
    )
    reason = response.main_reason.rstrip(".")

    if response.decision == "buy":
        return (
            f"{product.title} için satın alma eğilimindeyim: {reason}. {market_signal} "
            "Yayına çıkmadan önce en güçlü kanıt görünür yapılmalı."
        )
    if response.decision == "reject":
        return (
            f"{product.title} ürününü reddederim: {reason}. {market_signal} "
            "Bu sayfa daha net risk azaltıcı bilgiler gerektiriyor."
        )
    return (
        f"{product.title} için kararsız kalırım: {reason}. {market_signal} "
        "Trafik göndermeden önce ana itirazlar çözülmeli."
    )


def _stance_from_decision(decision: str) -> str:
    """Map a purchase decision to a debate stance."""
    if decision == "buy":
        return "support"
    if decision == "reject":
        return "oppose"
    return "neutral"


def _safe_decision(value: Any) -> str:
    """Normalize a raw decision into the supported decision labels."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in VALID_DECISIONS:
            return normalized
        if "buy" in normalized and "reject" not in normalized:
            return "buy"
        if "reject" in normalized:
            return "reject"
    return "hesitate"


def _safe_score(value: Any, default: int = 0) -> int:
    """Normalize a score into an integer from 0 to 100."""
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = default
    return max(0, min(100, score))


def _display_value(value: Any) -> Any:
    """Format missing product values without hiding valid zero values."""
    if value is None or value == "" or value == []:
        return "Not provided"
    return value


def _product_name(product: ProductInput) -> str:
    """Return a readable product name for dashboard-ready mock output."""
    identity = " ".join(
        part
        for part in [product.brand, product.model]
        if str(part or "").strip()
    ).strip()
    return identity or product.title or product.product_type or "This product"


def _competitor_name(product: ProductInput) -> str:
    """Return a safe competitor label from seller-provided context."""
    competitor = product.competitor_context
    if competitor is not None and competitor.competitor_name.strip():
        return competitor.competitor_name.strip()
    return "the competitor"


def _category_fix(product: ProductInput, missing_fields: list[str]) -> str:
    """Suggest a concrete category-specific fix for deterministic mock mode."""
    category = product.normalized_category or product.category
    missing_text = ", ".join(missing_fields[:4]) or "category-critical details"
    if "electronics" in category or "earbud" in product.product_type.lower():
        return "Pil süresi, garanti dönemi, uyumluluk, teknik özellikler ve gerçek mikrofon/ses kanıtı ekle."
    if "fashion_shoes" in category or "shoe" in product.product_type.lower():
        return "Beden rehberi, kalıp notları, materyal detayları, değişim politikası ve gerçek ürün fotoğrafları ekle."
    if "small_home_appliance" in category or "coffee" in product.product_type.lower():
        return "Garanti, teknik özellikler, temizlik detayları, güç/kapasite bilgisi ve servis koşulları ekle."
    if "digital_service" in category or "service" in product.product_type.lower():
        return "Kapsam, teslim süresi, revizyon politikası, portfolyo kanıtı ve destek koşulları ekle."
    if "online_course" in category or "course" in product.product_type.lower():
        return "Eğitmen kanıtı, müfredat detayı, öğrenme çıktıları, örnek ders ve gerçekse iade koşulları ekle."
    return f"Şunlar için net kanıt ekle: {missing_text}."


def _purchase_intent_from_decision(decision: str) -> int:
    """Provide a safe purchase intent when Gemini omits one."""
    if decision == "buy":
        return 75
    if decision == "reject":
        return 20
    return 45


def _short_list(*values: Any, limit: int = 3) -> list[str]:
    """Return up to a few short text items from mixed JSON values."""
    items: list[str] = []
    for value in values:
        if isinstance(value, list):
            items.extend(str(item).strip() for item in value if str(item).strip())
        elif isinstance(value, str) and value.strip():
            items.append(value.strip())

    return items[:limit]


def _first_text(*values: Any) -> str:
    """Return the first non-empty string-like value."""
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "No detail provided."


def _first_list_item(value: Any) -> str | None:
    """Return the first string item from a list-like response field."""
    if isinstance(value, list) and value:
        first_item = str(value[0]).strip()
        return first_item or None
    return None

