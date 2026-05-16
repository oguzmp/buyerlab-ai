"""Buyer persona simulation agents for BuyerLab AI."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from src.gemini_client import generate_json
from src.price_intelligence import (
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
        return _agent_response_from_json(persona, raw_response)
    except Exception as exc:
        return _failed_agent_response(persona, exc)


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


def _failed_agent_response(persona: BuyerPersona, exc: Exception) -> AgentResponse:
    """Create a hesitant response when one persona evaluation fails."""
    return AgentResponse(
        persona_id=persona.id,
        decision="hesitate",
        confidence=0,
        purchase_intent=0,
        main_reason=f"{persona.name} evaluation failed.",
        objections=["Simulation error"],
        missing_information=[_short_error(exc)],
        suggested_fix="Retry this persona evaluation or enable BUYERLAB_MOCK_MODE=true.",
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
        f"Group signal: {buy_count} buy, {reject_count} reject, "
        f"{hesitation_count} hesitate."
    )
    reason = response.main_reason.rstrip(".")

    if response.decision == "buy":
        return (
            f"I would buy {product.title}: {reason}. {market_signal} "
            "Strengthen the strongest proof before launch."
        )
    if response.decision == "reject":
        return (
            f"I would reject {product.title}: {reason}. {market_signal} "
            "This page needs clearer risk reduction."
        )
    return (
        f"I would hesitate on {product.title}: {reason}. {market_signal} "
        "Resolve the top objections before pushing traffic."
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


def _short_error(exc: Exception) -> str:
    """Format an exception as a short dashboard-safe message."""
    return str(exc).splitlines()[0][:160] or exc.__class__.__name__
