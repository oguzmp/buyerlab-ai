"""AI-simulated attention and conversion friction scoring for BuyerLab AI."""

from __future__ import annotations

import json
from dataclasses import asdict
from statistics import mean
from typing import Any

from src.gemini_client import generate_json
from src.state import (
    AgentResponse,
    AttentionMapReport,
    PAGE_SECTION_NAMES,
    PageSectionScore,
    ProductInput,
)


HIGH_SCORE_THRESHOLD = 60


SECTION_FIELD_MAP = {
    "title": "title",
    "price": "price",
    "hero_image": "image_notes",
    "description": "description",
    "value_proposition": "value_proposition",
    "warranty_or_return_policy": "warranty_or_return_policy",
    "shipping_info": "shipping_info",
    "trust_signals": "trust_signals",
    "reviews_or_social_proof": "reviews_or_social_proof",
    "call_to_action": "call_to_action",
}


SECTION_KEYWORDS = {
    "title": ["title", "headline", "name"],
    "price": ["price", "cost", "value", "worth", "discount"],
    "hero_image": ["image", "visual", "photo", "hero"],
    "description": ["description", "detail", "unclear", "spec"],
    "value_proposition": ["value", "benefit", "why", "proposition"],
    "warranty_or_return_policy": ["warranty", "return", "refund", "guarantee"],
    "shipping_info": ["shipping", "delivery"],
    "trust_signals": ["trust", "credibility", "proof", "secure"],
    "reviews_or_social_proof": ["review", "social proof", "testimonial", "rating"],
    "call_to_action": ["cta", "call to action", "buy", "urgency", "checkout"],
}


def build_attention_map_prompt(
    product: ProductInput,
    agent_responses: list[AgentResponse],
    buyer_loss_analysis: list[dict[str, Any]] | None = None,
) -> str:
    """Build a prompt for AI-simulated attention and friction scoring."""
    context = {
        "important_note": (
            "This is AI-simulated buyer attention and conversion friction analysis, "
            "not real eye-tracking or analytics data."
        ),
        "product": _product_context(product),
        "agent_responses": [_response_as_dict(response) for response in agent_responses],
        "buyer_loss_analysis": buyer_loss_analysis or [],
        "sections_to_score": PAGE_SECTION_NAMES,
    }

    return f"""
You are creating an AI-simulated buyer attention and conversion friction map for
BuyerLab AI. This is not real eye-tracking. Do not claim users actually looked
at any section.

Evaluate how the buyer personas would react to each product page section.

Context:
{json.dumps(context, ensure_ascii=True, indent=2)}

Return only valid JSON with this exact shape:
{{
  "section_scores": [
    {{
      "section_name": "title",
      "attention_score": 0,
      "friction_score": 0,
      "sentiment": "positive | neutral | negative",
      "reason": "",
      "suggested_fix": ""
    }}
  ]
}}

Rules:
- Include exactly one score for each section in sections_to_score.
- Scores must be integers from 0 to 100.
- Use sentiment values: positive, neutral, or negative.
- Keep reason and suggested_fix short enough for dashboard cards.
- Do not invent real analytics, heatmap, click, scroll, gaze, or eye-tracking data.
""".strip()


def generate_attention_map(
    product: ProductInput,
    agent_responses: list[AgentResponse],
    buyer_loss_analysis: list[dict[str, Any]] | None = None,
) -> AttentionMapReport:
    """Generate an AI-simulated attention and conversion friction report."""
    fallback_scores = _fallback_section_scores(
        product,
        agent_responses,
        buyer_loss_analysis,
    )

    try:
        raw_response = generate_json(
            build_attention_map_prompt(product, agent_responses, buyer_loss_analysis)
        )
        section_scores = _section_scores_from_json(raw_response, fallback_scores)
    except Exception:
        section_scores = fallback_scores

    return aggregate_section_scores(section_scores)


def aggregate_section_scores(section_scores: list[PageSectionScore]) -> AttentionMapReport:
    """Aggregate section-level or persona-level scores into one final report."""
    normalized_scores = _ensure_all_sections(section_scores)
    grouped_scores = _group_scores_by_section(normalized_scores)
    aggregated_scores = [
        _aggregate_one_section(section_name, grouped_scores[section_name])
        for section_name in PAGE_SECTION_NAMES
    ]

    strongest = max(
        aggregated_scores,
        key=lambda score: score.attention_score - score.friction_score,
    )
    weakest = min(
        aggregated_scores,
        key=lambda score: score.attention_score - score.friction_score,
    )
    highest_friction = max(aggregated_scores, key=lambda score: score.friction_score)

    return AttentionMapReport(
        section_scores=aggregated_scores,
        strongest_section=strongest.section_name,
        weakest_section=weakest.section_name,
        highest_friction_section=highest_friction.section_name,
        summary=(
            "AI-simulated buyer attention highlights "
            f"{strongest.section_name}; conversion friction is highest in "
            f"{highest_friction.section_name}."
        ),
    )


def get_section_priority_label(attention_score: int, friction_score: int) -> str:
    """Label a section by simulated attention and conversion friction."""
    high_attention = _safe_score(attention_score) >= HIGH_SCORE_THRESHOLD
    high_friction = _safe_score(friction_score) >= HIGH_SCORE_THRESHOLD

    if high_attention and not high_friction:
        return "strong_conversion_area"
    if high_attention and high_friction:
        return "critical_fix_area"
    if not high_attention and high_friction:
        return "hidden_risk_area"
    return "low_priority_area"


def _section_scores_from_json(
    raw_response: dict[str, Any],
    fallback_scores: list[PageSectionScore],
) -> list[PageSectionScore]:
    """Parse Gemini JSON into section scores, filling incomplete sections safely."""
    raw_scores = raw_response.get("section_scores")
    if raw_scores is None and isinstance(raw_response.get("attention_map"), dict):
        raw_scores = raw_response["attention_map"].get("section_scores")
    if raw_scores is None:
        raw_scores = raw_response.get("sections")
    if not isinstance(raw_scores, list):
        raise ValueError("Attention map response is missing section_scores.")

    parsed_scores: list[PageSectionScore] = []
    for raw_score in raw_scores:
        parsed_score = _page_section_score_from_json(raw_score)
        if parsed_score is not None:
            parsed_scores.append(parsed_score)

    if not parsed_scores:
        raise ValueError("Attention map response did not contain usable section scores.")

    fallback_by_section = {score.section_name: score for score in fallback_scores}
    parsed_by_section = {score.section_name: score for score in parsed_scores}
    return [
        parsed_by_section.get(section_name, fallback_by_section[section_name])
        for section_name in PAGE_SECTION_NAMES
    ]


def _page_section_score_from_json(raw_score: Any) -> PageSectionScore | None:
    """Convert one raw JSON section score into a safe PageSectionScore."""
    if not isinstance(raw_score, dict):
        return None

    section_name = _normalize_section_name(raw_score.get("section_name"))
    if section_name not in PAGE_SECTION_NAMES:
        return None

    return PageSectionScore(
        section_name=section_name,
        attention_score=_safe_score(raw_score.get("attention_score"), default=50),
        friction_score=_safe_score(raw_score.get("friction_score"), default=50),
        sentiment=_safe_sentiment(raw_score.get("sentiment")),
        reason=_short_text(raw_score.get("reason"), "Simulated buyer signal."),
        suggested_fix=_short_text(raw_score.get("suggested_fix"), "Clarify this section."),
    )


def _fallback_section_scores(
    product: ProductInput,
    agent_responses: list[AgentResponse],
    buyer_loss_analysis: list[dict[str, Any]] | None,
) -> list[PageSectionScore]:
    """Create safe deterministic section scores when Gemini output is unavailable."""
    buyer_text = _combined_buyer_text(agent_responses, buyer_loss_analysis)
    average_intent = _average_purchase_intent(agent_responses)

    scores: list[PageSectionScore] = []
    for section_name in PAGE_SECTION_NAMES:
        content = _section_content(product, section_name)
        has_content = bool(content)
        keyword_hits = _keyword_hits(section_name, buyer_text)
        attention_score = _section_attention(section_name, has_content, average_intent)
        friction_score = _section_friction(section_name, has_content, keyword_hits)
        sentiment = _sentiment_from_scores(attention_score, friction_score)

        scores.append(
            PageSectionScore(
                section_name=section_name,
                attention_score=attention_score,
                friction_score=friction_score,
                sentiment=sentiment,
                reason=_fallback_reason(section_name, has_content, keyword_hits),
                suggested_fix=_fallback_suggested_fix(section_name, has_content, keyword_hits),
            )
        )

    return scores


def _section_attention(section_name: str, has_content: bool, average_intent: int) -> int:
    """Estimate simulated section attention from content presence and buyer intent."""
    baseline_by_section = {
        "title": 82,
        "price": 78,
        "hero_image": 74,
        "description": 64,
        "value_proposition": 72,
        "warranty_or_return_policy": 55,
        "shipping_info": 52,
        "trust_signals": 66,
        "reviews_or_social_proof": 62,
        "call_to_action": 76,
    }
    baseline = baseline_by_section.get(section_name, 55)
    content_adjustment = 0 if has_content else -25
    intent_adjustment = round((average_intent - 50) * 0.2)
    return _safe_score(baseline + content_adjustment + intent_adjustment)


def _section_friction(section_name: str, has_content: bool, keyword_hits: int) -> int:
    """Estimate simulated conversion friction for one section."""
    base_friction = 25 if has_content else 68
    critical_section_adjustment = 10 if section_name in {
        "price",
        "value_proposition",
        "warranty_or_return_policy",
        "shipping_info",
        "trust_signals",
        "reviews_or_social_proof",
        "call_to_action",
    } else 0
    return _safe_score(base_friction + critical_section_adjustment + keyword_hits * 14)


def _aggregate_one_section(
    section_name: str,
    scores: list[PageSectionScore],
) -> PageSectionScore:
    """Aggregate multiple scores for the same section."""
    attention_score = _safe_score(round(mean(score.attention_score for score in scores)))
    friction_score = _safe_score(round(mean(score.friction_score for score in scores)))
    sentiment = _sentiment_from_scores(attention_score, friction_score)
    highest_friction_score = max(scores, key=lambda score: score.friction_score)

    return PageSectionScore(
        section_name=section_name,
        attention_score=attention_score,
        friction_score=friction_score,
        sentiment=sentiment,
        reason=_short_text(
            highest_friction_score.reason,
            "AI-simulated buyer attention signal.",
        ),
        suggested_fix=_short_text(
            highest_friction_score.suggested_fix,
            "Clarify this section.",
        ),
    )


def _ensure_all_sections(section_scores: list[PageSectionScore]) -> list[PageSectionScore]:
    """Ensure every required page section has at least one safe score."""
    scores_by_section: dict[str, list[PageSectionScore]] = {
        section_name: [] for section_name in PAGE_SECTION_NAMES
    }
    for score in section_scores:
        if score.section_name in scores_by_section:
            scores_by_section[score.section_name].append(score)

    complete_scores: list[PageSectionScore] = []
    for section_name in PAGE_SECTION_NAMES:
        complete_scores.extend(
            scores_by_section[section_name]
            or [
                PageSectionScore(
                    section_name=section_name,
                    attention_score=40,
                    friction_score=40,
                    sentiment="neutral",
                    reason="No simulated buyer section signal available.",
                    suggested_fix="Review this section for clarity.",
                )
            ]
        )
    return complete_scores


def _group_scores_by_section(
    section_scores: list[PageSectionScore],
) -> dict[str, list[PageSectionScore]]:
    """Group section scores by section name."""
    grouped_scores: dict[str, list[PageSectionScore]] = {
        section_name: [] for section_name in PAGE_SECTION_NAMES
    }
    for score in section_scores:
        grouped_scores[score.section_name].append(score)
    return grouped_scores


def _product_context(product: ProductInput) -> dict[str, Any]:
    """Serialize product input for prompt context."""
    product_context = asdict(product)
    product_context["trust_signals"] = product.trust_signals
    return product_context


def _response_as_dict(response: Any) -> dict[str, Any]:
    """Serialize agent responses defensively."""
    if isinstance(response, dict):
        return dict(response)
    try:
        return asdict(response)
    except TypeError:
        return {
            "persona_id": getattr(response, "persona_id", ""),
            "decision": getattr(response, "decision", "hesitate"),
            "purchase_intent": getattr(response, "purchase_intent", 0),
            "main_reason": getattr(response, "main_reason", ""),
        }


def _combined_buyer_text(
    agent_responses: list[AgentResponse],
    buyer_loss_analysis: list[dict[str, Any]] | None,
) -> str:
    """Combine buyer objections and loss analysis into searchable text."""
    response_parts: list[str] = []
    for response in agent_responses:
        response_parts.extend(
            [
                getattr(response, "main_reason", ""),
                " ".join(getattr(response, "objections", [])),
                " ".join(getattr(response, "missing_information", [])),
                getattr(response, "suggested_fix", ""),
            ]
        )

    for loss in buyer_loss_analysis or []:
        response_parts.extend(
            [
                str(loss.get("main_loss_reason", "")),
                " ".join(str(item) for item in loss.get("objections", [])),
                " ".join(str(item) for item in loss.get("missing_information", [])),
                str(loss.get("suggested_fix", "")),
            ]
        )

    return " ".join(part for part in response_parts if part).lower()


def _average_purchase_intent(agent_responses: list[AgentResponse]) -> int:
    """Return average purchase intent with a neutral fallback."""
    if not agent_responses:
        return 50
    intents = [
        _safe_score(getattr(response, "purchase_intent", 50), default=50)
        for response in agent_responses
    ]
    return _safe_score(round(mean(intents)), default=50)


def _section_content(product: ProductInput, section_name: str) -> Any:
    """Read the content represented by one product page section."""
    value = getattr(product, SECTION_FIELD_MAP[section_name])
    if isinstance(value, list):
        return [item for item in value if str(item).strip()]
    return value


def _keyword_hits(section_name: str, buyer_text: str) -> int:
    """Count capped keyword hits for a section in buyer feedback."""
    return min(
        3,
        sum(1 for keyword in SECTION_KEYWORDS[section_name] if keyword in buyer_text),
    )


def _fallback_reason(section_name: str, has_content: bool, keyword_hits: int) -> str:
    """Create a short fallback reason for a section score."""
    if not has_content:
        return f"{section_name} lacks enough page content for simulated buyers."
    if keyword_hits:
        return f"Buyer feedback points to friction around {section_name}."
    return f"{section_name} has limited simulated friction."


def _fallback_suggested_fix(section_name: str, has_content: bool, keyword_hits: int) -> str:
    """Create a short fallback improvement suggestion."""
    if not has_content:
        return f"Add clear {section_name} content before launch."
    if keyword_hits:
        return f"Reduce buyer uncertainty in {section_name}."
    return f"Keep {section_name} concise and easy to scan."


def _sentiment_from_scores(attention_score: int, friction_score: int) -> str:
    """Convert attention and friction scores into section sentiment."""
    if friction_score >= 65:
        return "negative"
    if attention_score >= 60 and friction_score <= 45:
        return "positive"
    return "neutral"


def _safe_score(value: Any, default: int = 0) -> int:
    """Normalize a score into an integer from 0 to 100."""
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = default
    return max(0, min(100, score))


def _safe_sentiment(value: Any) -> str:
    """Normalize section sentiment into the supported labels."""
    if isinstance(value, str) and value.strip().lower() in {"positive", "neutral", "negative"}:
        return value.strip().lower()
    return "neutral"


def _normalize_section_name(value: Any) -> str:
    """Normalize a raw section name from Gemini output."""
    if not isinstance(value, str):
        return ""
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _short_text(value: Any, default: str, limit: int = 120) -> str:
    """Keep dashboard text compact and single-line."""
    text = " ".join(str(value or default).split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."
