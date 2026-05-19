"""Local price and competitor heuristics for BuyerLab AI audits."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Optional

from src.category_intelligence import (
    apply_category_persona_weights,
    build_category_context,
    get_category_profile,
    missing_required_information,
    normalize_category,
)
from src.state import (
    CompetitorContext,
    CompetitorGapReport,
    NormalizedCategory,
    PriceBand,
    PricePerceptionReport,
    ProductInput,
)


GENERIC_PRICE_BANDS: dict[NormalizedCategory, tuple[float, float, float, float]] = {
    "electronics_accessory": (25, 90, 180, 350),
    "fashion_shoes": (30, 95, 175, 320),
    "small_home_appliance": (40, 130, 280, 600),
    "handmade_bag": (25, 90, 180, 400),
    "digital_service": (20, 100, 300, 800),
    "online_course": (30, 150, 500, 1_200),
    "general_product": (20, 80, 180, 400),
}

BAND_RISK = {
    "budget": 20,
    "mid_range": 35,
    "upper_mid": 58,
    "premium": 74,
    "irrational": 95,
}

HIGH_PROOF_BANDS = {"upper_mid", "premium", "irrational"}


def detect_price_band(price: float, currency: str, category: str) -> PriceBand:
    """Detect a heuristic price band without converting Turkish lira."""
    safe_price = max(0.0, float(price or 0.0))
    normalized_category = normalize_category(category)
    budget_max, mid_max, upper_mid_max, premium_max = _bands_for_currency(currency)[
        normalized_category
    ]

    if safe_price > premium_max:
        return "irrational"
    if safe_price > upper_mid_max:
        return "premium"
    if safe_price > mid_max:
        return "upper_mid"
    if safe_price > budget_max:
        return "mid_range"
    return "budget"


def analyze_local_price_perception(product: ProductInput) -> PricePerceptionReport:
    """Create a heuristic local price perception report, not live market data."""
    normalized_category = normalize_category(product.normalized_category or product.category)
    currency = normalize_currency(product.currency)
    price_band = detect_price_band(product.price, currency, normalized_category)
    local_market = _local_market(product, currency)
    risk = _perceived_value_risk(product, price_band)

    return PricePerceptionReport(
        currency=currency,
        price=max(0.0, float(product.price or 0.0)),
        local_market=local_market,
        normalized_category=normalized_category,
        price_band=price_band,
        perceived_value_risk=risk,
        expected_customer_questions=_expected_customer_questions(
            product,
            price_band,
            local_market,
        ),
        required_value_proofs=_required_value_proofs(product, price_band),
        pricing_comment=_pricing_comment(product, price_band, risk, local_market),
        suggested_price_positioning=_suggested_price_positioning(price_band, risk),
    )


def analyze_competitor_gap(product: ProductInput) -> CompetitorGapReport:
    """Analyze seller-provided competitor context without live competitor research."""
    competitor = product.competitor_context
    price_report = analyze_local_price_perception(product)
    required_proofs = list(price_report.required_value_proofs)

    if not _has_competitor_context(competitor):
        return CompetitorGapReport(
            value_gap_summary="No seller-provided competitor context.",
            competitor_advantage="Unknown without competitor context.",
            required_proofs_to_win=required_proofs,
            competitor_positioning_comment=price_report.suggested_price_positioning,
        )

    assert competitor is not None
    price_gap = _competitor_price_gap(product, competitor)
    unproven_claims = _unproven_competitor_claims(product)

    if competitor.competitor_strengths:
        required_proofs.append("Proof that directly addresses competitor strengths")
    if competitor.our_differentiator:
        required_proofs.append("Evidence for the stated differentiator")
    required_proofs.extend(unproven_claims)

    return CompetitorGapReport(
        price_gap=price_gap,
        value_gap_summary=_competitor_gap_summary(product, competitor, price_gap),
        competitor_advantage=_competitor_advantage(competitor),
        our_unproven_claims=unproven_claims,
        required_proofs_to_win=_dedupe(required_proofs)[:7],
        competitor_positioning_comment=_competitor_positioning_comment(
            product,
            competitor,
            price_gap,
            price_report,
        ),
    )


def analyze_competitor_context(
    product: ProductInput,
    price_report: Optional[PricePerceptionReport] = None,
) -> dict[str, Any]:
    """Backward-compatible dict wrapper for competitor gap analysis."""
    report = analyze_competitor_gap(product)
    data = asdict(report)
    data["has_competitor_context"] = _has_competitor_context(product.competitor_context)
    data["price_positioning_verdict"] = data["competitor_positioning_comment"]
    data["competitor_gap_summary"] = data["value_gap_summary"]
    data["required_price_proofs"] = data["required_proofs_to_win"]
    if price_report is not None:
        data["local_price_band"] = price_report.price_band
    return data


def build_structured_product_brief(product: ProductInput) -> dict[str, Any]:
    """Build a structured product brief for simulated buyer assessment."""
    normalized = normalize_category(product.normalized_category or product.category)
    category_context = build_category_context(product)
    price_report = analyze_local_price_perception(product)
    competitor_gap = analyze_competitor_gap(product)

    return {
        "product_identity": {
            "brand": product.brand or "Not provided",
            "model": product.model or "Not provided",
            "product_type": product.product_type or product.category or "Not provided",
            "title": product.title,
            "market_segment": product.market_segment or "Not specified",
            "intended_use_case": product.intended_use_case or product.target_audience,
            "local_market": price_report.local_market,
        },
        "normalized_category": normalized,
        "category_context": category_context,
        "price_perception": asdict(price_report),
        "competitor_context": _competitor_as_dict(product.competitor_context),
        "competitor_gap": asdict(competitor_gap),
        "proof_assets": product.proof_assets,
        "known_limitations": product.known_limitations,
        "missing_required_information": missing_required_information(product),
        "important_note": (
            "This is heuristic local price perception and simulated buyer "
            "assessment, not real market prediction or live competitor research."
        ),
    }


def build_product_brief_context(product: ProductInput) -> str:
    """Build compact prompt text from product identity, category, price, and competitor context."""
    brief = build_structured_product_brief(product)
    return (
        "Structured product brief context:\n"
        f"{json.dumps(brief, ensure_ascii=True, indent=2)}"
    )


def build_price_context_for_prompt(product: ProductInput) -> str:
    """Build Bargain Hunter prompt context from local price and competitor heuristics."""
    brief = build_structured_product_brief(product)
    price_report = brief["price_perception"]
    competitor_gap = brief["competitor_gap"]
    category_context = brief["category_context"]

    return "\n".join(
        [
            "Structured price, category, and competitor context:",
            (
                "- Method: heuristic local price perception based on static demo "
                "bands, not live market research."
            ),
            f"- Product identity: {brief['product_identity']}",
            f"- Local market: {price_report['local_market']}",
            f"- Currency: {price_report['currency']}",
            f"- Normalized category: {price_report['normalized_category']}",
            (
                "- Category expectations: "
                f"{'; '.join(category_context['typical_customer_expectations'])}"
            ),
            f"- Price: {price_report['price']:g}",
            f"- Price band: {price_report['price_band']}",
            f"- Perceived value risk: {price_report['perceived_value_risk']}/100",
            (
                "- Expected customer questions: "
                f"{'; '.join(price_report['expected_customer_questions'])}"
            ),
            f"- Required value proofs: {'; '.join(price_report['required_value_proofs'])}",
            f"- Pricing comment: {price_report['pricing_comment']}",
            f"- Suggested positioning: {price_report['suggested_price_positioning']}",
            f"- Competitor gap summary: {competitor_gap['value_gap_summary']}",
            f"- Competitor positioning: {competitor_gap['competitor_positioning_comment']}",
            f"- Known limitations: {'; '.join(brief['known_limitations']) or 'None provided'}",
        ]
    )


def normalize_currency(currency: str) -> str:
    """Normalize currency aliases while preserving non-TRY currencies."""
    normalized = (currency or "").strip().upper()
    if _is_try(normalized):
        return "TRY"
    if normalized in {"TURKISH LIRA", "TURKISH LIRASI", "LIRA"}:
        return "TRY"
    return normalized or "UNKNOWN"


def _bands_for_currency(currency: str) -> dict[NormalizedCategory, tuple[float, float, float, float]]:
    """Return TRY local bands for Turkey, otherwise a generic fallback."""
    if _is_try(currency):
        return {
            category: get_category_profile(category).try_price_bands
            for category in GENERIC_PRICE_BANDS
        }
    return GENERIC_PRICE_BANDS


def _is_try(currency: str) -> bool:
    """Return True for Turkish lira labels without doing currency conversion."""
    return (currency or "").strip().upper() in {
        "TRY",
        "TL",
        "TURKISH LIRA",
        "TURKISH LIRASI",
        "LIRA",
        "\u20ba",
    }


def _local_market(product: ProductInput, currency: str) -> str:
    """Infer local market for pricing heuristics from explicit input or currency."""
    if product.local_market.strip():
        return product.local_market.strip()
    if _is_try(currency):
        return "Turkey"
    return "generic"


def _perceived_value_risk(product: ProductInput, price_band: PriceBand) -> int:
    """Estimate how much proof buyers may need for this price point."""
    risk = BAND_RISK[price_band]
    if price_band in HIGH_PROOF_BANDS and _weak_copy(product):
        risk += 12
    if price_band in HIGH_PROOF_BANDS and not product.proof_assets:
        risk += 10
    if price_band in HIGH_PROOF_BANDS and not product.warranty_or_return_policy.strip():
        risk += 8
    if price_band in HIGH_PROOF_BANDS and not product.shipping_info.strip():
        risk += 6
    if price_band in HIGH_PROOF_BANDS and not product.trust_signals:
        risk += 8
    if product.known_limitations:
        risk += min(12, len(product.known_limitations) * 4)
    if product.competitor_context and product.competitor_context.our_differentiator:
        risk -= 5
    if product.proof_assets:
        risk -= 8
    if product.trust_signals:
        risk -= 5
    if product.warranty_or_return_policy.strip():
        risk -= 4
    return max(0, min(100, risk))


def _weak_copy(product: ProductInput) -> bool:
    """Detect when a higher price has too little value explanation."""
    combined = " ".join(
        [
            product.description or "",
            product.value_proposition or "",
            product.reviews_or_social_proof or "",
            product.intended_use_case or "",
            " ".join(product.proof_assets),
        ]
    ).strip()
    return len(combined) < 180


def _expected_customer_questions(
    product: ProductInput,
    price_band: PriceBand,
    local_market: str,
) -> list[str]:
    """Return likely buyer questions for this local price and category."""
    questions = [
        "Why is this price worth paying for this product type?",
        "What exactly is included in the total cost?",
    ]
    if _is_turkey_market(local_market):
        questions[0] = "Is this TRY price justified for this category?"
    if product.competitor_context and product.competitor_context.competitor_name:
        questions.append("Why buy this instead of the named competitor?")
    if not product.proof_assets:
        questions.append("What proof supports the key product claims?")
    if not product.shipping_info.strip():
        questions.append("How much will shipping cost and how fast is delivery?")
    if not product.warranty_or_return_policy.strip():
        questions.append("What happens if I want to return it?")
    if price_band in HIGH_PROOF_BANDS:
        questions.append("What proof or outcomes justify the higher price?")
    if product.known_limitations:
        questions.append("Are the known limitations acceptable for this price?")
    return _dedupe(questions)[:6]


def _required_value_proofs(product: ProductInput, price_band: PriceBand) -> list[str]:
    """Return proof points needed to defend the detected price band."""
    proofs = ["Clear benefit explanation", "Shipping and return terms"]
    if price_band in HIGH_PROOF_BANDS:
        proofs.extend(
            [
                "Materials, specs, deliverables, or outcome proof",
                "Authentic proof assets",
                "Warranty, guarantee, or support details",
            ]
        )
    proofs.extend(missing_required_information(product))
    if product.competitor_context and product.competitor_context.our_differentiator:
        proofs.append("Evidence for the stated competitor differentiator")
    if product.known_limitations:
        proofs.append("Clear explanation of known limitations")
    return _dedupe(proofs)[:8]


def _pricing_comment(
    product: ProductInput,
    price_band: PriceBand,
    risk: int,
    local_market: str,
) -> str:
    """Write a concise pricing comment for dashboard and prompts."""
    market_phrase = (
        "For TRY pricing, this uses Turkish local price perception heuristics."
        if _is_turkey_market(local_market)
        else "This uses a generic price perception fallback."
    )
    if price_band == "irrational":
        return (
            f"{market_phrase} The price looks unusually high for the category and "
            "needs very strong proof or repositioning."
        )
    if price_band in {"upper_mid", "premium"} and _weak_copy(product):
        return (
            f"{market_phrase} The page needs stronger value proof before buyers "
            "will accept this price."
        )
    if risk >= 65:
        return (
            f"{market_phrase} Buyers may question value unless proof, warranty, "
            "category details, and competitor differentiation are clearer."
        )
    return f"{market_phrase} The price can work if the page explains value clearly."


def _suggested_price_positioning(price_band: PriceBand, risk: int) -> str:
    """Suggest seller-facing price positioning."""
    if price_band == "irrational":
        return "Reposition as premium with strong proof, or reconsider the launch price."
    if price_band == "premium":
        return "Frame as premium and justify with proof, quality, and risk reduction."
    if price_band == "upper_mid":
        return "Position as better-than-standard value with clear differentiators."
    if price_band == "mid_range":
        return "Position as practical value and explain what buyers get for the price."
    if risk >= 50:
        return "Keep the price accessible but clarify total cost and basic proof."
    return "Position as budget-friendly while avoiding cheap-quality signals."


def _is_turkey_market(local_market: str) -> bool:
    """Detect Turkey market labels without requiring one exact spelling."""
    return (local_market or "").strip().lower() in {"turkey", "turkiye", "türkiye"}


def _has_competitor_context(competitor: Optional[CompetitorContext]) -> bool:
    """Return True when seller-provided competitor context exists."""
    if competitor is None:
        return False
    return any(
        [
            competitor.competitor_name.strip(),
            competitor.competitor_price is not None,
            competitor.competitor_strengths,
            competitor.competitor_weaknesses,
            competitor.our_differentiator.strip(),
        ]
    )


def _competitor_as_dict(competitor: Optional[CompetitorContext]) -> Optional[dict[str, Any]]:
    """Serialize optional competitor context for prompts."""
    if competitor is None:
        return None
    return asdict(competitor)


def _competitor_price_gap(
    product: ProductInput,
    competitor: CompetitorContext,
) -> Optional[float]:
    """Return same-currency price gap; do not convert currencies."""
    if competitor.competitor_price is None or competitor.competitor_price <= 0:
        return None
    product_currency = normalize_currency(product.currency)
    competitor_currency = normalize_currency(competitor.competitor_currency or product.currency)
    if product_currency != competitor_currency:
        return None
    return round(float(product.price or 0.0) - competitor.competitor_price, 2)


def _competitor_gap_summary(
    product: ProductInput,
    competitor: CompetitorContext,
    price_gap: Optional[float],
) -> str:
    """Summarize competitor value gap from user-provided context only."""
    name = competitor.competitor_name.strip() or "the provided competitor"
    currency = normalize_currency(product.currency)
    competitor_currency = normalize_currency(competitor.competitor_currency or product.currency)
    if competitor_currency != currency:
        return (
            f"Competitor price uses {competitor_currency}, while product uses {currency}; "
            "no currency conversion was performed."
        )
    if price_gap is None:
        return (
            f"Seller provided competitor context for {name}, but no usable "
            "same-currency competitor price was supplied."
        )
    if price_gap > 0:
        return (
            f"Product is priced {price_gap:g} {currency} above {name}; the page "
            "must prove the difference."
        )
    if price_gap < 0:
        return (
            f"Product is priced {abs(price_gap):g} {currency} below {name}; the "
            "page should make the value advantage obvious."
        )
    return f"Product is priced at parity with {name}; differentiation must carry the decision."


def _competitor_advantage(competitor: CompetitorContext) -> str:
    """Return the main competitor advantage from seller-provided strengths."""
    if competitor.competitor_strengths:
        return competitor.competitor_strengths[0]
    if competitor.competitor_name:
        return "Named competitor may feel safer unless differentiation is proven."
    return "Unknown without competitor context."


def _unproven_competitor_claims(product: ProductInput) -> list[str]:
    """Find differentiator claims that are not backed by proof assets."""
    competitor = product.competitor_context
    if competitor is None or not competitor.our_differentiator.strip():
        return []
    if product.proof_assets:
        return []
    return [f"Unproven differentiator: {competitor.our_differentiator.strip()}"]


def _competitor_positioning_comment(
    product: ProductInput,
    competitor: CompetitorContext,
    price_gap: Optional[float],
    report: PricePerceptionReport,
) -> str:
    """Create a short competitor positioning comment."""
    if report.price_band == "irrational":
        return "High-risk price positioning; needs premium proof or price revision."
    if price_gap is None:
        return report.suggested_price_positioning

    competitor_price = competitor.competitor_price or 0
    meaningful_gap = competitor_price * 0.15
    if price_gap > meaningful_gap:
        return "Premium versus competitor; must prove stronger value and risk reduction."
    if price_gap < -meaningful_gap:
        return "Lower than competitor; emphasize value without looking low-quality."
    return "Near competitor parity; differentiator and proof must be explicit."


def _dedupe(items: list[str]) -> list[str]:
    """Deduplicate short strings while preserving order."""
    deduped: list[str] = []
    for item in items:
        text = " ".join(str(item).split())
        if text and text not in deduped:
            deduped.append(text)
    return deduped


__all__ = [
    "apply_category_persona_weights",
    "normalize_category",
    "get_category_profile",
    "detect_price_band",
    "analyze_local_price_perception",
    "analyze_competitor_gap",
    "analyze_competitor_context",
    "build_structured_product_brief",
    "build_product_brief_context",
    "build_price_context_for_prompt",
    "normalize_currency",
]
