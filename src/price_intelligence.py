"""Category normalization and local price perception heuristics for BuyerLab AI."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import Any, cast

from src.state import (
    BuyerPersona,
    CategoryProfile,
    CompetitorContext,
    NormalizedCategory,
    PriceBand,
    PricePerceptionReport,
    ProductInput,
)


# These category profiles are static demo heuristics for pre-launch diagnostics.
# They are not live market data, competitor research, or currency conversion.
CATEGORY_PROFILES: dict[NormalizedCategory, CategoryProfile] = {
    "electronics_accessory": CategoryProfile(
        normalized_category="electronics_accessory",
        display_name="Electronics accessory",
        typical_customer_expectations=[
            "clear specs and compatibility",
            "battery or durability claims with proof",
            "warranty and return safety",
            "reviews or technical proof",
        ],
        critical_information_fields=[
            "description",
            "warranty_or_return_policy",
            "trust_signals",
            "reviews_or_social_proof",
            "shipping_info",
        ],
        default_persona_weights={
            "skeptic_buyer": 1.2,
            "bargain_hunter": 1.1,
            "impulsive_buyer": 0.8,
            "trust_seeker": 1.1,
        },
        try_price_bands=(750, 2_000, 4_500, 9_000),
    ),
    "fashion_shoes": CategoryProfile(
        normalized_category="fashion_shoes",
        display_name="Fashion shoes",
        typical_customer_expectations=[
            "fit and sizing clarity",
            "material and comfort details",
            "easy returns",
            "strong product photos",
        ],
        critical_information_fields=[
            "description",
            "image_notes",
            "warranty_or_return_policy",
            "shipping_info",
            "reviews_or_social_proof",
        ],
        default_persona_weights={
            "skeptic_buyer": 0.9,
            "bargain_hunter": 1.1,
            "impulsive_buyer": 1.15,
            "trust_seeker": 1.0,
        },
        try_price_bands=(900, 2_500, 4_500, 8_000),
    ),
    "small_home_appliance": CategoryProfile(
        normalized_category="small_home_appliance",
        display_name="Small home appliance",
        typical_customer_expectations=[
            "safety and usage details",
            "warranty and service clarity",
            "delivery and return confidence",
            "proof of quality",
        ],
        critical_information_fields=[
            "description",
            "warranty_or_return_policy",
            "shipping_info",
            "trust_signals",
            "reviews_or_social_proof",
        ],
        default_persona_weights={
            "skeptic_buyer": 1.2,
            "bargain_hunter": 1.0,
            "impulsive_buyer": 0.8,
            "trust_seeker": 1.2,
        },
        try_price_bands=(1_200, 3_500, 7_500, 15_000),
    ),
    "handmade_bag": CategoryProfile(
        normalized_category="handmade_bag",
        display_name="Handmade bag",
        typical_customer_expectations=[
            "materials and craftsmanship",
            "size and capacity details",
            "authentic maker story",
            "clear return expectations",
        ],
        critical_information_fields=[
            "description",
            "value_proposition",
            "image_notes",
            "warranty_or_return_policy",
            "trust_signals",
        ],
        default_persona_weights={
            "skeptic_buyer": 1.0,
            "bargain_hunter": 0.9,
            "impulsive_buyer": 1.2,
            "trust_seeker": 1.1,
        },
        try_price_bands=(700, 2_000, 4_500, 9_000),
    ),
    "digital_service": CategoryProfile(
        normalized_category="digital_service",
        display_name="Digital service",
        typical_customer_expectations=[
            "clear deliverables",
            "transparent recurring costs",
            "support and cancellation details",
            "credible outcome proof",
        ],
        critical_information_fields=[
            "description",
            "value_proposition",
            "trust_signals",
            "reviews_or_social_proof",
            "call_to_action",
        ],
        default_persona_weights={
            "skeptic_buyer": 1.15,
            "bargain_hunter": 1.15,
            "impulsive_buyer": 0.8,
            "trust_seeker": 1.1,
        },
        try_price_bands=(500, 2_000, 6_000, 15_000),
    ),
    "online_course": CategoryProfile(
        normalized_category="online_course",
        display_name="Online course",
        typical_customer_expectations=[
            "curriculum and learning outcomes",
            "instructor credibility",
            "time commitment",
            "refund or satisfaction policy",
        ],
        critical_information_fields=[
            "description",
            "value_proposition",
            "warranty_or_return_policy",
            "trust_signals",
            "reviews_or_social_proof",
        ],
        default_persona_weights={
            "skeptic_buyer": 1.2,
            "bargain_hunter": 1.1,
            "impulsive_buyer": 0.85,
            "trust_seeker": 1.15,
        },
        try_price_bands=(750, 2_500, 7_000, 20_000),
    ),
    "general_product": CategoryProfile(
        normalized_category="general_product",
        display_name="General product",
        typical_customer_expectations=[
            "clear value proposition",
            "shipping and return clarity",
            "basic product proof",
            "trustworthy checkout signals",
        ],
        critical_information_fields=[
            "description",
            "value_proposition",
            "warranty_or_return_policy",
            "shipping_info",
            "trust_signals",
        ],
        default_persona_weights={
            "skeptic_buyer": 1.0,
            "bargain_hunter": 1.0,
            "impulsive_buyer": 1.0,
            "trust_seeker": 1.0,
        },
        try_price_bands=(500, 1_500, 3_500, 8_000),
    ),
}

GENERIC_PRICE_BANDS: dict[NormalizedCategory, tuple[float, float, float, float]] = {
    "electronics_accessory": (25, 90, 180, 350),
    "fashion_shoes": (30, 95, 175, 320),
    "small_home_appliance": (40, 130, 280, 600),
    "handmade_bag": (25, 90, 180, 400),
    "digital_service": (20, 100, 300, 800),
    "online_course": (30, 150, 500, 1_200),
    "general_product": (20, 80, 180, 400),
}

CATEGORY_KEYWORDS = {
    "electronics_accessory": [
        "audio",
        "earbud",
        "headphone",
        "electronics",
        "phone",
        "charger",
        "accessory",
        "consumer electronics",
    ],
    "fashion_shoes": [
        "shoe",
        "sneaker",
        "footwear",
        "running",
        "fashion",
    ],
    "small_home_appliance": [
        "appliance",
        "kettle",
        "blender",
        "coffee",
        "home appliance",
        "kitchen",
    ],
    "handmade_bag": [
        "bag",
        "tote",
        "handmade",
        "leather",
        "purse",
    ],
    "digital_service": [
        "digital service",
        "subscription",
        "saas",
        "software",
        "service",
    ],
    "online_course": [
        "course",
        "bootcamp",
        "training",
        "workshop",
        "education",
    ],
}

BAND_RISK = {
    "budget": 20,
    "mid_range": 35,
    "upper_mid": 58,
    "premium": 74,
    "irrational": 95,
}

HIGH_PROOF_BANDS = {"upper_mid", "premium", "irrational"}


def normalize_category(category: str) -> NormalizedCategory:
    """Normalize free-form category text into a supported BuyerLab category."""
    text = (category or "").strip().lower()
    if not text:
        return "general_product"

    if text in CATEGORY_PROFILES:
        return cast(NormalizedCategory, text)

    for normalized, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return cast(NormalizedCategory, normalized)

    return "general_product"


def get_category_profile(category: str) -> CategoryProfile:
    """Return the category profile for free-form or normalized category text."""
    return CATEGORY_PROFILES[normalize_category(category)]


def apply_category_persona_weights(
    personas: list[BuyerPersona],
    category: str,
) -> list[BuyerPersona]:
    """Apply normalized category persona weights without mutating the source personas."""
    weights = get_category_profile(category).default_persona_weights
    return [
        replace(persona, weight=weights.get(persona.id, persona.weight))
        for persona in personas
    ]


def detect_price_band(price: float, currency: str, category: str) -> PriceBand:
    """Detect a simulated price band without converting Turkish lira to another currency."""
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
    """Create a dashboard-ready heuristic local price perception report."""
    normalized_category = normalize_category(product.category)
    currency = normalize_currency(product.currency)
    price_band = detect_price_band(product.price, currency, normalized_category)
    local_market = "Turkey" if _is_try(currency) else "generic"
    risk = _perceived_value_risk(product, price_band)
    expected_questions = _expected_customer_questions(product, price_band, local_market)
    required_proofs = _required_value_proofs(product, price_band)

    return PricePerceptionReport(
        currency=currency,
        price=max(0.0, float(product.price or 0.0)),
        local_market=local_market,
        normalized_category=normalized_category,
        price_band=price_band,
        perceived_value_risk=risk,
        expected_customer_questions=expected_questions,
        required_value_proofs=required_proofs,
        pricing_comment=_pricing_comment(product, price_band, risk, local_market),
        suggested_price_positioning=_suggested_price_positioning(price_band, risk),
    )


def build_structured_product_brief(product: ProductInput) -> dict[str, Any]:
    """Build a structured business brief for agent prompts and judge context."""
    profile = get_category_profile(product.category)
    price_report = analyze_local_price_perception(product)
    competitor_analysis = analyze_competitor_context(product, price_report)

    return {
        "normalized_category": profile.normalized_category,
        "category_display_name": profile.display_name,
        "typical_customer_expectations": profile.typical_customer_expectations,
        "critical_information_fields": profile.critical_information_fields,
        "missing_critical_information": _missing_critical_information(product, profile),
        "persona_weights": profile.default_persona_weights,
        "try_price_bands": {
            "budget_max": profile.try_price_bands[0],
            "mid_range_max": profile.try_price_bands[1],
            "upper_mid_max": profile.try_price_bands[2],
            "premium_max": profile.try_price_bands[3],
        },
        "price_perception": asdict(price_report),
        "competitor_context": _competitor_as_dict(product.competitor_context),
        "competitor_analysis": competitor_analysis,
        "important_note": (
            "This is heuristic local price perception and seller-provided "
            "competitor context, not live market research."
        ),
    }


def build_price_context_for_prompt(product: ProductInput) -> str:
    """Build Bargain Hunter prompt context from local price and category heuristics."""
    brief = build_structured_product_brief(product)
    price_report = brief["price_perception"]
    competitor_analysis = brief["competitor_analysis"]

    return "\n".join(
        [
            "Structured price and category context:",
            (
                "- Method: heuristic local price perception based on static demo "
                "bands, not live market research."
            ),
            f"- Local market: {price_report['local_market']}",
            f"- Currency: {price_report['currency']}",
            f"- Normalized category: {price_report['normalized_category']}",
            f"- Category expectations: {'; '.join(brief['typical_customer_expectations'])}",
            f"- Critical fields: {'; '.join(brief['critical_information_fields'])}",
            f"- Missing critical fields: {'; '.join(brief['missing_critical_information']) or 'None'}",
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
            f"- Competitor gap summary: {competitor_analysis['competitor_gap_summary']}",
            f"- Price positioning verdict: {competitor_analysis['price_positioning_verdict']}",
            f"- JSON brief: {json.dumps(brief, ensure_ascii=True)}",
        ]
    )


def analyze_competitor_context(
    product: ProductInput,
    price_report: PricePerceptionReport | None = None,
) -> dict[str, Any]:
    """Analyze seller-provided competitor context without claiming live pricing data."""
    report = price_report or analyze_local_price_perception(product)
    competitor = product.competitor_context
    required_price_proofs = list(report.required_value_proofs)

    if not _has_competitor_context(competitor):
        return {
            "has_competitor_context": False,
            "price_positioning_verdict": report.suggested_price_positioning,
            "competitor_gap_summary": "No seller-provided competitor context.",
            "required_price_proofs": required_price_proofs,
        }

    assert competitor is not None
    price_gap = _competitor_price_gap(product, competitor)
    summary = _competitor_gap_summary(product, competitor, price_gap)
    verdict = _price_positioning_verdict(product, competitor, price_gap, report)

    if competitor.competitor_strengths:
        required_price_proofs.append("Explain why your offer wins against competitor strengths")
    if competitor.our_differentiator:
        required_price_proofs.append("Make the stated differentiator visible near the price")

    return {
        "has_competitor_context": True,
        "price_positioning_verdict": verdict,
        "competitor_gap_summary": summary,
        "required_price_proofs": _dedupe(required_price_proofs)[:6],
        "competitor_name": competitor.competitor_name,
        "competitor_price": competitor.competitor_price,
        "our_price": product.price,
        "price_gap": price_gap,
        "our_differentiator": competitor.our_differentiator,
    }


def normalize_currency(currency: str) -> str:
    """Normalize accepted currency aliases while preserving non-TRY currencies."""
    normalized = (currency or "").strip().upper()
    if _is_try(normalized):
        return "TRY"
    return normalized or "UNKNOWN"


def _bands_for_currency(currency: str) -> dict[NormalizedCategory, tuple[float, float, float, float]]:
    """Return TRY bands for Turkey, otherwise a generic non-live fallback."""
    if _is_try(currency):
        return {
            category: profile.try_price_bands
            for category, profile in CATEGORY_PROFILES.items()
        }
    return GENERIC_PRICE_BANDS


def _is_try(currency: str) -> bool:
    """Return True for common Turkish lira currency labels."""
    return (currency or "").strip().upper() in {"TRY", "TL", "\u20ba"}


def _perceived_value_risk(product: ProductInput, price_band: PriceBand) -> int:
    """Estimate how much value proof a buyer may need at this price point."""
    risk = BAND_RISK[price_band]
    if price_band in HIGH_PROOF_BANDS and _weak_copy(product):
        risk += 12
    if price_band in HIGH_PROOF_BANDS and not product.warranty_or_return_policy.strip():
        risk += 8
    if price_band in HIGH_PROOF_BANDS and not product.shipping_info.strip():
        risk += 6
    if price_band in HIGH_PROOF_BANDS and not product.trust_signals:
        risk += 8
    if product.competitor_context and product.competitor_context.our_differentiator:
        risk -= 5
    if product.trust_signals:
        risk -= 5
    if product.warranty_or_return_policy.strip():
        risk -= 4
    if product.shipping_info.strip():
        risk -= 3
    return max(0, min(100, risk))


def _weak_copy(product: ProductInput) -> bool:
    """Detect when upper price bands have too little value explanation."""
    combined = " ".join(
        [
            product.description or "",
            product.value_proposition or "",
            product.reviews_or_social_proof or "",
        ]
    ).strip()
    return len(combined) < 160


def _expected_customer_questions(
    product: ProductInput,
    price_band: PriceBand,
    local_market: str,
) -> list[str]:
    """Return concise buyer questions likely to appear for this price point."""
    questions = [
        "Why is this price worth paying for this category?",
        "What exactly is included in the total cost?",
    ]

    if local_market == "Turkey":
        questions[0] = "Is this TRY price justified for this category?"
    if product.competitor_context and product.competitor_context.competitor_name:
        questions.append("Why buy this instead of the named competitor?")
    if not product.shipping_info.strip():
        questions.append("How much will shipping cost and how fast is delivery?")
    if not product.warranty_or_return_policy.strip():
        questions.append("What happens if I want to return it?")
    if price_band in HIGH_PROOF_BANDS:
        questions.append("What proof or outcomes justify the higher price?")
    if price_band == "irrational":
        questions.append("Is this price intentionally premium or a pricing mistake?")

    return _dedupe(questions)[:5]


def _required_value_proofs(product: ProductInput, price_band: PriceBand) -> list[str]:
    """Return proof points needed to support the detected price band."""
    profile = get_category_profile(product.category)
    proofs = ["Clear benefit explanation", "Shipping and return terms"]
    if price_band in HIGH_PROOF_BANDS:
        proofs.extend(
            [
                "Materials, specs, deliverables, or outcome proof",
                "Authentic reviews or buyer proof",
                "Warranty, guarantee, or support details",
            ]
        )
    proofs.extend(_proofs_for_missing_fields(profile, product))
    if product.trust_signals:
        proofs.append("Make existing trust signals visible near the price")

    return _dedupe(proofs)[:6]


def _pricing_comment(
    product: ProductInput,
    price_band: PriceBand,
    risk: int,
    local_market: str,
) -> str:
    """Write a short, dashboard-safe pricing comment."""
    market_phrase = (
        "For TRY pricing, this uses Turkish local price perception heuristics."
        if local_market == "Turkey"
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
            "shipping, and category details are clearer."
        )
    return f"{market_phrase} The price can work if the page explains value clearly."


def _suggested_price_positioning(price_band: PriceBand, risk: int) -> str:
    """Suggest concise seller-facing price positioning."""
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


def _missing_critical_information(
    product: ProductInput,
    profile: CategoryProfile,
) -> list[str]:
    """Return category-critical information fields that appear weak or missing."""
    missing = [
        field_name
        for field_name in profile.critical_information_fields
        if _field_is_missing(product, field_name)
    ]
    return missing[:6]


def _proofs_for_missing_fields(
    profile: CategoryProfile,
    product: ProductInput,
) -> list[str]:
    """Translate missing category fields into price-proof requirements."""
    proof_map = {
        "description": "Concrete product details and category-specific specs",
        "value_proposition": "Clear reason to choose this offer",
        "warranty_or_return_policy": "Warranty, return, or refund policy",
        "shipping_info": "Shipping cost and delivery timing",
        "trust_signals": "Visible trust signals near price and CTA",
        "reviews_or_social_proof": "Authentic reviews, testimonials, or proof",
        "image_notes": "Visual proof of fit, quality, scale, or usage",
        "call_to_action": "Clear next step and purchase expectation",
    }
    return [
        proof_map.get(field_name, field_name.replace("_", " "))
        for field_name in _missing_critical_information(product, profile)
    ]


def _field_is_missing(product: ProductInput, field_name: str) -> bool:
    """Check a ProductInput field for empty or weak content."""
    value = getattr(product, field_name, None)
    if value is None:
        return True
    if isinstance(value, list):
        return len(value) == 0
    if isinstance(value, str):
        if field_name in {"description", "value_proposition"}:
            return len(value.strip()) < 20
        return not value.strip()
    return False


def _has_competitor_context(competitor: CompetitorContext | None) -> bool:
    """Return True when any seller-provided competitor signal exists."""
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


def _competitor_as_dict(competitor: CompetitorContext | None) -> dict[str, Any] | None:
    """Serialize optional competitor context for prompt use."""
    if competitor is None:
        return None
    return asdict(competitor)


def _competitor_price_gap(
    product: ProductInput,
    competitor: CompetitorContext,
) -> float | None:
    """Return same-currency seller-provided price gap when competitor price exists."""
    if competitor.competitor_price is None or competitor.competitor_price <= 0:
        return None
    return round(float(product.price or 0.0) - competitor.competitor_price, 2)


def _competitor_gap_summary(
    product: ProductInput,
    competitor: CompetitorContext,
    price_gap: float | None,
) -> str:
    """Summarize competitor value gap without claiming live market coverage."""
    name = competitor.competitor_name.strip() or "the provided competitor"
    if price_gap is None:
        return (
            f"Seller provided competitor context for {name}, but no usable "
            "competitor price was supplied."
        )
    if price_gap > 0:
        return (
            f"Product is priced {price_gap:g} {normalize_currency(product.currency)} "
            f"above {name}; the page must prove the difference."
        )
    if price_gap < 0:
        return (
            f"Product is priced {abs(price_gap):g} {normalize_currency(product.currency)} "
            f"below {name}; the page should make the value advantage obvious."
        )
    return f"Product is priced at parity with {name}; differentiation must carry the decision."


def _price_positioning_verdict(
    product: ProductInput,
    competitor: CompetitorContext,
    price_gap: float | None,
    report: PricePerceptionReport,
) -> str:
    """Create a short price positioning verdict for the Judge report."""
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
