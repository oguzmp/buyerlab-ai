"""Local price perception heuristics for BuyerLab AI."""

from __future__ import annotations

import json
from dataclasses import asdict

from src.state import PriceBand, PricePerceptionReport, ProductInput


# These TRY bands are demo heuristics for simulated buyer perception only.
# They are not live market data, competitor research, or currency conversion.
TRY_PRICE_BANDS: dict[str, tuple[float, float, float, float]] = {
    "electronics_accessory": (750, 2_000, 4_500, 9_000),
    "fashion_shoes": (900, 2_500, 4_500, 8_000),
    "small_home_appliance": (1_200, 3_500, 7_500, 15_000),
    "handmade_bag": (700, 2_000, 4_500, 9_000),
    "digital_service": (500, 2_000, 6_000, 15_000),
    "online_course": (750, 2_500, 7_000, 20_000),
    "general_product": (500, 1_500, 3_500, 8_000),
}

GENERIC_PRICE_BANDS: dict[str, tuple[float, float, float, float]] = {
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


def normalize_category(category: str) -> str:
    """Normalize free-form product category text into a pricing heuristic bucket."""
    text = (category or "").strip().lower()
    if not text:
        return "general_product"

    for normalized, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return normalized

    return "general_product"


def detect_price_band(price: float, currency: str, category: str) -> PriceBand:
    """Detect a simulated price band without converting TRY/TL to another currency."""
    safe_price = max(0.0, float(price or 0.0))
    normalized_category = normalize_category(category)
    bands = _bands_for_currency(currency).get(
        normalized_category,
        _bands_for_currency(currency)["general_product"],
    )
    budget_max, mid_max, upper_mid_max, premium_max = bands

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
    """Create a dashboard-ready simulated local price perception report."""
    category = normalize_category(product.category)
    currency = (product.currency or "").strip().upper()
    price_band = detect_price_band(product.price, currency, category)
    local_market = "Turkey" if _is_try(currency) else "generic"
    risk = _perceived_value_risk(product, price_band)
    expected_questions = _expected_customer_questions(product, price_band, local_market)
    required_proofs = _required_value_proofs(product, price_band)

    return PricePerceptionReport(
        currency=currency or "UNKNOWN",
        category=category,
        price=max(0.0, float(product.price or 0.0)),
        local_market=local_market,
        price_band=price_band,
        perceived_value_risk=risk,
        expected_customer_questions=expected_questions,
        required_value_proofs=required_proofs,
        pricing_comment=_pricing_comment(product, price_band, risk, local_market),
        suggested_price_positioning=_suggested_price_positioning(price_band, risk),
    )


def build_price_context_for_prompt(product: ProductInput) -> str:
    """Build Bargain Hunter prompt context from local price perception heuristics."""
    report = analyze_local_price_perception(product)
    report_data = asdict(report)
    questions = "; ".join(report.expected_customer_questions) or "No major questions."
    proofs = "; ".join(report.required_value_proofs) or "No extra proof required."

    return "\n".join(
        [
            "Price perception context:",
            (
                "- Method: simulated local pricing assessment based on demo "
                "heuristics, not live market research."
            ),
            f"- Local market: {report.local_market}",
            f"- Currency: {report.currency}",
            f"- Normalized category: {report.category}",
            f"- Price: {report.price:g}",
            f"- Price band: {report.price_band}",
            f"- Perceived value risk: {report.perceived_value_risk}/100",
            f"- Expected customer questions: {questions}",
            f"- Required value proofs: {proofs}",
            f"- Pricing comment: {report.pricing_comment}",
            f"- Suggested positioning: {report.suggested_price_positioning}",
            f"- JSON summary: {json.dumps(report_data, ensure_ascii=True)}",
        ]
    )


def _bands_for_currency(currency: str) -> dict[str, tuple[float, float, float, float]]:
    """Return TRY bands for Turkey, otherwise a generic non-live fallback."""
    if _is_try(currency):
        return TRY_PRICE_BANDS
    return GENERIC_PRICE_BANDS


def _is_try(currency: str) -> bool:
    """Return True for common Turkish lira currency labels."""
    return (currency or "").strip().upper() in {"TRY", "TL", "₺"}


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
    if product.trust_signals:
        risk -= 5
    if product.warranty_or_return_policy.strip():
        risk -= 4
    if product.shipping_info.strip():
        risk -= 3
    return max(0, min(100, risk))


def _weak_copy(product: ProductInput) -> bool:
    """Detect when premium pricing has too little value explanation."""
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
        questions[0] = "Is this TL price justified for this category?"
    if not product.shipping_info.strip():
        questions.append("How much will shipping cost and how fast is delivery?")
    if not product.warranty_or_return_policy.strip():
        questions.append("What happens if I want to return it?")
    if price_band in HIGH_PROOF_BANDS:
        questions.append("What materials, proof, or outcomes justify the higher price?")
    if price_band == "irrational":
        questions.append("Is this price intentionally premium or a pricing mistake?")

    return _dedupe(questions)[:5]


def _required_value_proofs(product: ProductInput, price_band: PriceBand) -> list[str]:
    """Return proof points needed to support the detected price band."""
    proofs = ["Clear benefit explanation", "Shipping and return terms"]
    if price_band in HIGH_PROOF_BANDS:
        proofs.extend(
            [
                "Materials, specs, or deliverables",
                "Authentic reviews or buyer proof",
                "Warranty, guarantee, or support details",
            ]
        )
    if product.category and normalize_category(product.category) == "online_course":
        proofs.append("Curriculum, instructor credibility, and outcome examples")
    if product.category and normalize_category(product.category) == "handmade_bag":
        proofs.append("Material, craftsmanship, and size details")
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
            "shipping, and materials are clearer."
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


def _dedupe(items: list[str]) -> list[str]:
    """Deduplicate short strings while preserving order."""
    deduped: list[str] = []
    for item in items:
        text = " ".join(str(item).split())
        if text and text not in deduped:
            deduped.append(text)
    return deduped
