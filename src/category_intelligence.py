"""Heuristic category intelligence for BuyerLab AI product-page audits."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

from src.state import BuyerPersona, CategoryProfile, NormalizedCategory, ProductInput


CATEGORY_PROFILES: dict[NormalizedCategory, CategoryProfile] = {
    "electronics_accessory": CategoryProfile(
        normalized_category="electronics_accessory",
        display_name="Electronics accessory",
        typical_customer_expectations=[
            "clear specs and compatibility",
            "battery life or durability claims with proof",
            "warranty and return safety",
            "real usage proof for microphone, sound, or performance claims",
        ],
        required_information_fields=[
            "battery life",
            "warranty",
            "technical specifications",
            "compatibility",
            "return policy",
            "real usage proof",
        ],
        common_conversion_blockers=[
            "unclear specs",
            "unproven performance claims",
            "missing warranty",
            "weak compatibility information",
        ],
        common_return_risks=[
            "compatibility mismatch",
            "battery life disappointment",
            "sound or microphone quality mismatch",
        ],
        recommended_trust_signals=[
            "warranty terms",
            "technical spec table",
            "real usage photos or clips",
            "support contact",
        ],
        default_persona_weights={
            "skeptic_buyer": 1.25,
            "bargain_hunter": 1.1,
            "impulsive_buyer": 0.8,
            "trust_seeker": 1.15,
        },
        try_price_bands=(750, 2_000, 4_500, 9_000),
    ),
    "fashion_shoes": CategoryProfile(
        normalized_category="fashion_shoes",
        display_name="Fashion shoes",
        typical_customer_expectations=[
            "size guide and fit clarity",
            "material and comfort details",
            "easy return or exchange policy",
            "real product photos",
        ],
        required_information_fields=[
            "size guide",
            "fit information",
            "material",
            "return/exchange policy",
            "comfort use case",
            "real product photos",
        ],
        common_conversion_blockers=[
            "unclear sizing",
            "weak material details",
            "no exchange policy",
            "generic product photos",
        ],
        common_return_risks=[
            "wrong size",
            "fit feels different than expected",
            "material quality mismatch",
        ],
        recommended_trust_signals=[
            "size chart",
            "exchange policy",
            "on-foot photos",
            "material closeups",
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
            "usage and safety details",
            "warranty and service clarity",
            "delivery and return confidence",
            "proof of quality",
        ],
        required_information_fields=[
            "technical specifications",
            "usage instructions",
            "warranty",
            "service terms",
            "return policy",
            "delivery information",
        ],
        common_conversion_blockers=[
            "missing warranty",
            "unclear safety details",
            "no service information",
            "weak quality proof",
        ],
        common_return_risks=[
            "does not fit kitchen or home setup",
            "performance below expectation",
            "service process unclear",
        ],
        recommended_trust_signals=[
            "warranty terms",
            "safety and usage notes",
            "delivery details",
            "support channel",
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
        required_information_fields=[
            "material",
            "dimensions",
            "craftsmanship proof",
            "care instructions",
            "return policy",
            "real product photos",
        ],
        common_conversion_blockers=[
            "unclear material quality",
            "no size details",
            "weak authenticity proof",
            "unclear return policy",
        ],
        common_return_risks=[
            "size mismatch",
            "material expectation mismatch",
            "color or finish mismatch",
        ],
        recommended_trust_signals=[
            "maker story",
            "material closeups",
            "dimensions",
            "care and return terms",
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
            "clear scope",
            "delivery time",
            "revision policy",
            "portfolio proof",
            "support terms",
        ],
        required_information_fields=[
            "scope",
            "delivery time",
            "revision policy",
            "portfolio proof",
            "support terms",
            "what is included and excluded",
        ],
        common_conversion_blockers=[
            "scope ambiguity",
            "unclear deliverables",
            "no portfolio proof",
            "unclear revision terms",
        ],
        common_return_risks=[
            "deliverable mismatch",
            "timeline mismatch",
            "revision expectation mismatch",
        ],
        recommended_trust_signals=[
            "portfolio examples",
            "scope checklist",
            "revision terms",
            "support channel",
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
        required_information_fields=[
            "curriculum",
            "instructor credibility",
            "learning outcomes",
            "time commitment",
            "refund policy",
            "student proof",
        ],
        common_conversion_blockers=[
            "vague outcome promise",
            "missing instructor proof",
            "unclear curriculum",
            "no refund policy",
        ],
        common_return_risks=[
            "course depth mismatch",
            "outcome expectation mismatch",
            "time commitment mismatch",
        ],
        recommended_trust_signals=[
            "curriculum outline",
            "instructor proof",
            "student examples",
            "refund terms",
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
        required_information_fields=[
            "description",
            "value proposition",
            "shipping information",
            "return policy",
            "trust signals",
            "proof assets",
        ],
        common_conversion_blockers=[
            "unclear value",
            "missing proof",
            "weak shipping or return clarity",
            "generic copy",
        ],
        common_return_risks=[
            "expectation mismatch",
            "usage mismatch",
            "unclear return terms",
        ],
        recommended_trust_signals=[
            "secure checkout",
            "return terms",
            "support contact",
            "authentic product proof",
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
        "wireless earbuds",
    ],
    "fashion_shoes": ["shoe", "sneaker", "footwear", "running", "fashion"],
    "small_home_appliance": [
        "appliance",
        "kettle",
        "blender",
        "coffee",
        "home appliance",
        "kitchen",
    ],
    "handmade_bag": ["bag", "tote", "handmade", "leather", "purse"],
    "digital_service": [
        "digital service",
        "subscription",
        "saas",
        "software",
        "service",
    ],
    "online_course": ["course", "bootcamp", "training", "workshop", "education"],
}

FIELD_HINTS = {
    "battery life": ["battery", "hour", "mah"],
    "compatibility": ["compatib", "ios", "android", "device", "bluetooth"],
    "technical specifications": ["spec", "bluetooth", "watt", "mah", "material"],
    "microphone or sound quality proof": ["microphone", "mic", "sound", "audio"],
    "warranty": ["warranty", "guarantee"],
    "return policy": ["return", "refund", "exchange"],
    "real usage proof": ["review", "tester", "photo", "video", "proof"],
    "size guide": ["size", "guide", "fit"],
    "fit information": ["fit", "comfort", "width"],
    "material": ["material", "leather", "cotton", "mesh", "fabric"],
    "return/exchange policy": ["return", "exchange", "refund"],
    "comfort use case": ["comfort", "daily", "walking", "running"],
    "real product photos": ["photo", "image", "on-foot", "lifestyle"],
    "scope": ["scope", "include", "deliverable"],
    "delivery time": ["delivery", "timeline", "day", "week"],
    "revision policy": ["revision", "change", "round"],
    "portfolio proof": ["portfolio", "case", "example"],
    "support terms": ["support", "contact", "help"],
    "what is included and excluded": ["include", "exclude", "deliverable"],
    "curriculum": ["curriculum", "module", "lesson"],
    "instructor credibility": ["instructor", "mentor", "experience"],
    "learning outcomes": ["learn", "outcome", "result"],
    "time commitment": ["hour", "week", "time"],
    "refund policy": ["refund", "return", "guarantee"],
    "student proof": ["student", "testimonial", "case", "proof"],
    "description": ["description"],
    "value proposition": ["value", "benefit", "why"],
    "shipping information": ["shipping", "delivery"],
    "trust signals": ["secure", "support", "trust", "warranty"],
    "proof assets": ["proof", "review", "photo", "video"],
}


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
    """Return heuristic expectations for a normalized product category."""
    return CATEGORY_PROFILES[normalize_category(category)]


def apply_category_persona_weights(
    personas: list[BuyerPersona],
    category: str,
) -> list[BuyerPersona]:
    """Apply category-specific persona weights without mutating source personas."""
    weights = get_category_profile(category).default_persona_weights
    return [
        replace(persona, weight=weights.get(persona.id, persona.weight))
        for persona in personas
    ]


def build_category_expectation_check(product: ProductInput) -> list[dict[str, str]]:
    """Score required category information as present, weak, or missing."""
    profile = get_category_profile(product.normalized_category or product.category)
    return [
        {
            "field": field_name,
            "status": _field_status(product, field_name),
            "note": _field_note(product, field_name),
        }
        for field_name in profile.required_information_fields
    ]


def missing_required_information(product: ProductInput) -> list[str]:
    """Return category-required information fields that are weak or missing."""
    return [
        item["field"]
        for item in build_category_expectation_check(product)
        if item["status"] != "present"
    ]


def build_category_context(product: ProductInput) -> dict[str, object]:
    """Build category context for prompts; this is heuristic, not market data."""
    normalized = normalize_category(product.normalized_category or product.category)
    profile = get_category_profile(normalized)
    return {
        "normalized_category": normalized,
        "display_name": profile.display_name,
        "typical_customer_expectations": profile.typical_customer_expectations,
        "required_information_fields": profile.required_information_fields,
        "common_conversion_blockers": profile.common_conversion_blockers,
        "common_return_risks": profile.common_return_risks,
        "recommended_trust_signals": profile.recommended_trust_signals,
        "default_persona_weights": profile.default_persona_weights,
        "try_price_bands": {
            "budget_max": profile.try_price_bands[0],
            "mid_range_max": profile.try_price_bands[1],
            "upper_mid_max": profile.try_price_bands[2],
            "premium_max": profile.try_price_bands[3],
        },
        "category_expectation_check": build_category_expectation_check(product),
    }


def _field_status(product: ProductInput, field_name: str) -> str:
    """Return present, weak, or missing for one category requirement."""
    text = _product_search_text(product)
    hints = FIELD_HINTS.get(field_name, [field_name])
    has_hint = any(hint.lower() in text for hint in hints)

    if has_hint:
        return "present"
    if field_name in {"proof assets", "real usage proof", "portfolio proof", "student proof"}:
        return "present" if product.proof_assets else "missing"
    if field_name in {"warranty", "return policy", "return/exchange policy", "refund policy"}:
        return "present" if product.warranty_or_return_policy.strip() else "missing"
    if field_name in {"shipping information", "delivery information", "delivery time"}:
        return "present" if product.shipping_info.strip() else "missing"
    if field_name in {"description", "scope", "curriculum"} and product.description.strip():
        return "weak"
    if field_name in {"value proposition", "learning outcomes"} and product.value_proposition.strip():
        return "weak"
    return "missing"


def _field_note(product: ProductInput, field_name: str) -> str:
    """Return a short note for category expectation checks."""
    status = _field_status(product, field_name)
    if status == "present":
        return "Covered in the provided product brief."
    if status == "weak":
        return "Partially mentioned, but needs clearer proof or specifics."
    return "Missing from the provided product brief."


def _product_search_text(product: ProductInput) -> str:
    """Combine product brief fields for lightweight requirement detection."""
    parts = [
        product.brand,
        product.model,
        product.product_type,
        product.title,
        product.category,
        product.description,
        product.value_proposition,
        product.warranty_or_return_policy,
        product.shipping_info,
        product.reviews_or_social_proof,
        product.call_to_action,
        product.image_notes or "",
        product.intended_use_case,
        " ".join(product.trust_signals),
        " ".join(product.proof_assets),
        " ".join(product.known_limitations),
    ]
    return " ".join(parts).lower()
