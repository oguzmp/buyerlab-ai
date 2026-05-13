"""Core domain models for BuyerLab AI simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Decision = Literal["buy", "reject", "hesitate"]
DebateStance = Literal["support", "oppose", "neutral"]
SectionSentiment = Literal["positive", "neutral", "negative"]
PriceBand = Literal["budget", "mid_range", "upper_mid", "premium", "irrational"]


PAGE_SECTION_NAMES = [
    "title",
    "price",
    "hero_image",
    "description",
    "value_proposition",
    "warranty_or_return_policy",
    "shipping_info",
    "trust_signals",
    "reviews_or_social_proof",
    "call_to_action",
]


def _validate_score(name: str, value: int) -> None:
    """Validate that a score is an integer from 0 to 100."""
    if not isinstance(value, int) or not 0 <= value <= 100:
        raise ValueError(f"{name} must be an integer from 0 to 100.")


def _validate_literal(name: str, value: str, allowed_values: set[str]) -> None:
    """Validate that a string value is one of the allowed labels."""
    if value not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))
        raise ValueError(f"{name} must be one of: {allowed}.")


@dataclass(slots=True)
class ProductInput:
    """Product or service page content being tested before launch."""

    title: str = ""
    category: str = ""
    price: float = 0.0
    currency: str = "USD"
    description: str = ""
    target_audience: str = ""
    value_proposition: str = ""
    warranty_or_return_policy: str = ""
    shipping_info: str = ""
    trust_signals: list[str] = field(default_factory=list)
    reviews_or_social_proof: str = ""
    call_to_action: str = ""
    image_notes: str | None = None


@dataclass(slots=True)
class BuyerPersona:
    """AI buyer profile with a distinct decision style and buying criteria."""

    id: str = ""
    name: str = ""
    role: str = ""
    decision_style: str = ""
    priority_factors: list[str] = field(default_factory=list)
    rejection_triggers: list[str] = field(default_factory=list)
    weight: float = 1.0


@dataclass(slots=True)
class AgentResponse:
    """One buyer agent's simulated purchase decision and objections."""

    persona_id: str = ""
    decision: Decision = "hesitate"
    confidence: int = 0
    purchase_intent: int = 0
    main_reason: str = ""
    objections: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    suggested_fix: str = ""

    def __post_init__(self) -> None:
        _validate_literal("decision", self.decision, {"buy", "reject", "hesitate"})
        _validate_score("confidence", self.confidence)
        _validate_score("purchase_intent", self.purchase_intent)


@dataclass(slots=True)
class DebateTurn:
    """One message in the simulated buyer debate."""

    speaker: str = ""
    message: str = ""
    stance: DebateStance = "neutral"

    def __post_init__(self) -> None:
        _validate_literal("stance", self.stance, {"support", "oppose", "neutral"})


@dataclass(slots=True)
class PageSectionScore:
    """AI-simulated attention and friction for one page section, not real eye-tracking."""

    section_name: str = ""
    attention_score: int = 0
    friction_score: int = 0
    sentiment: SectionSentiment = "neutral"
    reason: str = ""
    suggested_fix: str = ""

    def __post_init__(self) -> None:
        _validate_score("attention_score", self.attention_score)
        _validate_score("friction_score", self.friction_score)
        _validate_literal("sentiment", self.sentiment, {"positive", "neutral", "negative"})


@dataclass(slots=True)
class AttentionMapReport:
    """Future AI-simulated attention and conversion friction map report."""

    section_scores: list[PageSectionScore] = field(default_factory=list)
    strongest_section: str = ""
    weakest_section: str = ""
    highest_friction_section: str = ""
    summary: str = ""


@dataclass(slots=True)
class PricePerceptionReport:
    """Simulated local pricing assessment for a product, not live market research."""

    currency: str = ""
    category: str = "general_product"
    price: float = 0.0
    local_market: str = "generic"
    price_band: PriceBand = "mid_range"
    perceived_value_risk: int = 0
    expected_customer_questions: list[str] = field(default_factory=list)
    required_value_proofs: list[str] = field(default_factory=list)
    pricing_comment: str = ""
    suggested_price_positioning: str = ""

    def __post_init__(self) -> None:
        _validate_literal(
            "price_band",
            self.price_band,
            {"budget", "mid_range", "upper_mid", "premium", "irrational"},
        )
        _validate_score("perceived_value_risk", self.perceived_value_risk)


@dataclass(slots=True)
class SimulationReport:
    """Final judge report summarizing simulated conversion performance."""

    simulated_conversion_score: int = 0
    buyer_loss_reasons: list[str] = field(default_factory=list)
    winning_personas: list[str] = field(default_factory=list)
    lost_personas: list[str] = field(default_factory=list)
    trust_risk_score: int = 0
    price_resistance_score: int = 0
    clarity_score: int = 0
    return_risk_score: int = 0
    top_action_items: list[str] = field(default_factory=list)
    summary: str = ""

    def __post_init__(self) -> None:
        _validate_score("simulated_conversion_score", self.simulated_conversion_score)
        _validate_score("trust_risk_score", self.trust_risk_score)
        _validate_score("price_resistance_score", self.price_resistance_score)
        _validate_score("clarity_score", self.clarity_score)
        _validate_score("return_risk_score", self.return_risk_score)


@dataclass(slots=True)
class SimulationState:
    """Full graph state for a BuyerLab AI product-page simulation."""

    product: ProductInput = field(default_factory=ProductInput)
    personas: list[BuyerPersona] = field(default_factory=list)
    first_round_responses: list[AgentResponse] = field(default_factory=list)
    debate_history: list[DebateTurn] = field(default_factory=list)
    final_report: SimulationReport | None = None
    attention_map: AttentionMapReport | None = None
    optimized_product_copy: str | None = None
    before_score: int | None = None
    after_score: int | None = None

    def __post_init__(self) -> None:
        if self.before_score is not None:
            _validate_score("before_score", self.before_score)
        if self.after_score is not None:
            _validate_score("after_score", self.after_score)


def get_default_personas() -> list[BuyerPersona]:
    """Return the default buyer personas used in the first simulation flow."""
    return [
        BuyerPersona(
            id="skeptic_buyer",
            name="Skeptic Buyer",
            role="Risk-focused product evaluator",
            decision_style="Rejects vague claims and looks for concrete proof.",
            priority_factors=[
                "technical details",
                "warranty or return policy",
                "product proof",
            ],
            rejection_triggers=[
                "unclear claims",
                "missing warranty",
                "weak evidence",
            ],
            weight=1.0,
        ),
        BuyerPersona(
            id="bargain_hunter",
            name="Bargain Hunter",
            role="Price-sensitive value evaluator",
            decision_style="Compares price, total cost, and perceived value.",
            priority_factors=[
                "price",
                "discounts",
                "shipping cost",
            ],
            rejection_triggers=[
                "unclear value",
                "unexpected costs",
                "weak price justification",
            ],
            weight=1.0,
        ),
        BuyerPersona(
            id="impulsive_buyer",
            name="Impulsive Buyer",
            role="Emotion-led purchase evaluator",
            decision_style="Responds to desire, urgency, and visual appeal.",
            priority_factors=[
                "emotional appeal",
                "urgency",
                "visual attractiveness",
            ],
            rejection_triggers=[
                "boring copy",
                "low excitement",
                "weak call to action",
            ],
            weight=1.0,
        ),
        BuyerPersona(
            id="trust_seeker",
            name="Trust Seeker",
            role="Credibility-focused buyer evaluator",
            decision_style="Looks for trust signals before considering purchase.",
            priority_factors=[
                "reviews",
                "seller credibility",
                "guarantee",
            ],
            rejection_triggers=[
                "weak social proof",
                "unprofessional language",
                "missing guarantee",
            ],
            weight=1.0,
        ),
    ]


def get_default_page_sections() -> list[PageSectionScore]:
    """Return neutral sections for a future simulated attention and friction map."""
    return [
        PageSectionScore(
            section_name=section_name,
            sentiment="neutral",
            reason="Not scored yet.",
        )
        for section_name in PAGE_SECTION_NAMES
    ]


def create_empty_state(product: ProductInput) -> SimulationState:
    """Create an empty simulation state for a product before agents run."""
    return SimulationState(
        product=product,
        personas=get_default_personas(),
        first_round_responses=[],
        debate_history=[],
        final_report=None,
        attention_map=None,
        optimized_product_copy=None,
        before_score=None,
        after_score=None,
    )
