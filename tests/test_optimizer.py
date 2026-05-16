import os

from src.optimizer import generate_optimized_product
from src.state import CompetitorContext, ProductInput, SimulationReport


def test_fix_pack_includes_checklists_without_inventing_fake_proof():
    os.environ["BUYERLAB_MOCK_MODE"] = "true"
    product = ProductInput(
        brand="SoundPeak",
        model="AirBass X2",
        product_type="Wireless earbuds",
        title="SoundPeak AirBass X2",
        category="Electronics accessory",
        normalized_category="electronics_accessory",
        price=799,
        currency="TRY",
        description="Wireless earbuds with claimed better microphone quality.",
        value_proposition="Affordable daily earbuds.",
        warranty_or_return_policy="",
        shipping_info="",
        trust_signals=[],
        reviews_or_social_proof="",
        competitor_context=CompetitorContext(
            competitor_name="TuneGo Lite Pro",
            competitor_price=699,
            competitor_currency="TRY",
            our_differentiator="Claimed better microphone and longer battery life",
        ),
        proof_assets=[],
        known_limitations=["Battery life is not specified"],
    )
    report = SimulationReport(
        simulated_conversion_score=32,
        launch_readiness_score=20,
        launch_status="not_ready",
        trust_risk_score=80,
        price_resistance_score=73,
        clarity_score=42,
        return_risk_score=75,
        required_fix_before_launch=[
            "Add exact battery life, warranty period, compatibility, technical specs, return policy, and real usage proof before launch."
        ],
        price_justification_verdict=(
            "799 TRY places this product in a mid_range perception band."
        ),
        launch_decision_summary="Decision: Do not launch yet.",
    )

    suggestion = generate_optimized_product(product, report)

    assert suggestion.trust_proof_checklist
    assert suggestion.missing_information_checklist
    assert "battery life" in " ".join(suggestion.missing_information_checklist).lower()
    assert "TuneGo Lite Pro" in suggestion.competitor_comparison_suggestion
    forbidden = " ".join(
        [
            *suggestion.trust_proof_checklist,
            suggestion.competitor_comparison_suggestion,
            suggestion.description,
        ]
    ).lower()
    assert "certified" not in forbidden
    assert "guaranteed" not in forbidden
