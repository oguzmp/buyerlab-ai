from src.launch_readiness import (
    build_category_expectation_check,
    build_buyer_persona_verdicts,
    build_launch_readiness_report,
    determine_launch_status,
)
from src.state import (
    AgentResponse,
    CompetitorContext,
    ProductInput,
    SimulationState,
    get_default_personas,
)


def test_launch_readiness_blocks_ready_when_category_and_price_proof_are_weak():
    product = ProductInput(
        brand="SoundPeak",
        model="AirBass X2",
        product_type="Wireless earbuds",
        title="SoundPeak AirBass X2",
        category="Electronics accessory",
        normalized_category="electronics_accessory",
        market_segment="mid_range",
        intended_use_case="Daily music and online meetings.",
        local_market="Türkiye",
        price=799,
        currency="TRY",
        description="Wireless earbuds with better microphone quality.",
        value_proposition="Affordable daily earbuds.",
        warranty_or_return_policy="",
        shipping_info="",
        trust_signals=[],
        reviews_or_social_proof="",
        call_to_action="Join the launch list",
        competitor_context=CompetitorContext(
            competitor_name="TuneGo Lite Pro",
            competitor_price=699,
            competitor_currency="TRY",
            competitor_strengths=["Lower listed price"],
            our_differentiator="Claimed better microphone and longer battery life",
        ),
        proof_assets=[],
        known_limitations=["Battery life not specified"],
    )
    state = SimulationState(
        product=product,
        personas=get_default_personas(),
        first_round_responses=[
            AgentResponse(
                persona_id="skeptic_buyer",
                decision="reject",
                confidence=88,
                purchase_intent=18,
                main_reason="Technical proof, warranty, and battery life are missing.",
                objections=["missing warranty", "unproven microphone claim"],
                missing_information=["battery life", "technical specifications"],
                suggested_fix="Add exact battery life, warranty, specs, and proof.",
            ),
            AgentResponse(
                persona_id="bargain_hunter",
                decision="hesitate",
                confidence=72,
                purchase_intent=45,
                main_reason="The 799 TRY price is not justified against the competitor.",
                objections=["price/value gap", "competitor is cheaper"],
                missing_information=["proof for differentiator"],
                suggested_fix="Explain why it is worth 100 TRY more than the competitor.",
            ),
        ],
    )

    report = build_launch_readiness_report(state)

    assert report.launch_status != "ready"
    assert report.launch_readiness_score < 80
    assert report.simulated_conversion_score == 32
    assert report.main_blocker
    assert any(
        row["field_name"] == "battery life" and row["status"] in {"missing", "weak"}
        for row in report.category_expectation_check
    )
    assert any("100 TRY more" in fix for fix in report.required_fix_before_launch)
    assert "not a real market prediction" in report.summary
    assert "not ready to launch" in report.executive_verdict.lower()
    assert "BuyerLab lost" in report.buyer_loss_summary
    assert "799 TRY" in report.price_justification_verdict
    assert "TuneGo Lite Pro" in report.competitor_gap_verdict
    assert report.launch_decision_summary.startswith("Decision:")


def test_category_expectation_check_uses_dashboard_ready_fields():
    product = ProductInput(
        title="CreatorPath First Launch",
        category="Online course",
        normalized_category="online_course",
        price=9500,
        currency="TRY",
        description="Weekly modules for creators.",
        value_proposition="Helps creators launch.",
    )

    rows = build_category_expectation_check(product)

    assert rows
    assert {"field_name", "status", "impact", "explanation", "suggested_fix"} <= rows[0].keys()
    assert any(row["field_name"] == "student proof" for row in rows)


def test_launch_status_thresholds_and_critical_blockers():
    assert determine_launch_status(82, trust_risk_score=20, category_expectation_check=[]) == "ready"
    assert (
        determine_launch_status(
            85,
            trust_risk_score=72,
            category_expectation_check=[],
        )
        == "needs_fixes"
    )
    assert determine_launch_status(45, trust_risk_score=20, category_expectation_check=[]) == "not_ready"
    assert (
        determine_launch_status(
            86,
            trust_risk_score=20,
            return_risk_score=82,
            simulated_conversion_score=80,
            category_expectation_check=[],
        )
        == "not_ready"
    )


def test_buyer_persona_verdicts_are_short_and_normalized():
    verdicts = build_buyer_persona_verdicts(
        [
            AgentResponse(
                persona_id="trust_seeker",
                decision="reject",
                confidence=81,
                purchase_intent=22,
                main_reason="Trust signals are weak.",
                objections=["no reviews", "missing guarantee"],
                suggested_fix="Add real trust signals.",
            )
        ],
        get_default_personas(),
    )

    assert verdicts == [
        {
            "persona_name": "Trust Seeker",
            "decision": "reject",
            "purchase_intent": 22,
            "confidence": 81,
            "main_reason": "Trust signals are weak.",
            "top_objection": "no reviews",
            "suggested_fix": "Add real trust signals.",
            "business_impact": "high",
        }
    ]
