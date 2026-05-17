import json
import os
from pathlib import Path

from src.graph import run_sample_simulation
from src.category_intelligence import build_category_expectation_check
from src.price_intelligence import (
    analyze_competitor_gap,
    analyze_local_price_perception,
    build_product_brief_context,
    build_price_context_for_prompt,
    build_structured_product_brief,
    detect_price_band,
    get_category_profile,
    normalize_category,
    normalize_currency,
)
from src.state import PAGE_SECTION_NAMES, CompetitorContext, ProductInput


DATA_PATH = Path(__file__).parents[1] / "data" / "sample_products.json"


def test_sample_products_are_valid_and_safe():
    products = json.loads(DATA_PATH.read_text(encoding="utf-8"))

    assert len(products) >= 7
    assert {product["name"] for product in products} >= {
        "Wireless Earbuds Demo",
        "Running Shoes Demo",
        "Coffee Machine Demo",
        "Digital Service Demo",
        "Online Course Demo",
        "Edge Case: Overpriced Pencil",
        "Edge Case: Empty Product Page",
    }
    assert any(not product["trust_signals"] for product in products)
    assert any(not product["shipping_info"] for product in products)

    required_fields = {
        "title",
        "brand",
        "model",
        "product_type",
        "category",
        "normalized_category",
        "market_segment",
        "intended_use_case",
        "local_market",
        "price",
        "currency",
        "description",
        "target_audience",
        "value_proposition",
        "warranty_or_return_policy",
        "shipping_info",
        "trust_signals",
        "reviews_or_social_proof",
        "call_to_action",
        "image_notes",
        "competitor_context",
        "proof_assets",
        "known_limitations",
    }
    competitor_fields = {
        "competitor_name",
        "competitor_price",
        "competitor_currency",
        "competitor_strengths",
        "competitor_weaknesses",
        "our_differentiator",
    }
    real_brand_blocklist = {
        "adidas",
        "amazon",
        "apple",
        "bosch",
        "delonghi",
        "dyson",
        "google",
        "huawei",
        "jbl",
        "lg",
        "meta",
        "microsoft",
        "nike",
        "philips",
        "samsung",
        "sony",
        "xiaomi",
    }
    for product in products:
        assert required_fields <= product.keys()
        assert isinstance(product["trust_signals"], list)
        assert isinstance(product["proof_assets"], list)
        assert isinstance(product["known_limitations"], list)
        assert isinstance(product["competitor_context"], dict)
        assert competitor_fields <= product["competitor_context"].keys()
        assert isinstance(product["competitor_context"]["competitor_strengths"], list)
        assert isinstance(product["competitor_context"]["competitor_weaknesses"], list)
        assert "certified" not in json.dumps(product).lower()

        searchable_brand_text = " ".join(
            [
                product.get("brand", ""),
                product.get("model", ""),
                product.get("title", ""),
                product["competitor_context"].get("competitor_name", ""),
            ]
        ).lower()
        assert not any(brand in searchable_brand_text for brand in real_brand_blocklist)


def test_sample_products_include_demo_edge_cases():
    products = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    by_id = {product["id"]: product for product in products}

    overpriced = by_id["overpriced-pencil-edge"]
    empty_page = by_id["empty-product-page-edge"]

    assert overpriced["price"] >= 1_000_000
    assert overpriced["normalized_category"] == "general_product"
    assert "Irrational price" in overpriced["known_limitations"]
    assert empty_page["price"] == 0
    assert empty_page["description"] == ""
    assert empty_page["competitor_context"]["competitor_name"] == ""


def test_sample_simulation_runs_in_mock_mode():
    os.environ["BUYERLAB_MOCK_MODE"] = "true"
    product = json.loads(DATA_PATH.read_text(encoding="utf-8"))[0]

    result = run_sample_simulation(product)

    assert result.product.title == product["title"]
    assert len(result.first_round_responses) == 4
    assert len(result.debate_history) == 4
    assert result.final_report is not None
    assert 0 <= result.final_report.simulated_conversion_score <= 100


def test_default_page_sections_match_attention_map_contract():
    assert PAGE_SECTION_NAMES == [
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


def test_try_price_perception_uses_local_heuristics_without_conversion():
    product = ProductInput(
        title="Daily Runner",
        category="Footwear",
        price=4200,
        currency="TRY",
        product_type="Daily running shoes",
        description="Lightweight daily shoes.",
        value_proposition="Comfortable everyday support.",
    )

    report = analyze_local_price_perception(product)
    prompt_context = build_price_context_for_prompt(product)

    assert detect_price_band(4200, "TRY", "Footwear") == "upper_mid"
    assert normalize_currency("TL") == "TRY"
    assert report.local_market == "Turkey"
    assert report.currency == "TRY"
    assert report.normalized_category == "fashion_shoes"
    assert report.price_band == "upper_mid"
    assert report.perceived_value_risk >= 70
    assert any("TRY price" in question for question in report.expected_customer_questions)
    assert "TL price" not in prompt_context
    assert "not live market research" in prompt_context


def test_product_identity_and_competitor_context_feed_business_brief():
    product = ProductInput(
        brand="CreatorLab",
        model="Launch Sprint",
        product_type="Online course",
        title="Creator Course",
        category="online workshop",
        market_segment="premium",
        intended_use_case="Creators launching their first digital product.",
        local_market="Turkey",
        price=9500,
        currency="TRY",
        description="A practical creator course with weekly lessons.",
        value_proposition="Helps creators package and sell their first digital product.",
        competitor_context=CompetitorContext(
            competitor_name="Comparable course",
            competitor_price=7000,
            competitor_strengths=["Larger lesson library"],
            competitor_weaknesses=["Less hands-on feedback"],
            our_differentiator="Live feedback sessions and launch templates",
        ),
        proof_assets=[],
        known_limitations=["No public student outcomes yet"],
    )

    brief = build_structured_product_brief(product)
    context = build_product_brief_context(product)
    profile = get_category_profile(product.category)
    competitor_gap = analyze_competitor_gap(product)
    expectation_check = build_category_expectation_check(product)

    assert normalize_category("online workshop") == "online_course"
    assert profile.display_name == "Online course"
    assert brief["normalized_category"] == "online_course"
    assert brief["product_identity"]["brand"] == "CreatorLab"
    assert brief["competitor_gap"]["price_gap"] == 2500
    assert "above" in competitor_gap.value_gap_summary
    assert brief["price_perception"]["price_band"] == "premium"
    assert "heuristic local price perception" in context
    assert any(item["field"] == "student proof" for item in expectation_check)
