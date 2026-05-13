import json
import os
from pathlib import Path

from src.graph import run_sample_simulation
from src.price_intelligence import (
    analyze_local_price_perception,
    build_price_context_for_prompt,
    detect_price_band,
)
from src.state import PAGE_SECTION_NAMES, ProductInput


DATA_PATH = Path(__file__).parents[1] / "data" / "sample_products.json"


def test_sample_products_are_valid_and_safe():
    products = json.loads(DATA_PATH.read_text(encoding="utf-8"))

    assert len(products) >= 2
    assert any(not product["trust_signals"] for product in products)
    assert any(not product["shipping_info"] for product in products)

    required_fields = {
        "title",
        "category",
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
    }
    for product in products:
        assert required_fields <= product.keys()
        assert isinstance(product["trust_signals"], list)
        assert "certified" not in json.dumps(product).lower()


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
        description="Lightweight daily shoes.",
        value_proposition="Comfortable everyday support.",
    )

    report = analyze_local_price_perception(product)
    prompt_context = build_price_context_for_prompt(product)

    assert detect_price_band(4200, "TRY", "Footwear") == "upper_mid"
    assert report.local_market == "Turkey"
    assert report.price_band == "upper_mid"
    assert report.perceived_value_risk >= 70
    assert any("TL price" in question for question in report.expected_customer_questions)
    assert "not live market research" in prompt_context
