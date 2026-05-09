from src.graph import run_sample_simulation


def test_sample_simulation_returns_judged_state():
    product = {
        "id": "test-product",
        "name": "Test Product",
        "category": "Test Category",
        "price": 25.0,
        "description": "A simple product for testing the simulation scaffold.",
        "target_audience": "Early testers",
    }

    result = run_sample_simulation(product)

    assert result["product"]["name"] == "Test Product"
    assert result["buyer_response"]
    assert result["judge_summary"]
    assert 1 <= result["purchase_intent_score"] <= 10
