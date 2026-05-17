from streamlit.testing.v1 import AppTest


def _button(app: AppTest, label: str):
    """Return a Streamlit test button by label."""
    return next(button for button in app.button if button.label == label)


def _value(elements, label: str):
    """Return a Streamlit test element value by label."""
    return next(element.value for element in elements if element.label == label)


def test_streamlit_sample_loader_populates_product_brief(monkeypatch):
    monkeypatch.setenv("BUYERLAB_MOCK_MODE", "true")

    app = AppTest.from_file("app.py")
    app.run(timeout=10)
    _button(app, "Load sample").click().run(timeout=10)

    assert not app.exception
    assert _value(app.text_input, "Brand") == "SoundPeak"
    assert _value(app.text_input, "Model") == "AirBass X2"
    assert _value(app.number_input, "Price") == 799.0
    assert _value(app.text_input, "Competitor name") == "TuneGo Lite Pro"


def test_streamlit_mock_audit_renders_core_dashboard(monkeypatch):
    monkeypatch.setenv("BUYERLAB_MOCK_MODE", "true")

    app = AppTest.from_file("app.py")
    app.run(timeout=10)
    _button(app, "Load sample").click().run(timeout=10)
    _button(app, "Run Pre-launch Audit").click().run(timeout=30)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Launch Readiness",
        "Buyer Personas",
        "Market Context",
        "Category Audit",
        "Conversion Friction Map",
        "Fix Pack",
        "Before / After",
    ]
    metric_labels = {metric.label for metric in app.metric}
    assert "Launch readiness score" in metric_labels
    assert "Launch status" in metric_labels
    assert "Simulated conversion score" in metric_labels
