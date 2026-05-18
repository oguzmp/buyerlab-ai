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
    _button(app, "Kablosuz Kulaklık (Demo) çalıştır").click().run(timeout=30)

    assert not app.exception
    assert _value(app.number_input, "Fiyat") == 799.0
    assert _value(app.text_input, "Ürün başlığı") == "SoundPeak AirBass X2 Wireless Earbuds"


def test_streamlit_mock_audit_renders_core_dashboard(monkeypatch):
    monkeypatch.setenv("BUYERLAB_MOCK_MODE", "true")

    app = AppTest.from_file("app.py")
    app.run(timeout=10)
    _button(app, "Kablosuz Kulaklık (Demo) çalıştır").click().run(timeout=30)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "1 Dakikalık Rapor",
        "Müşteri İtirazları",
        "Dönüşüm Sürtünme Haritası",
        "Düzeltme Planı",
    ]
    metric_labels = {metric.label for metric in app.metric}
    assert "Yayına Hazırlık Skoru" in metric_labels
    assert "Yayın Durumu" in metric_labels
    assert "Simüle Dönüşüm Skoru" in metric_labels
