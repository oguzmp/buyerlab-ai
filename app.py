"""Streamlit dashboard for BuyerLab AI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

from src.attention_map import generate_attention_map, get_section_priority_label
from src.graph import run_simulation
from src.judge import analyze_buyer_losses
from src.launch_readiness import build_category_expectation_check
from src.optimizer import (
    apply_optimization_to_product,
    compare_before_after,
    generate_optimized_product,
)
from src.price_intelligence import (
    analyze_competitor_gap,
    analyze_local_price_perception,
    normalize_category,
    normalize_currency,
)
from src.state import (
    AgentResponse,
    AttentionMapReport,
    CompetitorContext,
    ProductInput,
    SimulationReport,
    SimulationState,
    get_default_personas,
)


DATA_PATH = Path(__file__).parent / "data" / "sample_products.json"

LANGUAGE_OPTIONS = {"English": "en", "Türkçe": "tr"}

CATEGORY_OPTIONS = [
    "electronics_accessory",
    "fashion_shoes",
    "small_home_appliance",
    "handmade_bag",
    "digital_service",
    "online_course",
    "general_product",
]

MARKET_SEGMENTS = ["budget", "mid_range", "upper_mid", "premium"]
CURRENCY_OPTIONS = ["TRY", "USD", "EUR", "GBP"]

UI_TEXT = {
    "en": {
        "language": "Language / Dil",
        "app_eyebrow": "Product Launch Audit Dashboard",
        "tagline": "Test your product with AI buyers before launch.",
        "hero_copy": (
            "BuyerLab AI audits product pages before launch using AI buyer "
            "personas, category standards, local price perception, and "
            "competitor context."
        ),
        "hero_disclaimer": "Scores are AI-simulated diagnostics, not real market predictions.",
        "brief_wizard": "Product Brief Wizard",
        "brief_help": "Prepare a structured product brief for a simulated buyer assessment.",
        "sample_loader": "Quick Start Samples",
        "sample_help": "Load a realistic demo case, then adjust the product brief.",
        "sample_product": "Sample product",
        "load_sample": "Load sample",
        "identity": "A) Product Identity",
        "pricing": "B) Pricing and Market Context",
        "trust": "C) Proof and Trust",
        "content": "D) Product Page Content",
        "brand": "Brand",
        "model": "Model",
        "product_type": "Product type",
        "category": "Category",
        "market_segment": "Market segment",
        "intended_use_case": "Intended use case",
        "price": "Price",
        "currency": "Currency",
        "local_market": "Local market",
        "competitor_name": "Competitor name",
        "competitor_price": "Competitor price",
        "competitor_strengths": "Competitor strengths",
        "competitor_weaknesses": "Competitor weaknesses",
        "our_differentiator": "Our differentiator",
        "warranty": "Warranty / return policy",
        "shipping": "Shipping info",
        "trust_signals": "Trust signals",
        "proof_assets": "Proof assets",
        "known_limitations": "Known limitations",
        "product_title": "Product title",
        "value_proposition": "Value proposition",
        "product_description": "Product description",
        "social_proof": "Reviews or social proof",
        "cta": "Call to action",
        "image_notes": "Image notes",
        "run": "Run Pre-launch Audit",
        "missing_title": "Add a product title before running the pre-launch audit.",
        "mock_on": "Mock mode active",
        "missing_key": (
            "Gemini is not connected. Add GEMINI_API_KEY or set "
            "BUYERLAB_MOCK_MODE=true for demo mode."
        ),
        "key_detected": "Gemini connected",
        "empty_title": "Build a product brief, then run the audit",
        "empty_copy": (
            "The report will show launch readiness, buyer persona reactions, "
            "heuristic local price perception, conversion friction, and a practical fix pack."
        ),
        "tab_launch": "Launch Readiness",
        "tab_personas": "Buyer Personas",
        "tab_market": "Market Context",
        "tab_category": "Category Audit",
        "tab_friction": "Conversion Friction Map",
        "tab_fix": "Fix Pack",
        "tab_compare": "Before / After",
        "launch_score": "Launch readiness score",
        "launch_status": "Launch status",
        "simulated_score": "Simulated conversion score",
        "main_blocker": "Main blocker",
        "executive_verdict": "Executive verdict",
        "next_actions": "Next best actions",
        "required_fixes": "Required fixes before launch",
        "judge_missing": "Judge report is not available yet.",
        "decision": "Decision",
        "intent": "purchase intent",
        "confidence": "confidence",
        "top_objection": "Top objection",
        "suggested_fix": "Suggested fix",
        "no_objection": "No major objection.",
        "waiting": "Waiting for buyer evaluation.",
        "terminal": "Live Debate / Market Terminal",
        "no_debate": "No debate history yet.",
        "buyer_loss": "Buyer Loss Reasons",
        "market_note": "Heuristic local price perception, not live market research.",
        "price_band": "Price band",
        "value_risk": "Perceived value risk",
        "required_value_proofs": "Required value proofs",
        "pricing_comment": "Pricing comment",
        "price_positioning": "Suggested price positioning",
        "competitor_gap": "Competitor gap summary",
        "proofs_to_win": "Required proofs to win against competitor",
        "expected_questions": "Expected customer questions",
        "no_competitor": "Competitor context was not provided.",
        "category_audit_note": "Category checks use heuristic category profiles for pre-launch audit diagnostics.",
        "required_field": "Required information field",
        "status": "Status",
        "impact": "Impact",
        "explanation": "Explanation",
        "business_impact": "Business impact",
        "attention_caption": "This is AI-simulated buyer attention, not real eye-tracking.",
        "section": "Section",
        "attention": "Attention",
        "friction": "Friction",
        "priority": "Priority",
        "sentiment": "Sentiment",
        "reason": "Reason",
        "strongest": "Strongest section",
        "weakest": "Weakest section",
        "highest_friction": "Highest friction",
        "optimized_title": "Rewritten product title",
        "optimized_description": "Improved description",
        "improved_value": "Improved value proposition",
        "warranty_improvement": "Warranty / return improvement suggestion",
        "shipping_improvement": "Shipping improvement suggestion",
        "improved_trust": "Trust signal suggestions",
        "faq": "FAQ suggestions",
        "cta_suggestion": "CTA suggestion",
        "change_summary": "Change summary",
        "before_score": "Before simulated conversion score",
        "after_score": "After simulated conversion score",
        "score_delta": "Score delta",
        "improved_sections": "Improved sections",
        "remaining_risks": "Remaining risks",
        "no_section_lift": "No section lift yet",
        "score_caption": "Scores are simulated buyer testing signals, not real market predictions.",
        "running": "Running pre-launch audit...",
        "simulation_failed": "Simulation failed safely:",
        "score_unavailable": "Simulated conversion score unavailable.",
    },
    "tr": {
        "language": "Language / Dil",
        "app_eyebrow": "Product Launch Audit Dashboard",
        "tagline": "Urununu lansmandan once AI alicilarla test et.",
        "hero_copy": (
            "BuyerLab AI, urun sayfalarini lansman oncesinde AI alici "
            "personalari, kategori standartlari, yerel fiyat algisi ve rakip "
            "baglami ile denetler."
        ),
        "hero_disclaimer": "Scores are AI-simulated diagnostics, not real market predictions.",
        "brief_wizard": "Urun Brief Sihirbazi",
        "brief_help": "Simule alici degerlendirmesi icin yapilandirilmis urun brief'i hazirla.",
        "sample_loader": "Hizli Demo Ornekleri",
        "sample_help": "Gercekci bir demo vakasi yukle, sonra brief'i duzenle.",
        "sample_product": "Ornek urun",
        "load_sample": "Ornegi yukle",
        "identity": "A) Urun Kimligi",
        "pricing": "B) Fiyat ve Pazar Baglami",
        "trust": "C) Kanit ve Guven",
        "content": "D) Urun Sayfasi Icerigi",
        "brand": "Marka",
        "model": "Model",
        "product_type": "Urun tipi",
        "category": "Kategori",
        "market_segment": "Pazar segmenti",
        "intended_use_case": "Kullanim senaryosu",
        "price": "Fiyat",
        "currency": "Para birimi",
        "local_market": "Yerel pazar",
        "competitor_name": "Rakip adi",
        "competitor_price": "Rakip fiyati",
        "competitor_strengths": "Rakip guclu yonleri",
        "competitor_weaknesses": "Rakip zayif yonleri",
        "our_differentiator": "Bizim farkimiz",
        "warranty": "Garanti / iade politikasi",
        "shipping": "Kargo bilgisi",
        "trust_signals": "Guven sinyalleri",
        "proof_assets": "Kanit varliklari",
        "known_limitations": "Bilinen eksikler",
        "product_title": "Urun basligi",
        "value_proposition": "Deger onerisi",
        "product_description": "Urun aciklamasi",
        "social_proof": "Yorumlar veya sosyal kanit",
        "cta": "Eylem cagrisi",
        "image_notes": "Gorsel notlari",
        "run": "Lansman Audit'ini Calistir",
        "missing_title": "Lansman audit'i icin once urun basligi ekle.",
        "mock_on": "Mock mode active",
        "missing_key": (
            "Gemini bagli degil. GEMINI_API_KEY ekle veya demo icin "
            "BUYERLAB_MOCK_MODE=true kullan."
        ),
        "key_detected": "Gemini bagli",
        "empty_title": "Urun brief'ini hazirla, sonra audit'i calistir",
        "empty_copy": (
            "Rapor; launch readiness, alici persona tepkileri, heuristic local "
            "price perception, conversion friction ve pratik fix pack gosterecek."
        ),
        "tab_launch": "Launch Readiness",
        "tab_personas": "Alici Personalar",
        "tab_market": "Pazar Baglami",
        "tab_category": "Category Audit",
        "tab_friction": "Conversion Friction Map",
        "tab_fix": "Fix Pack",
        "tab_compare": "Once / Sonra",
        "launch_score": "Launch readiness skoru",
        "launch_status": "Launch status",
        "simulated_score": "Simulated conversion score",
        "main_blocker": "Ana engel",
        "executive_verdict": "Yonetici ozeti",
        "next_actions": "Siradaki en iyi aksiyonlar",
        "required_fixes": "Launch oncesi zorunlu fix'ler",
        "judge_missing": "Judge raporu henuz yok.",
        "decision": "Karar",
        "intent": "satin alma niyeti",
        "confidence": "guven",
        "top_objection": "Ana itiraz",
        "suggested_fix": "Onerilen fix",
        "no_objection": "Buyuk itiraz yok.",
        "waiting": "Alici degerlendirmesi bekleniyor.",
        "terminal": "Live Debate / Market Terminal",
        "no_debate": "Henuz tartisma yok.",
        "buyer_loss": "Buyer Loss Reasons",
        "market_note": "Heuristic local price perception; canli pazar arastirmasi degildir.",
        "price_band": "Fiyat bandi",
        "value_risk": "Algilanan deger riski",
        "required_value_proofs": "Gerekli deger kanitlari",
        "pricing_comment": "Fiyat yorumu",
        "price_positioning": "Onerilen fiyat konumlandirmasi",
        "competitor_gap": "Rakip gap ozeti",
        "proofs_to_win": "Rakibe karsi kazanmak icin kanitlar",
        "expected_questions": "Beklenen musteri sorulari",
        "no_competitor": "Rakip baglami girilmedi.",
        "category_audit_note": "Kategori kontrolleri pre-launch audit icin heuristic kategori profillerini kullanir.",
        "required_field": "Gerekli bilgi alani",
        "status": "Durum",
        "impact": "Etki",
        "explanation": "Aciklama",
        "business_impact": "Business impact",
        "attention_caption": "This is AI-simulated buyer attention, not real eye-tracking.",
        "section": "Bolum",
        "attention": "Dikkat",
        "friction": "Surtunme",
        "priority": "Oncelik",
        "sentiment": "Duygu",
        "reason": "Neden",
        "strongest": "En guclu bolum",
        "weakest": "En zayif bolum",
        "highest_friction": "En yuksek surtunme",
        "optimized_title": "Yeniden yazilan urun basligi",
        "optimized_description": "Iyilestirilmis aciklama",
        "improved_value": "Iyilestirilmis deger onerisi",
        "warranty_improvement": "Garanti / iade iyilestirme onerisi",
        "shipping_improvement": "Kargo iyilestirme onerisi",
        "improved_trust": "Guven sinyali onerileri",
        "faq": "SSS onerileri",
        "cta_suggestion": "CTA onerisi",
        "change_summary": "Degisiklik ozeti",
        "before_score": "Once simulated conversion score",
        "after_score": "Sonra simulated conversion score",
        "score_delta": "Skor farki",
        "improved_sections": "Iyilesen bolumler",
        "remaining_risks": "Kalan riskler",
        "no_section_lift": "Henuz bolum iyilesmesi yok",
        "score_caption": "Skorlar simule alici test sinyalleridir; gercek pazar tahmini degildir.",
        "running": "Pre-launch audit calisiyor...",
        "simulation_failed": "Simulasyon guvenli sekilde durdu:",
        "score_unavailable": "Simulated conversion score kullanilamiyor.",
    },
}

PERSONA_NAMES = {
    "tr": {
        "Skeptic Buyer": "Supheci Alici",
        "Bargain Hunter": "Firsat Avcisi",
        "Impulsive Buyer": "Durtusel Alici",
        "Trust Seeker": "Guven Arayan",
    }
}

LABELS = {
    "en": {
        "ready": "Ready",
        "needs_fixes": "Needs Fixes",
        "not_ready": "Not Ready",
        "budget": "Budget",
        "mid_range": "Mid-range",
        "upper_mid": "Upper-mid",
        "premium": "Premium",
        "irrational": "Irrational",
    },
    "tr": {
        "buy": "satin alir",
        "reject": "reddeder",
        "hesitate": "kararsiz",
        "support": "destekliyor",
        "oppose": "karsi",
        "neutral": "notr",
        "high": "yuksek",
        "medium": "orta",
        "low": "dusuk",
        "positive": "olumlu",
        "negative": "olumsuz",
        "ready": "Hazir",
        "needs_fixes": "Fix Gerekli",
        "not_ready": "Hazir Degil",
        "budget": "Budget",
        "mid_range": "Mid-range",
        "upper_mid": "Upper-mid",
        "premium": "Premium",
        "irrational": "Mantiksiz Yuksek",
        "strong_conversion_area": "guclu donusum alani",
        "critical_fix_area": "kritik fix alani",
        "hidden_risk_area": "gizli risk alani",
        "low_priority_area": "dusuk oncelik",
    },
}

CATEGORY_LABELS = {
    "en": {
        "electronics_accessory": "Electronics / Accessory",
        "fashion_shoes": "Fashion / Shoes",
        "small_home_appliance": "Small Home Appliance",
        "handmade_bag": "Handmade Bag",
        "digital_service": "Digital Service",
        "online_course": "Online Course",
        "general_product": "General Product",
    },
    "tr": {
        "electronics_accessory": "Elektronik / Aksesuar",
        "fashion_shoes": "Moda / Ayakkabi",
        "small_home_appliance": "Kucuk Ev Aleti",
        "handmade_bag": "El Yapimi Canta",
        "digital_service": "Dijital Servis",
        "online_course": "Online Kurs",
        "general_product": "Genel Urun",
    },
}

SECTION_LABELS = {
    "en": {
        "title": "Title",
        "price": "Price",
        "hero_image": "Hero image",
        "description": "Description",
        "value_proposition": "Value proposition",
        "warranty_or_return_policy": "Warranty / return policy",
        "shipping_info": "Shipping info",
        "trust_signals": "Trust signals",
        "reviews_or_social_proof": "Reviews / social proof",
        "call_to_action": "Call to action",
    },
    "tr": {
        "title": "Baslik",
        "price": "Fiyat",
        "hero_image": "Hero gorseli",
        "description": "Aciklama",
        "value_proposition": "Deger onerisi",
        "warranty_or_return_policy": "Garanti / iade",
        "shipping_info": "Kargo bilgisi",
        "trust_signals": "Guven sinyalleri",
        "reviews_or_social_proof": "Yorumlar / sosyal kanit",
        "call_to_action": "CTA",
    },
}


def main() -> None:
    """Render the BuyerLab AI dashboard."""
    _load_env_for_indicator()
    st.set_page_config(
        page_title="BuyerLab AI",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()
    _initialize_session_state()

    _render_sidebar()
    _render_header()

    if st.session_state.get("last_error"):
        st.error(st.session_state["last_error"])

    results = st.session_state.get("results")
    if results:
        _render_dashboard(results)
    else:
        _render_empty_state()


def load_sample_products() -> list[dict[str, Any]]:
    """Load sample products for quick demos."""
    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _language_code() -> str:
    """Return the active UI language code."""
    return LANGUAGE_OPTIONS.get(st.session_state.get("language", "English"), "en")


def _t(key: str) -> str:
    """Translate a static UI label."""
    language = _language_code()
    return UI_TEXT[language].get(key, UI_TEXT["en"].get(key, key))


def _localized_label(value: Any) -> str:
    """Translate compact enum-like labels while leaving free text untouched."""
    text = str(value or "")
    language = _language_code()
    return LABELS.get(language, {}).get(text, LABELS["en"].get(text, text))


def _persona_name(name: str) -> str:
    """Translate known buyer persona names."""
    return PERSONA_NAMES.get(_language_code(), {}).get(name, name)


def _category_label(category: str) -> str:
    """Translate normalized category labels for fixed-choice inputs."""
    normalized = normalize_category(category)
    return CATEGORY_LABELS.get(_language_code(), {}).get(
        normalized,
        normalized.replace("_", " ").title(),
    )


def _section_name(name: str) -> str:
    """Translate known product page section names."""
    text = str(name or "")
    return SECTION_LABELS.get(_language_code(), {}).get(
        text,
        text.replace("_", " ").title(),
    )


def _display_label(value: Any) -> str:
    """Translate compact labels and known persona names for display."""
    return _persona_name(_localized_label(value))


def _render_sidebar() -> None:
    """Render the product brief wizard and simulation action."""
    language_names = list(LANGUAGE_OPTIONS)
    current_language = st.session_state.get("language", "English")
    if current_language not in language_names:
        current_language = "English"
        st.session_state["language"] = current_language
    st.sidebar.radio(
        _t("language"),
        language_names,
        index=language_names.index(current_language),
        key="language",
        horizontal=True,
    )

    st.sidebar.markdown(f"## {_t('brief_wizard')}")
    st.sidebar.caption(_t("brief_help"))

    _render_sample_loader()
    _render_product_identity_inputs()
    _render_pricing_inputs()
    _render_trust_inputs()
    _render_content_inputs()

    st.sidebar.divider()
    run_clicked = st.sidebar.button(_t("run"), type="primary", use_container_width=True)
    if run_clicked:
        product = _product_from_inputs()
        if not product.title:
            st.session_state["last_error"] = _t("missing_title")
            st.rerun()
        _run_dashboard_simulation(product)
        st.rerun()

    _render_environment_notice()


def _render_sample_loader() -> None:
    """Render a separate demo sample loader."""
    st.sidebar.markdown(f"### {_t('sample_loader')}")
    st.sidebar.caption(_t("sample_help"))
    samples = load_sample_products()
    sample_names = [sample.get("name", sample.get("title", "Sample")) for sample in samples]
    if st.session_state.get("selected_sample") not in sample_names:
        st.session_state["selected_sample"] = sample_names[0]
    selected_sample = st.sidebar.selectbox(
        _t("sample_product"),
        sample_names,
        key="selected_sample",
    )

    if st.sidebar.button(_t("load_sample"), use_container_width=True):
        sample = samples[sample_names.index(selected_sample)]
        _load_product_into_state(sample)
        st.session_state["last_error"] = ""
        st.rerun()
    st.sidebar.divider()


def _render_product_identity_inputs() -> None:
    """Render product identity fields."""
    st.sidebar.markdown(f"### {_t('identity')}")
    st.sidebar.text_input(_t("brand"), key="brand")
    st.sidebar.text_input(_t("model"), key="model")
    st.sidebar.text_input(_t("product_type"), key="product_type")
    _normalize_category_input()
    st.sidebar.selectbox(
        _t("category"),
        CATEGORY_OPTIONS,
        key="product_category",
        format_func=_category_label,
    )
    _normalize_market_segment_input()
    st.sidebar.selectbox(
        _t("market_segment"),
        MARKET_SEGMENTS,
        key="market_segment",
        format_func=_localized_label,
    )
    st.sidebar.text_area(_t("intended_use_case"), key="intended_use_case", height=70)


def _render_pricing_inputs() -> None:
    """Render pricing and competitor context fields."""
    st.sidebar.markdown(f"### {_t('pricing')}")
    st.sidebar.number_input(_t("price"), min_value=0.0, step=1.0, key="product_price")
    _normalize_currency_input()
    st.sidebar.selectbox(
        _t("currency"),
        _currency_options(),
        key="product_currency",
    )
    st.sidebar.text_input(_t("local_market"), key="local_market")
    st.sidebar.text_input(_t("competitor_name"), key="competitor_name")
    st.sidebar.number_input(
        _t("competitor_price"),
        min_value=0.0,
        step=1.0,
        key="competitor_price",
    )
    st.sidebar.text_area(_t("competitor_strengths"), key="competitor_strengths", height=70)
    st.sidebar.text_area(_t("competitor_weaknesses"), key="competitor_weaknesses", height=70)
    st.sidebar.text_area(_t("our_differentiator"), key="our_differentiator", height=70)


def _render_trust_inputs() -> None:
    """Render proof, trust, and risk fields."""
    st.sidebar.markdown(f"### {_t('trust')}")
    st.sidebar.text_area(_t("warranty"), key="warranty_or_return_policy", height=75)
    st.sidebar.text_area(_t("shipping"), key="shipping_info", height=75)
    st.sidebar.text_area(_t("trust_signals"), key="trust_signals", height=85)
    st.sidebar.text_area(_t("proof_assets"), key="proof_assets", height=85)
    st.sidebar.text_area(_t("known_limitations"), key="known_limitations", height=85)


def _render_content_inputs() -> None:
    """Render product page content fields."""
    st.sidebar.markdown(f"### {_t('content')}")
    st.sidebar.text_input(_t("product_title"), key="product_title")
    st.sidebar.text_area(_t("value_proposition"), key="value_proposition", height=85)
    st.sidebar.text_area(_t("product_description"), key="product_description", height=120)
    st.sidebar.text_area(_t("social_proof"), key="reviews_or_social_proof", height=75)
    st.sidebar.text_input(_t("cta"), key="call_to_action")
    st.sidebar.text_area(_t("image_notes"), key="image_notes", height=75)


def _render_header() -> None:
    """Render the dashboard header."""
    mock_mode = os.getenv("BUYERLAB_MOCK_MODE", "").strip().lower() == "true"
    mock_badge = "<span class='status-pill mock'>Mock mode active</span>" if mock_mode else ""
    st.markdown(
        f"""
        <section class="hero">
          <div class="hero-copy-block">
            <p class="eyebrow">{_escape(_t("app_eyebrow"))}</p>
            <h1>BuyerLab AI</h1>
            <p class="tagline">{_escape(_t("tagline"))}</p>
            <p class="hero-copy">{_escape(_t("hero_copy"))}</p>
            <p class="hero-disclaimer">{_escape(_t("hero_disclaimer"))}</p>
          </div>
          <div class="hero-badges">
            <span class="status-pill">pre-launch audit</span>
            <span class="status-pill">launch readiness</span>
            <span class="status-pill">heuristic local price perception</span>
            {mock_badge}
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_empty_state() -> None:
    """Render the pre-run dashboard preview."""
    st.markdown(
        f"""
        <div class="empty-panel">
          <h3>{_escape(_t("empty_title"))}</h3>
          <p>{_escape(_t("empty_copy"))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_persona_cards([])


def _render_dashboard(results: dict[str, Any]) -> None:
    """Render simulation results in product-grade audit tabs."""
    before_state: SimulationState = results["before_state"]
    after_state: SimulationState = results["after_state"]
    buyer_loss_analysis: list[dict[str, Any]] = results["buyer_loss_analysis"]
    attention_map: AttentionMapReport = results["attention_map"]
    suggestion = results["optimization"]
    comparison: dict[str, Any] = results["comparison"]

    tabs = st.tabs(
        [
            _t("tab_launch"),
            _t("tab_personas"),
            _t("tab_market"),
            _t("tab_category"),
            _t("tab_friction"),
            _t("tab_fix"),
            _t("tab_compare"),
        ]
    )

    with tabs[0]:
        _render_launch_readiness(before_state.final_report, comparison)
    with tabs[1]:
        _render_persona_cards(before_state.first_round_responses, before_state.final_report)
        _render_buyer_loss_analysis(buyer_loss_analysis)
        _render_debate_terminal(before_state)
    with tabs[2]:
        _render_market_context(before_state)
    with tabs[3]:
        _render_category_audit(before_state)
    with tabs[4]:
        _render_attention_map(attention_map)
    with tabs[5]:
        _render_optimization(suggestion)
    with tabs[6]:
        _render_before_after(comparison, after_state)


def _render_launch_readiness(
    final_report: SimulationReport | None,
    comparison: dict[str, Any],
) -> None:
    """Render launch readiness summary that is readable in under 10 seconds."""
    if final_report is None:
        st.info(_t("judge_missing"))
        return

    cols = st.columns(4)
    cols[0].metric(_t("launch_score"), final_report.launch_readiness_score)
    cols[1].metric(_t("launch_status"), _localized_label(final_report.launch_status))
    cols[2].metric(_t("simulated_score"), comparison["before_score"])
    cols[3].metric(_t("main_blocker"), _short_metric(final_report.main_blocker))

    col1, col2 = st.columns([1, 1])
    with col1:
        _audit_panel(_t("main_blocker"), final_report.main_blocker or final_report.summary)
        _audit_panel(_t("executive_verdict"), final_report.executive_verdict or final_report.summary)
        if final_report.buyer_loss_summary:
            _audit_panel("Buyer Loss Summary", final_report.buyer_loss_summary)
        if final_report.launch_decision_summary:
            _audit_panel("Launch Decision Summary", final_report.launch_decision_summary)
    with col2:
        st.markdown(f"#### {_t('next_actions')}")
        _render_list(final_report.next_best_actions or final_report.top_action_items)
        st.markdown(f"#### {_t('required_fixes')}")
        _render_list(final_report.required_fix_before_launch or final_report.top_conversion_blockers)

    risk_cols = st.columns(4)
    risk_cols[0].metric("Trust Gap", final_report.trust_risk_score)
    risk_cols[1].metric("Price Justification Gap", final_report.price_resistance_score)
    risk_cols[2].metric("Clarity", final_report.clarity_score)
    risk_cols[3].metric("Return Risk", final_report.return_risk_score)
    st.caption(_t("score_caption"))


def _render_persona_cards(
    responses: list[AgentResponse],
    final_report: SimulationReport | None = None,
) -> None:
    """Render buyer persona cards."""
    personas = get_default_personas()
    responses_by_persona = {response.persona_id: response for response in responses}
    verdicts_by_name = _persona_verdicts_by_name(final_report)
    cols = st.columns(4)

    for index, persona in enumerate(personas):
        response = responses_by_persona.get(persona.id)
        verdict = verdicts_by_name.get(persona.name, {})
        with cols[index]:
            if response:
                objection = response.objections[0] if response.objections else _t("no_objection")
                business_impact = verdict.get(
                    "business_impact",
                    _business_impact(response.decision, response.purchase_intent),
                )
                _html_card(
                    title=_persona_name(persona.name),
                    body=(
                        f"<span class='badge {response.decision}'>{_escape(_localized_label(response.decision))}</span>"
                        f"<div class='intent'>{response.purchase_intent}/100 {_escape(_t('intent'))}</div>"
                        f"<div class='confidence'>{response.confidence}/100 {_escape(_t('confidence'))}</div>"
                        f"<p>{_escape(response.main_reason)}</p>"
                        f"<small><b>{_escape(_t('top_objection'))}:</b> {_escape(objection)}</small>"
                        f"<small><b>{_escape(_t('suggested_fix'))}:</b> {_escape(response.suggested_fix)}</small>"
                        f"<small><b>{_escape(_t('business_impact'))}:</b> {_escape(_localized_label(business_impact))}</small>"
                    ),
                )
            else:
                _html_card(
                    title=_persona_name(persona.name),
                    body=(
                        "<span class='badge pending'>pending</span>"
                        f"<p>{_escape(persona.decision_style)}</p>"
                        f"<small>{_escape(_t('waiting'))}</small>"
                    ),
                )


def _render_buyer_loss_analysis(buyer_loss_analysis: list[dict[str, Any]]) -> None:
    """Render buyer loss analysis rows."""
    st.markdown(f"### {_t('buyer_loss')}")
    if not buyer_loss_analysis:
        st.info(_t("buyer_loss"))
        return

    rows = [
        {
            "Persona": _persona_name(row.get("persona_name", row.get("persona_id", ""))),
            _t("decision"): _localized_label(row.get("final_decision", "")),
            _t("intent"): row.get("purchase_intent", 0),
            "Business impact": _localized_label(row.get("business_impact", "")),
            _t("main_blocker"): row.get("main_loss_reason", ""),
            _t("suggested_fix"): row.get("suggested_fix", ""),
        }
        for row in buyer_loss_analysis
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_debate_terminal(state: SimulationState) -> None:
    """Render debate history as a compact market terminal."""
    st.markdown(f"### {_t('terminal')}")
    if not state.debate_history:
        st.info(_t("no_debate"))
        return

    lines = []
    for turn in state.debate_history:
        lines.append(
            "<div class='terminal-line'>"
            f"<span>{_escape(_persona_name(turn.speaker))}</span>"
            f"<em>{_escape(_localized_label(turn.stance))}</em>"
            f"{_escape(turn.message)}"
            "</div>"
        )
    st.markdown(f"<div class='terminal'>{''.join(lines)}</div>", unsafe_allow_html=True)


def _render_market_context(state: SimulationState) -> None:
    """Render local price perception and competitor context."""
    product = state.product
    final_report = state.final_report
    price_report = analyze_local_price_perception(product)
    competitor_gap = analyze_competitor_gap(product)

    st.caption(_t("market_note"))
    cols = st.columns(4)
    cols[0].metric(_t("local_market"), price_report.local_market)
    cols[1].metric(_t("price_band"), _localized_label(price_report.price_band))
    cols[2].metric(_t("value_risk"), price_report.perceived_value_risk)
    cols[3].metric(_t("currency"), price_report.currency)

    col1, col2 = st.columns([1, 1])
    with col1:
        _audit_panel(
            "Price Justification Verdict",
            (
                final_report.price_justification_verdict
                if final_report and final_report.price_justification_verdict
                else price_report.pricing_comment
            ),
        )
        _audit_panel(_t("price_positioning"), price_report.suggested_price_positioning)
        st.markdown(f"#### {_t('required_value_proofs')}")
        _render_pills(price_report.required_value_proofs or [_t("no_objection")])
        st.markdown(f"#### {_t('expected_questions')}")
        _render_list(price_report.expected_customer_questions)

    with col2:
        if _has_competitor_context(product.competitor_context):
            _audit_panel(
                "Competitor Gap Verdict",
                (
                    final_report.competitor_gap_verdict
                    if final_report and final_report.competitor_gap_verdict
                    else competitor_gap.value_gap_summary
                ),
            )
            _audit_panel(_t("competitor_gap"), competitor_gap.value_gap_summary)
            _audit_panel("Competitor positioning", competitor_gap.competitor_positioning_comment)
            st.markdown(f"#### {_t('proofs_to_win')}")
            _render_list(competitor_gap.required_proofs_to_win)
        else:
            st.info(_t("no_competitor"))


def _render_category_audit(state: SimulationState) -> None:
    """Render category-specific launch readiness checks."""
    st.caption(_t("category_audit_note"))
    rows_source = (
        state.final_report.category_expectation_check
        if state.final_report and state.final_report.category_expectation_check
        else build_category_expectation_check(state.product)
    )
    rows = []
    for row in rows_source:
        rows.append(
            {
                _t("required_field"): row.get("field_name", row.get("field", "")),
                _t("status"): _localized_label(row.get("status", "")),
                _t("impact"): _localized_label(row.get("impact", "")),
                _t("explanation"): row.get("explanation", row.get("note", "")),
                _t("suggested_fix"): row.get("suggested_fix", ""),
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)
    missing_rows = [
        row
        for row in rows_source
        if row.get("status") in {"missing", "weak"}
    ]
    if missing_rows:
        st.markdown(f"#### {_t('required_fixes')}")
        _render_list([
            row.get("suggested_fix", "")
            for row in missing_rows
            if row.get("suggested_fix")
        ])


def _render_attention_map(attention_map: AttentionMapReport) -> None:
    """Render AI-simulated attention and conversion friction map."""
    st.caption(_t("attention_caption"))
    rows = []
    for score in attention_map.section_scores:
        priority = get_section_priority_label(score.attention_score, score.friction_score)
        rows.append(
            {
                _t("section"): _section_name(score.section_name),
                _t("attention"): score.attention_score,
                _t("friction"): score.friction_score,
                _t("priority"): _localized_label(priority),
                _t("sentiment"): _localized_label(score.sentiment),
                _t("reason"): score.reason,
                _t("suggested_fix"): score.suggested_fix,
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    cols = st.columns(3)
    cols[0].metric(_t("strongest"), _section_name(attention_map.strongest_section))
    cols[1].metric(_t("weakest"), _section_name(attention_map.weakest_section))
    cols[2].metric(_t("highest_friction"), _section_name(attention_map.highest_friction_section))
    st.info(attention_map.summary)


def _render_optimization(suggestion: Any) -> None:
    """Render optimized product page fix pack."""
    col1, col2 = st.columns([1, 1])
    with col1:
        _audit_panel(_t("optimized_title"), suggestion.title)
        _audit_panel(_t("improved_value"), suggestion.value_proposition)
        _audit_panel(_t("cta_suggestion"), suggestion.call_to_action)
    with col2:
        _audit_panel(_t("optimized_description"), suggestion.description)
        _audit_panel(_t("warranty_improvement"), suggestion.warranty_or_return_policy)
        _audit_panel(_t("shipping_improvement"), suggestion.shipping_info)
        st.markdown(f"#### {_t('improved_trust')}")
        _render_pills(suggestion.trust_signals)
        trust_checklist = getattr(suggestion, "trust_proof_checklist", [])
        if trust_checklist:
            st.markdown("#### Trust Proof Checklist")
            _render_list(trust_checklist)

    competitor_suggestion = getattr(suggestion, "competitor_comparison_suggestion", "")
    missing_checklist = getattr(suggestion, "missing_information_checklist", [])
    if competitor_suggestion or missing_checklist:
        col_comp, col_missing = st.columns([1, 1])
        with col_comp:
            _audit_panel("Competitor Comparison Suggestion", competitor_suggestion)
        with col_missing:
            st.markdown("#### Missing Information Checklist")
            _render_list(missing_checklist)

    col3, col4 = st.columns([1, 1])
    with col3:
        st.markdown(f"#### {_t('faq')}")
        _render_list(suggestion.faq_items)
    with col4:
        st.markdown(f"#### {_t('change_summary')}")
        _render_list(suggestion.change_summary)


def _render_before_after(comparison: dict[str, Any], after_state: SimulationState) -> None:
    """Render before-after simulation comparison."""
    col1, col2, col3 = st.columns(3)
    col1.metric(_t("before_score"), comparison["before_score"])
    col2.metric(_t("after_score"), comparison["after_score"])
    col3.metric(_t("score_delta"), comparison["score_delta"])

    col4, col5 = st.columns(2)
    with col4:
        st.markdown(f"#### {_t('improved_sections')}")
        improved_sections = [_section_name(section) for section in comparison["improved_sections"]]
        _render_pills(improved_sections or [_t("no_section_lift")])
    with col5:
        st.markdown(f"#### {_t('remaining_risks')}")
        _render_list(comparison["remaining_risks"])

    st.info(comparison["summary"])
    if after_state.final_report:
        st.caption(_t("score_caption"))


def _run_dashboard_simulation(product: ProductInput) -> None:
    """Run the full before-after dashboard workflow with readable error handling."""
    st.session_state["last_error"] = ""
    try:
        with st.spinner(_t("running")):
            before_state = run_simulation(product)
            before_report = before_state.final_report or SimulationReport(
                summary=_t("score_unavailable")
            )
            buyer_loss_analysis = analyze_buyer_losses(before_state.first_round_responses)
            attention_map = generate_attention_map(
                product,
                before_state.first_round_responses,
                buyer_loss_analysis,
            )
            before_state.attention_map = attention_map

            optimization = generate_optimized_product(
                product,
                before_report,
                attention_map,
                buyer_loss_analysis,
            )
            optimized_product = apply_optimization_to_product(product, optimization)
            after_state = run_simulation(optimized_product)
            after_loss_analysis = analyze_buyer_losses(after_state.first_round_responses)
            after_attention_map = generate_attention_map(
                optimized_product,
                after_state.first_round_responses,
                after_loss_analysis,
            )
            after_state.attention_map = after_attention_map
            after_state.after_score = (
                after_state.final_report.simulated_conversion_score
                if after_state.final_report
                else None
            )

            comparison = compare_before_after(before_state, after_state)

        st.session_state["results"] = {
            "before_state": before_state,
            "buyer_loss_analysis": buyer_loss_analysis,
            "attention_map": attention_map,
            "optimization": optimization,
            "optimized_product": optimized_product,
            "after_state": after_state,
            "comparison": comparison,
        }
    except Exception as exc:
        st.session_state["last_error"] = (
            f"{_t('simulation_failed')} {str(exc).splitlines()[0]}"
        )


def _product_from_inputs() -> ProductInput:
    """Create ProductInput from sidebar fields."""
    category = normalize_category(st.session_state["product_category"])
    currency = normalize_currency(st.session_state["product_currency"])
    competitor_context = _competitor_from_inputs(currency)
    intended_use_case = st.session_state["intended_use_case"].strip()

    return ProductInput(
        brand=st.session_state["brand"].strip(),
        model=st.session_state["model"].strip(),
        product_type=st.session_state["product_type"].strip(),
        title=st.session_state["product_title"].strip(),
        category=category,
        normalized_category=category,
        market_segment=st.session_state["market_segment"],
        intended_use_case=intended_use_case,
        local_market=st.session_state["local_market"].strip() or "Türkiye",
        price=float(st.session_state["product_price"]),
        currency=currency if currency != "UNKNOWN" else "TRY",
        description=st.session_state["product_description"].strip(),
        target_audience=intended_use_case,
        value_proposition=st.session_state["value_proposition"].strip(),
        warranty_or_return_policy=st.session_state["warranty_or_return_policy"].strip(),
        shipping_info=st.session_state["shipping_info"].strip(),
        trust_signals=_parse_list(st.session_state["trust_signals"], limit=8),
        reviews_or_social_proof=st.session_state["reviews_or_social_proof"].strip(),
        call_to_action=st.session_state["call_to_action"].strip(),
        image_notes=st.session_state["image_notes"].strip() or None,
        competitor_context=competitor_context,
        proof_assets=_parse_list(st.session_state["proof_assets"], limit=8),
        known_limitations=_parse_list(st.session_state["known_limitations"], limit=8),
    )


def _competitor_from_inputs(currency: str) -> CompetitorContext | None:
    """Build optional competitor context from seller-provided fields."""
    competitor_name = st.session_state["competitor_name"].strip()
    competitor_price = float(st.session_state["competitor_price"] or 0.0)
    competitor_strengths = _parse_list(st.session_state["competitor_strengths"], limit=6)
    competitor_weaknesses = _parse_list(st.session_state["competitor_weaknesses"], limit=6)
    our_differentiator = st.session_state["our_differentiator"].strip()

    if not any(
        [
            competitor_name,
            competitor_price > 0,
            competitor_strengths,
            competitor_weaknesses,
            our_differentiator,
        ]
    ):
        return None

    return CompetitorContext(
        competitor_name=competitor_name,
        competitor_price=competitor_price if competitor_price > 0 else None,
        competitor_currency=currency if currency != "UNKNOWN" else "TRY",
        competitor_strengths=competitor_strengths,
        competitor_weaknesses=competitor_weaknesses,
        our_differentiator=our_differentiator,
    )


def _load_product_into_state(sample: dict[str, Any]) -> None:
    """Load sample product values into sidebar session state."""
    competitor = sample.get("competitor_context") or {}
    st.session_state["brand"] = sample.get("brand", "")
    st.session_state["model"] = sample.get("model", "")
    st.session_state["product_type"] = sample.get("product_type", "")
    st.session_state["product_category"] = normalize_category(
        sample.get("normalized_category") or sample.get("category", "")
    )
    st.session_state["market_segment"] = _normalize_market_segment(
        sample.get("market_segment", "mid_range")
    )
    st.session_state["intended_use_case"] = sample.get("intended_use_case", "")
    st.session_state["product_price"] = float(sample.get("price", 0.0))
    st.session_state["product_currency"] = normalize_currency(sample.get("currency", "TRY"))
    st.session_state["local_market"] = sample.get("local_market", "") or "Türkiye"
    st.session_state["competitor_name"] = competitor.get("competitor_name", "")
    st.session_state["competitor_price"] = float(competitor.get("competitor_price") or 0.0)
    st.session_state["competitor_strengths"] = "\n".join(
        competitor.get("competitor_strengths", [])
    )
    st.session_state["competitor_weaknesses"] = "\n".join(
        competitor.get("competitor_weaknesses", [])
    )
    st.session_state["our_differentiator"] = competitor.get("our_differentiator", "")
    st.session_state["warranty_or_return_policy"] = sample.get(
        "warranty_or_return_policy",
        "",
    )
    st.session_state["shipping_info"] = sample.get("shipping_info", "")
    st.session_state["trust_signals"] = "\n".join(sample.get("trust_signals", []))
    st.session_state["proof_assets"] = "\n".join(sample.get("proof_assets", []))
    st.session_state["known_limitations"] = "\n".join(sample.get("known_limitations", []))
    st.session_state["product_title"] = sample.get("title", sample.get("name", ""))
    st.session_state["value_proposition"] = sample.get("value_proposition", "")
    st.session_state["product_description"] = sample.get("description", "")
    st.session_state["reviews_or_social_proof"] = sample.get("reviews_or_social_proof", "")
    st.session_state["call_to_action"] = sample.get("call_to_action", "")
    st.session_state["image_notes"] = sample.get("image_notes", "")


def _initialize_session_state() -> None:
    """Initialize product brief state once."""
    defaults = {
        "language": "English",
        "selected_sample": "",
        "brand": "",
        "model": "",
        "product_type": "",
        "product_category": "general_product",
        "market_segment": "mid_range",
        "intended_use_case": "",
        "product_price": 0.0,
        "product_currency": "TRY",
        "local_market": "Türkiye",
        "competitor_name": "",
        "competitor_price": 0.0,
        "competitor_strengths": "",
        "competitor_weaknesses": "",
        "our_differentiator": "",
        "warranty_or_return_policy": "",
        "shipping_info": "",
        "trust_signals": "",
        "proof_assets": "",
        "known_limitations": "",
        "product_title": "",
        "value_proposition": "",
        "product_description": "",
        "reviews_or_social_proof": "",
        "call_to_action": "",
        "image_notes": "",
        "last_error": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _normalize_category_input() -> None:
    """Normalize category aliases before rendering fixed category choices."""
    st.session_state["product_category"] = normalize_category(
        st.session_state.get("product_category", "")
    )


def _normalize_market_segment_input() -> None:
    """Normalize market segment aliases before rendering fixed choices."""
    st.session_state["market_segment"] = _normalize_market_segment(
        st.session_state.get("market_segment", "mid_range")
    )


def _normalize_market_segment(segment: str) -> str:
    """Return a safe market segment value."""
    normalized = str(segment or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"mid", "middle", "midrange"}:
        normalized = "mid_range"
    if normalized in {"uppermid", "upper_middle"}:
        normalized = "upper_mid"
    return normalized if normalized in MARKET_SEGMENTS else "mid_range"


def _normalize_currency_input() -> None:
    """Normalize common currency aliases shown in the sidebar."""
    normalized_currency = normalize_currency(st.session_state.get("product_currency", "TRY"))
    st.session_state["product_currency"] = (
        normalized_currency if normalized_currency in CURRENCY_OPTIONS else "TRY"
    )


def _currency_options() -> list[str]:
    """Return fixed currency options with TRY first for local price perception."""
    return CURRENCY_OPTIONS


def _render_environment_notice() -> None:
    """Show a compact API and mock mode status at the end of the wizard."""
    st.sidebar.divider()
    mock_mode = os.getenv("BUYERLAB_MOCK_MODE", "").strip().lower() == "true"
    has_api_key = bool(os.getenv("GEMINI_API_KEY"))

    if mock_mode:
        st.sidebar.success(_t("mock_on"))
    elif not has_api_key:
        st.sidebar.warning(_t("missing_key"))
    else:
        st.sidebar.caption(_t("key_detected"))


def _parse_list(raw_value: str, limit: int = 8) -> list[str]:
    """Parse a short list from newline or comma separated textarea input."""
    values: list[str] = []
    for line in str(raw_value or "").replace(",", "\n").splitlines():
        value = line.strip()
        if value and value not in values:
            values.append(value)
    return values[:limit]


def _has_competitor_context(competitor: CompetitorContext | None) -> bool:
    """Return True when seller-provided competitor context exists."""
    if competitor is None:
        return False
    return any(
        [
            competitor.competitor_name,
            competitor.competitor_price,
            competitor.competitor_strengths,
            competitor.competitor_weaknesses,
            competitor.our_differentiator,
        ]
    )


def _persona_verdicts_by_name(
    final_report: SimulationReport | None,
) -> dict[str, dict[str, Any]]:
    """Index normalized buyer persona verdicts by persona name."""
    if final_report is None:
        return {}
    verdicts: dict[str, dict[str, Any]] = {}
    for verdict in final_report.buyer_persona_verdicts:
        persona_name = str(verdict.get("persona_name", "")).strip()
        if persona_name:
            verdicts[persona_name] = verdict
    return verdicts


def _business_impact(decision: str, purchase_intent: int) -> str:
    """Return a safe fallback business impact for persona cards."""
    if decision == "reject" or purchase_intent < 35:
        return "high"
    if decision == "hesitate" or purchase_intent < 65:
        return "medium"
    return "low"


def _short_metric(value: Any, limit: int = 34) -> str:
    """Keep metric values readable in compact cards."""
    text = " ".join(str(value or "None").split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _render_list(values: list[str] | tuple[str, ...]) -> None:
    """Render concise dashboard list items."""
    if not values:
        st.caption("No items provided.")
        return
    for item in values[:6]:
        st.markdown(f"- {item}")


def _render_pills(values: list[str]) -> None:
    """Render compact labels."""
    safe_values = values or ["None"]
    html = "".join(
        f"<span class='pill'>{_escape(_display_label(value))}</span>" for value in safe_values
    )
    st.markdown(f"<div class='pill-row'>{html}</div>", unsafe_allow_html=True)


def _audit_panel(title: str, body: str) -> None:
    """Render a concise text panel."""
    st.markdown(
        f"""
        <div class="audit-panel">
          <h4>{_escape(title)}</h4>
          <p>{_escape(body or "Not provided.")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _html_card(title: str, body: str) -> None:
    """Render a small dashboard card."""
    st.markdown(
        f"""
        <div class="card">
          <h4>{_escape(title)}</h4>
          {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _load_env_for_indicator() -> None:
    """Load local .env values for sidebar status when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _escape(value: Any) -> str:
    """Escape text for small HTML snippets."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _inject_styles() -> None:
    """Inject compact dark dashboard styling."""
    st.markdown(
        """
        <style>
          .stApp {
            background: #0a0f1f;
            color: #e5e7eb;
          }
          [data-testid="stSidebar"] {
            background: #0f172a;
            border-right: 1px solid rgba(148, 163, 184, 0.18);
          }
          [data-testid="stSidebar"] h3 {
            margin-top: 18px;
            color: #f8fafc;
          }
          .hero {
            align-items: flex-start;
            background: linear-gradient(135deg, #111827 0%, #172033 56%, #0b1224 100%);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 8px;
            display: flex;
            gap: 28px;
            justify-content: space-between;
            margin-bottom: 22px;
            padding: 28px 32px;
          }
          .eyebrow {
            color: #38bdf8;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0;
            margin: 0 0 8px;
            text-transform: uppercase;
          }
          .hero h1 {
            color: #f8fafc;
            font-size: 44px;
            line-height: 1.05;
            margin: 0 0 8px;
          }
          .tagline {
            color: #d1d5db;
            font-size: 19px;
            margin: 0 0 10px;
          }
          .hero-copy {
            color: #94a3b8;
            max-width: 840px;
            margin: 0;
          }
          .hero-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            justify-content: flex-end;
            max-width: 360px;
          }
          .status-pill {
            background: rgba(56, 189, 248, 0.11);
            border: 1px solid rgba(56, 189, 248, 0.24);
            border-radius: 999px;
            color: #bae6fd;
            display: inline-block;
            font-size: 12px;
            font-weight: 700;
            padding: 6px 10px;
          }
          .status-pill.mock {
            background: rgba(34, 197, 94, 0.14);
            border-color: rgba(34, 197, 94, 0.28);
            color: #bbf7d0;
          }
          .empty-panel, .audit-panel, .card {
            background: #111827;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 8px;
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.18);
          }
          .empty-panel {
            margin: 18px 0;
            padding: 22px;
          }
          .audit-panel {
            margin-bottom: 14px;
            min-height: 116px;
            padding: 16px;
          }
          .audit-panel h4, .card h4 {
            color: #f8fafc;
            font-size: 15px;
            margin: 0 0 10px;
          }
          .audit-panel p {
            color: #cbd5e1;
            font-size: 14px;
            line-height: 1.45;
            margin: 0;
          }
          .card {
            min-height: 250px;
            padding: 16px;
          }
          .card p {
            color: #cbd5e1;
            font-size: 14px;
            line-height: 1.45;
            min-height: 68px;
          }
          .card small {
            color: #94a3b8;
            display: block;
            line-height: 1.35;
            margin-top: 8px;
          }
          .badge {
            border-radius: 999px;
            display: inline-block;
            font-size: 12px;
            font-weight: 800;
            padding: 4px 10px;
            text-transform: uppercase;
          }
          .badge.buy {
            background: rgba(34, 197, 94, 0.18);
            color: #bbf7d0;
          }
          .badge.reject {
            background: rgba(239, 68, 68, 0.18);
            color: #fecaca;
          }
          .badge.hesitate, .badge.pending {
            background: rgba(245, 158, 11, 0.18);
            color: #fde68a;
          }
          .intent, .confidence {
            color: #38bdf8;
            font-size: 13px;
            font-weight: 800;
            margin-top: 10px;
          }
          .confidence {
            color: #a7f3d0;
            margin-top: 4px;
          }
          .terminal {
            background: #020617;
            border: 1px solid rgba(56, 189, 248, 0.22);
            border-radius: 8px;
            font-family: Consolas, monospace;
            margin-bottom: 18px;
            padding: 14px;
          }
          .terminal-line {
            border-bottom: 1px solid rgba(148, 163, 184, 0.12);
            color: #cbd5e1;
            font-size: 13px;
            padding: 8px 0;
          }
          .terminal-line:last-child {
            border-bottom: 0;
          }
          .terminal-line span {
            color: #38bdf8;
            font-weight: 800;
            margin-right: 10px;
          }
          .terminal-line em {
            color: #a7f3d0;
            font-style: normal;
            margin-right: 10px;
          }
          .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 12px;
          }
          .pill {
            background: rgba(148, 163, 184, 0.14);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 999px;
            color: #e5e7eb;
            display: inline-block;
            font-size: 12px;
            padding: 5px 9px;
          }
          div[data-testid="stMetric"] {
            background: #111827;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 8px;
            padding: 12px 14px;
          }
          div[data-testid="stTabs"] button {
            font-weight: 700;
          }
          h3, h4 {
            color: #f8fafc;
          }
          @media (max-width: 900px) {
            .hero {
              display: block;
            }
            .hero-badges {
              justify-content: flex-start;
              margin-top: 18px;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
