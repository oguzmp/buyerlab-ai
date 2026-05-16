"""Streamlit dashboard for BuyerLab AI."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import streamlit as st

from src.attention_map import generate_attention_map, get_section_priority_label
from src.graph import run_simulation
from src.judge import analyze_buyer_losses
from src.optimizer import (
    apply_optimization_to_product,
    compare_before_after,
    generate_optimized_product,
)
from src.price_intelligence import normalize_currency
from src.state import (
    AgentResponse,
    AttentionMapReport,
    ProductInput,
    SimulationReport,
    SimulationState,
    get_default_personas,
)


DATA_PATH = Path(__file__).parent / "data" / "sample_products.json"

LANGUAGE_OPTIONS = {
    "English": "en",
    "Türkçe": "tr",
}

UI_TEXT = {
    "en": {
        "sidebar_title": "Product Input",
        "language": "Language",
        "sidebar_help": (
            "Start with a sample, then adjust the details. Short, concrete inputs "
            "produce better buyer feedback."
        ),
        "sample_product": "Sample product",
        "load_sample": "Load sample product",
        "product_title": "Product title",
        "category": "Category",
        "price": "Price",
        "currency": "Currency",
        "target_audience": "Target audience",
        "value_proposition": "Value proposition",
        "product_description": "Product description",
        "warranty": "Warranty / return policy",
        "shipping": "Shipping info",
        "trust_signals": "Trust signals",
        "trust_help": "Use one signal per line, such as secure checkout or real reviews.",
        "social_proof": "Reviews or social proof",
        "cta": "Call to action",
        "image_notes": "Image notes",
        "run": "Run Simulation",
        "missing_title": "Add a product title before running a simulation.",
        "mock_on": "Mock mode is enabled.",
        "missing_key": (
            "GEMINI_API_KEY is missing. Add it or set BUYERLAB_MOCK_MODE=true "
            "for demo-safe deterministic output."
        ),
        "key_detected": "Gemini API key detected.",
        "eyebrow": "Pre-launch conversion lab",
        "tagline": "Test your product with AI buyers before launch.",
        "hero_copy": (
            "BuyerLab AI simulates different buyer personas to identify conversion "
            "blockers before a product page goes live."
        ),
        "step_product": "Product input",
        "step_buyers": "AI buyer round",
        "step_judge": "Judge report",
        "step_optimize": "Optimize and compare",
        "simulation_flow": "Simulation Flow",
        "empty_title": "Ready for first simulation",
        "empty_copy": (
            "Add product details in the left panel, then run the AI buyer test. "
            "The dashboard will show buyer decisions, debate, conversion blockers, "
            "AI-simulated attention friction, and optimization suggestions."
        ),
        "simulated_score": "Simulated conversion score",
        "after_optimization": "After optimization",
        "trust_risk": "Trust risk",
        "price_resistance": "Price resistance",
        "intent": "intent",
        "top_objection": "Top objection",
        "no_objection": "No major objection.",
        "pending": "pending",
        "run_simulation_short": "Run simulation",
        "waiting": "Waiting for buyer evaluation.",
        "terminal": "Live Debate / Market Terminal",
        "no_debate": "No debate history yet.",
        "judge_report": "Judge Report",
        "judge_missing": "Judge report is not available yet.",
        "clarity": "Clarity",
        "return_risk": "Return risk",
        "winning_personas": "Winning personas",
        "lost_personas": "Lost personas",
        "none_yet": "None yet",
        "top_actions": "Top action items",
        "score_caption": "Scores are simulated buyer testing signals, not real market predictions.",
        "loss_analysis": "Lost Buyer Analysis",
        "loss_missing": "No buyer loss analysis available.",
        "attention_title": "Conversion Friction / Attention Map",
        "attention_caption": (
            "AI-simulated buyer attention and conversion friction analysis. "
            "This is not real eye-tracking."
        ),
        "strongest": "Strongest section",
        "weakest": "Weakest section",
        "highest_friction": "Highest friction",
        "optimization": "Optimization Plan",
        "optimized_title": "Optimized title",
        "improved_value": "Improved value proposition",
        "cta_suggestion": "CTA suggestion",
        "optimized_description": "Optimized description",
        "improved_trust": "Improved trust signals",
        "faq": "FAQ suggestions",
        "change_summary": "Change summary",
        "before_after": "Before / After Comparison",
        "before_score": "Before score",
        "after_score": "After score",
        "score_delta": "Score delta",
        "improved_sections": "Improved sections",
        "no_section_lift": "No section lift yet",
        "remaining_risks": "Remaining risks",
        "after_caption": "After score is still a simulated conversion score, not a market prediction.",
        "running": "Running AI buyer simulation...",
        "simulation_failed": "Simulation failed safely:",
        "score_unavailable": "Simulated conversion score unavailable.",
        "persona": "Persona",
        "decision": "Decision",
        "table_intent": "Intent",
        "impact": "Impact",
        "main_reason": "Main reason",
        "suggested_fix": "Suggested fix",
        "section": "Section",
        "attention": "Attention",
        "friction": "Friction",
        "priority": "Priority",
        "sentiment": "Sentiment",
        "reason": "Reason",
    },
    "tr": {
        "sidebar_title": "Ürün Bilgileri",
        "language": "Dil",
        "sidebar_help": (
            "Bir örnekle başlayıp detayları uyarlayın. Kısa ve somut girdiler "
            "daha iyi alıcı geri bildirimi verir."
        ),
        "sample_product": "Örnek ürün",
        "load_sample": "Örnek ürünü yükle",
        "product_title": "Ürün adı",
        "category": "Kategori",
        "price": "Fiyat",
        "currency": "Para birimi",
        "target_audience": "Hedef kitle",
        "value_proposition": "Değer önerisi",
        "product_description": "Ürün açıklaması",
        "warranty": "Garanti / iade politikası",
        "shipping": "Kargo bilgisi",
        "trust_signals": "Güven sinyalleri",
        "trust_help": "Her satıra bir sinyal yazın: güvenli ödeme, gerçek yorumlar gibi.",
        "social_proof": "Yorumlar veya sosyal kanıt",
        "cta": "Eylem çağrısı",
        "image_notes": "Görsel notları",
        "run": "Simülasyonu Çalıştır",
        "missing_title": "Simülasyonu çalıştırmadan önce ürün adı ekleyin.",
        "mock_on": "Mock modu açık.",
        "missing_key": (
            "GEMINI_API_KEY eksik. Bir anahtar ekleyin ya da demo için "
            "BUYERLAB_MOCK_MODE=true kullanın."
        ),
        "key_detected": "Gemini API anahtarı algılandı.",
        "eyebrow": "Lansman öncesi dönüşüm laboratuvarı",
        "tagline": "Ürününüzü lansmandan önce AI alıcılarla test edin.",
        "hero_copy": (
            "BuyerLab AI, ürün sayfası yayına alınmadan önce dönüşüm engellerini "
            "bulmak için farklı alıcı personelerini simüle eder."
        ),
        "step_product": "Ürün girişi",
        "step_buyers": "AI alıcı turu",
        "step_judge": "Judge raporu",
        "step_optimize": "Optimize et ve karşılaştır",
        "simulation_flow": "Simülasyon Akışı",
        "empty_title": "İlk simülasyon için hazır",
        "empty_copy": (
            "Sol panelden ürün bilgilerini ekleyin ve AI alıcı testini başlatın. "
            "Dashboard; alıcı kararlarını, tartışmayı, dönüşüm engellerini, "
            "AI-simüle dikkat sürtünmesini ve optimizasyon önerilerini gösterecek."
        ),
        "simulated_score": "Simüle dönüşüm skoru",
        "after_optimization": "Optimizasyon sonrası",
        "trust_risk": "Güven riski",
        "price_resistance": "Fiyat direnci",
        "intent": "satın alma niyeti",
        "top_objection": "Ana itiraz",
        "no_objection": "Büyük bir itiraz yok.",
        "pending": "bekliyor",
        "run_simulation_short": "Simülasyonu çalıştır",
        "waiting": "Alıcı değerlendirmesi bekleniyor.",
        "terminal": "Canlı Tartışma / Pazar Terminali",
        "no_debate": "Henüz tartışma geçmişi yok.",
        "judge_report": "Judge Raporu",
        "judge_missing": "Judge raporu henüz yok.",
        "clarity": "Netlik",
        "return_risk": "İade riski",
        "winning_personas": "Kazanan personalar",
        "lost_personas": "Kaybedilen personalar",
        "none_yet": "Henüz yok",
        "top_actions": "Öncelikli aksiyonlar",
        "score_caption": "Skorlar simüle alıcı test sinyalleridir; gerçek pazar tahmini değildir.",
        "loss_analysis": "Kaybedilen Alıcı Analizi",
        "loss_missing": "Alıcı kayıp analizi henüz yok.",
        "attention_title": "Dönüşüm Sürtünmesi / Dikkat Haritası",
        "attention_caption": (
            "AI-simüle alıcı dikkati ve dönüşüm sürtünmesi analizi. "
            "Bu gerçek eye-tracking değildir."
        ),
        "strongest": "En güçlü bölüm",
        "weakest": "En zayıf bölüm",
        "highest_friction": "En yüksek sürtünme",
        "optimization": "Optimizasyon Planı",
        "optimized_title": "Optimize ürün başlığı",
        "improved_value": "Geliştirilmiş değer önerisi",
        "cta_suggestion": "CTA önerisi",
        "optimized_description": "Optimize açıklama",
        "improved_trust": "Geliştirilmiş güven sinyalleri",
        "faq": "SSS önerileri",
        "change_summary": "Değişiklik özeti",
        "before_after": "Önce / Sonra Karşılaştırması",
        "before_score": "Önceki skor",
        "after_score": "Sonraki skor",
        "score_delta": "Skor farkı",
        "improved_sections": "İyileşen bölümler",
        "no_section_lift": "Henüz bölüm iyileşmesi yok",
        "remaining_risks": "Kalan riskler",
        "after_caption": "Sonraki skor da simüle dönüşüm skorudur; pazar tahmini değildir.",
        "running": "AI alıcı simülasyonu çalışıyor...",
        "simulation_failed": "Simülasyon güvenli şekilde durdu:",
        "score_unavailable": "Simüle dönüşüm skoru kullanılamıyor.",
        "persona": "Persona",
        "decision": "Karar",
        "table_intent": "Niyet",
        "impact": "Etki",
        "main_reason": "Ana neden",
        "suggested_fix": "Önerilen düzeltme",
        "section": "Bölüm",
        "attention": "Dikkat",
        "friction": "Sürtünme",
        "priority": "Öncelik",
        "sentiment": "Duygu",
        "reason": "Neden",
    },
}

PERSONA_NAMES = {
    "tr": {
        "Skeptic Buyer": "Şüpheci Alıcı",
        "Bargain Hunter": "Fırsat Avcısı",
        "Impulsive Buyer": "Dürtüsel Alıcı",
        "Trust Seeker": "Güven Arayan",
    }
}

LABELS = {
    "tr": {
        "buy": "satın alır",
        "reject": "reddeder",
        "hesitate": "kararsız",
        "support": "destekliyor",
        "oppose": "karşı",
        "neutral": "nötr",
        "high": "yüksek",
        "medium": "orta",
        "low": "düşük",
        "positive": "olumlu",
        "negative": "olumsuz",
        "strong_conversion_area": "güçlü dönüşüm alanı",
        "critical_fix_area": "kritik düzeltme alanı",
        "hidden_risk_area": "gizli risk alanı",
        "low_priority_area": "düşük öncelik",
    }
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
        "title": "Başlık",
        "price": "Fiyat",
        "hero_image": "Hero görseli",
        "description": "Açıklama",
        "value_proposition": "Değer önerisi",
        "warranty_or_return_policy": "Garanti / iade politikası",
        "shipping_info": "Kargo bilgisi",
        "trust_signals": "Güven sinyalleri",
        "reviews_or_social_proof": "Yorumlar / sosyal kanıt",
        "call_to_action": "Eylem çağrısı",
    },
}


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
    language = _language_code()
    text = str(value)
    return LABELS.get(language, {}).get(text, text)


def _persona_name(name: str) -> str:
    """Translate known buyer persona names."""
    language = _language_code()
    return PERSONA_NAMES.get(language, {}).get(name, name)


def _section_name(name: str) -> str:
    """Translate known product page section names."""
    language = _language_code()
    text = str(name)
    return SECTION_LABELS.get(language, {}).get(
        text,
        text.replace("_", " ").title(),
    )


def _display_label(value: Any) -> str:
    """Translate compact labels and known persona names for display."""
    return _persona_name(_localized_label(value))


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


def _render_sidebar() -> None:
    """Render product input controls and simulation actions."""
    language_names = list(LANGUAGE_OPTIONS)
    current_language = st.session_state.get("language", "English")
    if current_language not in language_names:
        current_language = "English"
    st.sidebar.selectbox(
        "Language / Dil",
        language_names,
        index=language_names.index(current_language),
        key="language",
    )

    st.sidebar.markdown(f"## {_t('sidebar_title')}")
    st.sidebar.caption(_t("sidebar_help"))
    _render_environment_notice()

    samples = load_sample_products()
    sample_names = [sample.get("name", sample.get("title", "Sample")) for sample in samples]
    selected_sample = st.sidebar.selectbox(_t("sample_product"), sample_names)

    if st.sidebar.button(_t("load_sample"), use_container_width=True):
        sample = samples[sample_names.index(selected_sample)]
        _load_product_into_state(sample)
        st.session_state["last_error"] = ""
        st.rerun()

    st.sidebar.text_input(_t("product_title"), key="product_title")
    st.sidebar.text_input(_t("category"), key="product_category")
    st.sidebar.number_input(_t("price"), min_value=0.0, step=1.0, key="product_price")
    _normalize_currency_input()
    st.sidebar.selectbox(
        _t("currency"),
        _currency_options(),
        key="product_currency",
    )
    st.sidebar.text_input(_t("target_audience"), key="target_audience")
    st.sidebar.text_area(_t("value_proposition"), key="value_proposition", height=90)
    st.sidebar.text_area(_t("product_description"), key="product_description", height=120)
    st.sidebar.text_area(_t("warranty"), key="warranty_or_return_policy", height=80)
    st.sidebar.text_area(_t("shipping"), key="shipping_info", height=80)
    st.sidebar.text_area(
        _t("trust_signals"),
        key="trust_signals",
        height=90,
        help=_t("trust_help"),
    )
    st.sidebar.text_area(_t("social_proof"), key="reviews_or_social_proof", height=80)
    st.sidebar.text_input(_t("cta"), key="call_to_action")
    st.sidebar.text_area(_t("image_notes"), key="image_notes", height=80)

    run_clicked = st.sidebar.button(
        _t("run"),
        type="primary",
        use_container_width=True,
    )

    if run_clicked:
        product = _product_from_inputs()
        if not product.title:
            st.session_state["last_error"] = _t("missing_title")
            st.rerun()
        _run_dashboard_simulation(product)
        st.rerun()


def _render_header() -> None:
    """Render the dashboard header."""
    st.markdown(
        f"""
        <section class="hero">
          <div>
            <p class="eyebrow">{_escape(_t("eyebrow"))}</p>
            <h1>BuyerLab AI</h1>
            <p class="tagline">{_escape(_t("tagline"))}</p>
            <p class="hero-copy">
              {_escape(_t("hero_copy"))}
            </p>
          </div>
        </section>
        <div class="workflow-row">
          <span>1. {_escape(_t("step_product"))}</span>
          <span>2. {_escape(_t("step_buyers"))}</span>
          <span>3. {_escape(_t("step_judge"))}</span>
          <span>4. {_escape(_t("step_optimize"))}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_empty_state() -> None:
    """Render the pre-run dashboard preview."""
    st.markdown(f"### {_t('simulation_flow')}")
    _render_persona_cards([])

    st.markdown(
        f"""
        <div class="empty-panel">
          <h3>{_escape(_t("empty_title"))}</h3>
          <p>
            {_escape(_t("empty_copy"))}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_dashboard(results: dict[str, Any]) -> None:
    """Render all simulation result sections."""
    before_state: SimulationState = results["before_state"]
    after_state: SimulationState = results["after_state"]
    buyer_loss_analysis = results["buyer_loss_analysis"]
    attention_map: AttentionMapReport = results["attention_map"]
    suggestion = results["optimization"]
    comparison = results["comparison"]

    _render_score_strip(before_state.final_report, comparison)
    st.markdown(f"### {_t('simulation_flow')}")
    _render_persona_cards(before_state.first_round_responses)
    _render_debate_terminal(before_state)
    _render_judge_report(before_state.final_report)
    _render_buyer_loss_analysis(buyer_loss_analysis)
    _render_attention_map(attention_map)
    _render_optimization(suggestion)
    _render_before_after(comparison, after_state)


def _render_score_strip(
    final_report: SimulationReport | None,
    comparison: dict[str, Any],
) -> None:
    """Render top dashboard metrics."""
    before_score = comparison["before_score"]
    after_score = comparison["after_score"]
    score_delta = comparison["score_delta"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(_t("simulated_score"), before_score)
    with col2:
        st.metric(_t("after_optimization"), after_score, delta=score_delta)
    with col3:
        st.metric(_t("trust_risk"), final_report.trust_risk_score if final_report else 0)
    with col4:
        st.metric(_t("price_resistance"), final_report.price_resistance_score if final_report else 0)


def _render_persona_cards(responses: list[AgentResponse]) -> None:
    """Render persona decision cards."""
    personas = get_default_personas()
    responses_by_persona = {response.persona_id: response for response in responses}
    cols = st.columns(4)

    for index, persona in enumerate(personas):
        response = responses_by_persona.get(persona.id)
        with cols[index]:
            if response:
                objection = response.objections[0] if response.objections else _t("no_objection")
                _html_card(
                    title=_persona_name(persona.name),
                    body=(
                        f"<span class='badge {response.decision}'>{_localized_label(response.decision)}</span>"
                        f"<div class='intent'>{response.purchase_intent}/100 {_t('intent')}</div>"
                        f"<p>{_escape(response.main_reason)}</p>"
                        f"<small>{_t('top_objection')}: {_escape(objection)}</small>"
                    ),
                )
            else:
                _html_card(
                    title=_persona_name(persona.name),
                    body=(
                        f"<span class='badge pending'>{_t('pending')}</span>"
                        f"<div class='intent'>{_t('run_simulation_short')}</div>"
                        f"<p>{_escape(persona.decision_style)}</p>"
                        f"<small>{_t('waiting')}</small>"
                    ),
                )


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


def _render_judge_report(final_report: SimulationReport | None) -> None:
    """Render the Judge report."""
    st.markdown(f"### {_t('judge_report')}")
    if final_report is None:
        st.info(_t("judge_missing"))
        return

    cols = st.columns(5)
    metrics = [
        (_t("simulated_score"), final_report.simulated_conversion_score),
        (_t("trust_risk"), final_report.trust_risk_score),
        (_t("price_resistance"), final_report.price_resistance_score),
        (_t("clarity"), final_report.clarity_score),
        (_t("return_risk"), final_report.return_risk_score),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, value)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{_t('winning_personas')}**")
        _render_pills(final_report.winning_personas or [_t("none_yet")])
    with col2:
        st.markdown(f"**{_t('lost_personas')}**")
        _render_pills(final_report.lost_personas or [_t("none_yet")])

    st.markdown(f"**{_t('top_actions')}**")
    for item in final_report.top_action_items:
        st.markdown(f"- {item}")

    st.info(final_report.summary)
    st.caption(_t("score_caption"))


def _render_buyer_loss_analysis(buyer_loss_analysis: list[dict[str, Any]]) -> None:
    """Render buyer loss rows."""
    st.markdown(f"### {_t('loss_analysis')}")
    if not buyer_loss_analysis:
        st.info(_t("loss_missing"))
        return

    rows = [
        {
            _t("persona"): _persona_name(row.get("persona_name", row.get("persona_id", ""))),
            _t("decision"): _localized_label(row.get("final_decision", "")),
            _t("table_intent"): row.get("purchase_intent", 0),
            _t("impact"): _localized_label(row.get("business_impact", "")),
            _t("main_reason"): row.get("main_loss_reason", ""),
            _t("suggested_fix"): row.get("suggested_fix", ""),
        }
        for row in buyer_loss_analysis
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_attention_map(attention_map: AttentionMapReport) -> None:
    """Render AI-simulated attention and conversion friction map."""
    st.markdown(f"### {_t('attention_title')}")
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
    """Render optimized product copy suggestions."""
    st.markdown(f"### {_t('optimization')}")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"**{_t('optimized_title')}**")
        st.write(suggestion.title)
        st.markdown(f"**{_t('improved_value')}**")
        st.write(suggestion.value_proposition)
        st.markdown(f"**{_t('cta_suggestion')}**")
        st.write(suggestion.call_to_action)
    with col2:
        st.markdown(f"**{_t('optimized_description')}**")
        st.write(suggestion.description)
        st.markdown(f"**{_t('improved_trust')}**")
        _render_pills(suggestion.trust_signals)

    st.markdown(f"**{_t('faq')}**")
    for item in suggestion.faq_items:
        st.markdown(f"- {item}")

    st.markdown(f"**{_t('change_summary')}**")
    for item in suggestion.change_summary:
        st.markdown(f"- {item}")


def _render_before_after(comparison: dict[str, Any], after_state: SimulationState) -> None:
    """Render before-after simulation comparison."""
    st.markdown(f"### {_t('before_after')}")
    col1, col2, col3 = st.columns(3)
    col1.metric(_t("before_score"), comparison["before_score"])
    col2.metric(_t("after_score"), comparison["after_score"])
    col3.metric(_t("score_delta"), comparison["score_delta"])

    col4, col5 = st.columns(2)
    with col4:
        st.markdown(f"**{_t('improved_sections')}**")
        improved_sections = [
            _section_name(section) for section in comparison["improved_sections"]
        ]
        _render_pills(improved_sections or [_t("no_section_lift")])
    with col5:
        st.markdown(f"**{_t('remaining_risks')}**")
        for risk in comparison["remaining_risks"]:
            st.markdown(f"- {risk}")

    st.info(comparison["summary"])
    if after_state.final_report:
        st.caption(_t("after_caption"))


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
    return ProductInput(
        title=st.session_state["product_title"].strip(),
        category=st.session_state["product_category"].strip(),
        price=float(st.session_state["product_price"]),
        currency=normalize_currency(st.session_state["product_currency"]) or "USD",
        description=st.session_state["product_description"].strip(),
        target_audience=st.session_state["target_audience"].strip(),
        value_proposition=st.session_state["value_proposition"].strip(),
        warranty_or_return_policy=st.session_state["warranty_or_return_policy"].strip(),
        shipping_info=st.session_state["shipping_info"].strip(),
        trust_signals=_parse_trust_signals(st.session_state["trust_signals"]),
        reviews_or_social_proof=st.session_state["reviews_or_social_proof"].strip(),
        call_to_action=st.session_state["call_to_action"].strip(),
        image_notes=st.session_state["image_notes"].strip() or None,
    )


def _normalize_currency_input() -> None:
    """Normalize common currency aliases shown in the sidebar."""
    raw_currency = st.session_state.get("product_currency", "")
    if not str(raw_currency).strip():
        return

    normalized_currency = normalize_currency(raw_currency)
    if normalized_currency != "UNKNOWN":
        st.session_state["product_currency"] = normalized_currency


def _currency_options() -> list[str]:
    """Return currency options while preserving an unusual existing currency."""
    standard_options = ["USD", "TRY", "EUR", "GBP"]
    current_currency = st.session_state.get("product_currency", "USD")
    normalized_currency = normalize_currency(current_currency)
    if normalized_currency in {"", "UNKNOWN"}:
        normalized_currency = "USD"
        st.session_state["product_currency"] = normalized_currency

    if normalized_currency not in standard_options:
        return [normalized_currency, *standard_options]
    return standard_options


def _load_product_into_state(sample: dict[str, Any]) -> None:
    """Load sample product values into sidebar session state."""
    st.session_state["product_title"] = sample.get("title", sample.get("name", ""))
    st.session_state["product_category"] = sample.get("category", "")
    st.session_state["product_price"] = float(sample.get("price", 0.0))
    st.session_state["product_currency"] = sample.get("currency", "USD")
    st.session_state["target_audience"] = sample.get("target_audience", "")
    st.session_state["value_proposition"] = sample.get("value_proposition", "")
    st.session_state["product_description"] = sample.get("description", "")
    st.session_state["warranty_or_return_policy"] = sample.get(
        "warranty_or_return_policy",
        "",
    )
    st.session_state["shipping_info"] = sample.get("shipping_info", "")
    st.session_state["trust_signals"] = "\n".join(sample.get("trust_signals", []))
    st.session_state["reviews_or_social_proof"] = sample.get("reviews_or_social_proof", "")
    st.session_state["call_to_action"] = sample.get("call_to_action", "")
    st.session_state["image_notes"] = sample.get("image_notes", "")


def _initialize_session_state() -> None:
    """Initialize product input state once."""
    defaults = {
        "language": "English",
        "product_title": "",
        "product_category": "",
        "product_price": 0.0,
        "product_currency": "USD",
        "target_audience": "",
        "value_proposition": "",
        "product_description": "",
        "warranty_or_return_policy": "",
        "shipping_info": "",
        "trust_signals": "",
        "reviews_or_social_proof": "",
        "call_to_action": "",
        "image_notes": "",
        "last_error": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _render_environment_notice() -> None:
    """Show API key and mock mode status."""
    mock_mode = os.getenv("BUYERLAB_MOCK_MODE", "").strip().lower() == "true"
    has_api_key = bool(os.getenv("GEMINI_API_KEY"))

    if mock_mode:
        st.sidebar.success(_t("mock_on"))
    elif not has_api_key:
        st.sidebar.warning(_t("missing_key"))
    else:
        st.sidebar.caption(_t("key_detected"))


def _parse_trust_signals(raw_value: str) -> list[str]:
    """Parse trust signals from a textarea."""
    signals: list[str] = []
    for line in raw_value.replace(",", "\n").splitlines():
        signal = line.strip()
        if signal and signal not in signals:
            signals.append(signal)
    return signals[:8]


def _render_pills(values: list[str]) -> None:
    """Render compact labels."""
    html = "".join(
        f"<span class='pill'>{_escape(_display_label(value))}</span>" for value in values
    )
    st.markdown(f"<div class='pill-row'>{html}</div>", unsafe_allow_html=True)


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
            background: #0b1020;
            color: #e5e7eb;
          }
          [data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid rgba(148, 163, 184, 0.18);
          }
          .hero {
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: linear-gradient(135deg, #111827 0%, #162033 55%, #0f172a 100%);
            padding: 28px 32px;
            border-radius: 8px;
            margin-bottom: 22px;
          }
          .eyebrow {
            color: #38bdf8;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0;
            margin-bottom: 8px;
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
            max-width: 820px;
            margin: 0;
          }
          .workflow-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin: -10px 0 22px;
          }
          .workflow-row span {
            background: rgba(17, 24, 39, 0.86);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 8px;
            color: #dbeafe;
            font-size: 13px;
            font-weight: 700;
            padding: 10px 12px;
            text-align: center;
          }
          .card, .empty-panel {
            min-height: 210px;
            background: #111827;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.18);
          }
          .empty-panel {
            min-height: 0;
            margin-top: 18px;
          }
          .card h4 {
            color: #f8fafc;
            margin: 0 0 14px;
            font-size: 16px;
          }
          .card p {
            color: #cbd5e1;
            font-size: 14px;
            line-height: 1.45;
            min-height: 62px;
          }
          .card small {
            color: #94a3b8;
            line-height: 1.35;
          }
          .badge {
            display: inline-block;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
          }
          .badge.buy {
            color: #bbf7d0;
            background: rgba(34, 197, 94, 0.18);
          }
          .badge.reject {
            color: #fecaca;
            background: rgba(239, 68, 68, 0.18);
          }
          .badge.hesitate, .badge.pending {
            color: #fde68a;
            background: rgba(245, 158, 11, 0.18);
          }
          .intent {
            color: #38bdf8;
            font-size: 13px;
            font-weight: 700;
            margin-top: 10px;
          }
          .terminal {
            background: #020617;
            border: 1px solid rgba(56, 189, 248, 0.22);
            border-radius: 8px;
            padding: 14px;
            font-family: Consolas, monospace;
            margin-bottom: 18px;
          }
          .terminal-line {
            color: #cbd5e1;
            font-size: 13px;
            padding: 8px 0;
            border-bottom: 1px solid rgba(148, 163, 184, 0.12);
          }
          .terminal-line:last-child {
            border-bottom: 0;
          }
          .terminal-line span {
            color: #38bdf8;
            font-weight: 700;
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
          h3 {
            color: #f8fafc;
            margin-top: 22px;
          }
          @media (max-width: 900px) {
            .workflow-row {
              grid-template-columns: repeat(2, minmax(0, 1fr));
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
