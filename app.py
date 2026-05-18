"""Streamlit dashboard for BuyerLab AI."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import streamlit as st

from src.attention_map import generate_attention_map, get_section_priority_label
from src.gemini_client import check_gemini_connection, generate_json
from src.graph import run_simulation
from src.judge import analyze_buyer_losses
from src.launch_readiness import build_category_expectation_check
from src.optimizer import (
    apply_optimization_to_product,
    compare_before_after,
    generate_optimized_product,
)
from src.price_intelligence import analyze_local_price_perception, normalize_category, normalize_currency
from src.state import (
    AgentResponse,
    AttentionMapReport,
    ProductInput,
    SimulationReport,
    SimulationState,
    get_default_personas,
)


DATA_PATH = Path(__file__).parent / "data" / "sample_products.json"
PRIMARY_SAMPLE_IDS = {"wireless-earbuds-demo", "online-course-demo"}
APP_STATE_VERSION = "2026-05-18-clean-seller-report-v3"

LANGUAGE_OPTIONS = {"Türkçe": "tr", "English": "en"}

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
        "tagline_support": "",
        "hero_copy": (
            "BuyerLab AI audits product pages before launch using AI buyer "
            "personas, category standards, local price perception, and "
            "launch readiness signals."
        ),
        "hero_disclaimer": "Designed to find product-page weaknesses before launch.",
        "brief_wizard": "Language and AI Status",
        "brief_help": "Keep the setup simple: choose a language and confirm AI is connected.",
        "workspace_title": "Test a product page in 60 seconds",
        "workspace_help": "Paste the basic product page details. BuyerLab returns a clear launch decision and fix plan without requiring a long form.",
        "core_details": "Required for a quick audit",
        "optional_details": "Optional: make the report sharper",
        "input_mode": "Input mode",
        "quick_mode": "Quick mode",
        "advanced_mode": "Advanced mode",
        "quick_mode_help": "Use only the fields needed for a fast product-page audit.",
        "advanced_mode_help": "Add proof, trust, identity, and page details for a sharper report.",
        "executive_report": "60-Second Report",
        "one_minute_decision": "Launch decision",
        "one_minute_reason": "Why it matters",
        "one_minute_action": "What to fix first",
        "first_three_fixes": "First 3 fixes",
        "expected_lift": "Expected simulated lift",
        "sample_loader": "Ready examples",
        "sample_help": "Use one polished example for a fast report.",
        "sample_product": "Sample product",
        "load_sample": "Load sample",
        "run_sample_report": "Run sample report",
        "sample_report_help": "Loads the wireless earbuds case and runs the full audit.",
        "ai_status": "AI status",
        "ai_test": "Test AI connection",
        "ai_test_passed": "AI connection test passed.",
        "ai_test_failed": "AI connection test failed:",
        "ai_mode": "Mode",
        "ai_model": "Model",
        "decision_sources": "Decision sources",
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
        "run": "Test Product Page",
        "missing_title": "Add a product title before running the pre-launch audit.",
        "mock_on": "Mock mode active",
        "missing_key": (
            "Gemini is not connected. Add GEMINI_API_KEY or set "
            "BUYERLAB_MOCK_MODE=true for test mode."
        ),
        "key_detected": "Gemini connected: live AI outputs will be used.",
        "empty_title": "One minute. One decision. One fix plan.",
        "empty_copy": (
            "The report will show launch readiness, buyer persona reactions, "
            "heuristic local price perception, conversion friction, and a practical fix pack."
        ),
        "tab_report": "60-Second Report",
        "tab_buyers_simple": "Buyer Objections",
        "tab_fix_simple": "Fix Plan",
        "audit_details": "Audit details",
        "quick_demo_title": "Start with a ready example",
        "quick_demo_help": "Run one of the two ready scenarios and get the report immediately.",
        "run_primary_demo": "Run wireless earbuds (Demo)",
        "run_secondary_demo": "Run online course (Demo)",
        "agent_discussion_details": "AI buyer discussion details",
        "three_step_1": "1. Describe the product",
        "three_step_2": "2. Pick category and price",
        "three_step_3": "3. Read the 60-second report",
        "advanced_toggle": "Add optional proof and trust details",
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
        "expected_questions": "Expected customer questions",
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
        "badge_audit": "pre-launch audit",
        "badge_readiness": "launch readiness",
        "badge_price": "heuristic local price perception",
        "buyer_loss_summary": "Buyer Loss Summary",
        "launch_decision_summary": "Launch Decision Summary",
        "brief_completeness": "Brief completeness",
        "analysis_confidence": "Analysis confidence",
        "report_meaning": "What this result means",
        "why_verdict": "Why this verdict?",
        "missing_info_context": "Missing information, not product failure",
        "page_weaknesses": "Product page weaknesses",
        "seller_questions": "Information that would strengthen the report",
        "price_justification_verdict": "Price Justification Verdict",
        "trust_proof_checklist": "Trust Proof Checklist",
        "missing_information_checklist": "Missing Information Checklist",
        "no_items": "No items provided.",
        "not_provided": "Not provided.",
        "market_price_title": "Local TRY Price Perception",
        "flow_caption": "Product Brief -> AI Buyer Evaluation -> Launch Readiness Report -> Friction Map -> Fix Pack -> Before / After",
    },
    "tr": {
        "language": "Language / Dil",
        "app_eyebrow": "Ürün Yayın Öncesi Denetim Paneli",
        "tagline": "Ürününü yayına almadan önce AI müşterilerle test et.",
        "tagline_support": "Test your product with AI buyers before launch.",
        "hero_copy": (
            "BuyerLab AI, ürün sayfalarını yapay zeka müşteri profilleriyle "
            "denetler; müşteri itirazlarını, fiyat sürtünmesini, güven "
            "eksiklerini, fiyat algısını ve yayına çıkmadan önce düzeltilmesi "
            "gereken noktaları raporlar."
        ),
        "hero_disclaimer": "Yayına çıkmadan önce ürün sayfasındaki zayıf noktaları bulmak için tasarlandı.",
        "brief_wizard": "Dil ve AI Durumu",
        "brief_help": "Kurulum basit kalsın: dili seç ve AI bağlantısını kontrol et.",
        "workspace_title": "Ürün sayfasını 60 saniyede test et",
        "workspace_help": "Temel ürün bilgilerini girmen yeterli. BuyerLab uzun form istemeden net yayın kararı ve düzeltme planı verir.",
        "core_details": "Hızlı denetim için gerekli alanlar",
        "optional_details": "Opsiyonel: raporu daha keskinleştir",
        "input_mode": "Giriş modu",
        "quick_mode": "Hızlı mod",
        "advanced_mode": "Gelişmiş mod",
        "quick_mode_help": "Hızlı ürün sayfası denetimi için yalnızca temel alanları kullan.",
        "advanced_mode_help": "Daha keskin rapor için kanıt, güven, kimlik ve sayfa detayları ekle.",
        "executive_report": "1 Dakikalık Rapor",
        "one_minute_decision": "Yayın kararı",
        "one_minute_reason": "Neden önemli?",
        "one_minute_action": "İlk ne düzeltilmeli?",
        "first_three_fixes": "İlk 3 düzeltme",
        "expected_lift": "Beklenen simüle iyileşme",
        "sample_loader": "Hazır örnekler",
        "sample_help": "Sade ve güçlü bir örnekle hızlıca rapor üret.",
        "sample_product": "Örnek ürün",
        "load_sample": "Örneği yükle",
        "run_sample_report": "Örnek Raporu Çalıştır",
        "sample_report_help": "Kablosuz kulaklık örneğini yükler ve tüm denetimi tek tıkla çalıştırır.",
        "ai_status": "AI Durumu",
        "ai_test": "AI Bağlantısını Test Et",
        "ai_test_passed": "AI bağlantı testi başarılı.",
        "ai_test_failed": "AI bağlantı testi başarısız:",
        "ai_mode": "Mod",
        "ai_model": "Model",
        "decision_sources": "Karar Kaynakları",
        "identity": "A) Ürün Kimliği",
        "pricing": "B) Fiyat ve Pazar Bağlamı",
        "trust": "C) Güven ve Kanıtlar",
        "content": "D) Ürün Sayfası İçeriği",
        "brand": "Marka",
        "model": "Model",
        "product_type": "Ürün tipi",
        "category": "Kategori",
        "market_segment": "Pazar segmenti",
        "intended_use_case": "Kullanım amacı",
        "price": "Fiyat",
        "currency": "Para birimi",
        "local_market": "Yerel pazar",
        "our_differentiator": "Bizim farkımız",
        "warranty": "Garanti / iade politikası",
        "shipping": "Kargo bilgisi",
        "trust_signals": "Güven sinyalleri",
        "proof_assets": "Kanıt varlıkları",
        "known_limitations": "Bilinen eksikler",
        "product_title": "Ürün başlığı",
        "value_proposition": "Değer önerisi",
        "product_description": "Ürün açıklaması",
        "social_proof": "Yorumlar veya sosyal kanıt",
        "cta": "Satın alma çağrısı",
        "image_notes": "Görsel notları",
        "run": "Ürün Sayfasını Test Et",
        "missing_title": "Denetimi çalıştırmadan önce ürün başlığı ekle.",
        "mock_on": "Mock mod aktif: test çıktıları kullanılıyor.",
        "missing_key": (
            "Gemini bağlı değil. GEMINI_API_KEY ekle veya test için "
            "BUYERLAB_MOCK_MODE=true kullan."
        ),
        "key_detected": "Gemini bağlı: canlı AI çıktıları kullanılacak.",
        "empty_title": "Bir dakika. Net karar. Uygulanabilir düzeltme planı.",
        "empty_copy": (
            "Rapor; yayına hazırlık, AI müşteri profilleri, yerel TL fiyat algısı, "
            "dönüşüm sürtünmesi ve pratik düzeltme paketini gösterecek."
        ),
        "tab_report": "1 Dakikalık Rapor",
        "tab_buyers_simple": "Müşteri İtirazları",
        "tab_fix_simple": "Düzeltme Planı",
        "audit_details": "Analiz detayları",
        "quick_demo_title": "Hazır örnekle başla",
        "quick_demo_help": "İki hazır senaryodan birini çalıştır ve raporu hemen gör.",
        "run_primary_demo": "Kablosuz Kulaklık (Demo) çalıştır",
        "run_secondary_demo": "Online Kurs (Demo) çalıştır",
        "agent_discussion_details": "AI müşteri tartışması detayları",
        "three_step_1": "1. Ürünü anlat",
        "three_step_2": "2. Kategori ve fiyatı seç",
        "three_step_3": "3. 1 dakikalık raporu oku",
        "advanced_toggle": "İsteğe bağlı kanıt ve güven detayları ekle",
        "tab_launch": "Yayına Hazırlık",
        "tab_personas": "AI Müşteri Profilleri",
        "tab_market": "Pazar Bağlamı",
        "tab_category": "Kategori Denetimi",
        "tab_friction": "Dönüşüm Sürtünme Haritası",
        "tab_fix": "Düzeltme Paketi",
        "tab_compare": "Önce / Sonra",
        "launch_score": "Yayına Hazırlık Skoru",
        "launch_status": "Yayın Durumu",
        "simulated_score": "Simüle Dönüşüm Skoru",
        "main_blocker": "Ana engel",
        "executive_verdict": "Yönetici özeti",
        "next_actions": "Sonraki en iyi aksiyonlar",
        "required_fixes": "Yayına Çıkmadan Önce Gerekli Düzeltmeler",
        "judge_missing": "Judge raporu henüz yok.",
        "decision": "Karar",
        "intent": "satın alma niyeti",
        "confidence": "güven",
        "top_objection": "Ana itiraz",
        "suggested_fix": "Önerilen düzeltme",
        "no_objection": "Büyük itiraz yok.",
        "waiting": "Müşteri değerlendirmesi bekleniyor.",
        "terminal": "Canlı Tartışma",
        "no_debate": "Henüz tartışma yok.",
        "buyer_loss": "Kaybedilen Müşteri Nedenleri",
        "market_note": "TL fiyat algısı, kategori bazlı sezgisel değerlendirme kullanır; döviz çevirisi veya canlı fiyat araştırması yapmaz.",
        "price_band": "Fiyat bandı",
        "value_risk": "Algılanan değer riski",
        "required_value_proofs": "Gerekli değer kanıtları",
        "pricing_comment": "Fiyat yorumu",
        "price_positioning": "Önerilen fiyat konumlandırması",
        "competitor_gap": "Rakip Farkı Analizi",
        "expected_questions": "Beklenen müşteri soruları",
        "category_audit_note": "Kategori denetimi, sezgisel kategori profillerine göre yapılan AI destekli tanı kontrolüdür.",
        "required_field": "Gerekli bilgi alanı",
        "status": "Durum",
        "impact": "Etki",
        "explanation": "Açıklama",
        "business_impact": "İş etkisi",
        "attention_caption": "Bu analiz gerçek göz takibi değildir. AI müşteri profillerine göre simüle edilmiş dikkat ve sürtünme analizidir.",
        "section": "Bölüm",
        "attention": "Dikkat",
        "friction": "Sürtünme",
        "priority": "Öncelik",
        "sentiment": "Duygu",
        "reason": "Neden",
        "strongest": "En güçlü bölüm",
        "weakest": "En zayıf bölüm",
        "highest_friction": "En yüksek sürtünme",
        "optimized_title": "Yeniden yazılan ürün başlığı",
        "optimized_description": "İyileştirilmiş açıklama",
        "improved_value": "İyileştirilmiş değer önerisi",
        "warranty_improvement": "Garanti / iade iyileştirme önerisi",
        "shipping_improvement": "Kargo iyileştirme önerisi",
        "improved_trust": "Güven sinyali önerileri",
        "faq": "SSS önerileri",
        "cta_suggestion": "CTA önerisi",
        "change_summary": "Değişiklik özeti",
        "before_score": "Önceki Simüle Dönüşüm Skoru",
        "after_score": "Sonraki Simüle Dönüşüm Skoru",
        "score_delta": "Skor farkı",
        "improved_sections": "İyileşen bölümler",
        "remaining_risks": "Kalan riskler",
        "no_section_lift": "Henüz bölüm iyileşmesi yok",
        "score_caption": "Bu skor gerçek satış oranı tahmini değildir. Yayın öncesi ürün sayfası zayıflıklarını bulmak için kullanılan AI destekli tanı skorudur.",
        "running": "Yayın öncesi denetim çalışıyor...",
        "simulation_failed": "Simülasyon güvenli şekilde durdu:",
        "score_unavailable": "Simüle dönüşüm skoru kullanılamıyor.",
        "badge_audit": "yayın öncesi denetim",
        "badge_readiness": "yayına hazırlık",
        "badge_price": "yerel TL fiyat algısı",
        "buyer_loss_summary": "Kaybedilen Müşteri Özeti",
        "launch_decision_summary": "Yayın Kararı Özeti",
        "brief_completeness": "Brief Tamlığı",
        "analysis_confidence": "Analiz Güveni",
        "report_meaning": "Bu Sonuç Ne Anlama Geliyor?",
        "why_verdict": "Bu Karar Neden Verildi?",
        "missing_info_context": "Eksik Bilgi, Ürün Hatası Değil",
        "page_weaknesses": "Ürün Sayfası Zayıflıkları",
        "seller_questions": "Raporu Güçlendirecek Bilgiler",
        "price_justification_verdict": "Fiyat Gerekçesi Kararı",
        "trust_proof_checklist": "Güven Kanıtı Kontrol Listesi",
        "missing_information_checklist": "Eksik Bilgi Kontrol Listesi",
        "no_items": "Öğe girilmedi.",
        "not_provided": "Bilgi girilmedi.",
        "market_price_title": "Yerel TL Fiyat Algısı",
        "flow_caption": "Product Brief → AI Müşteri Değerlendirmesi → Yayına Hazırlık Raporu → Sürtünme Haritası → Düzeltme Paketi → Önce / Sonra",
    },
}

PERSONA_NAMES = {
    "tr": {
        "Skeptic Buyer": "Şüpheci Müşteri / Skeptic Buyer",
        "Bargain Hunter": "Fiyat Odaklı Müşteri / Bargain Hunter",
        "Impulsive Buyer": "Dürtüsel Müşteri / Impulsive Buyer",
        "Trust Seeker": "Güven Arayan Müşteri / Trust Seeker",
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
        "ready": "Hazır",
        "needs_fixes": "Düzeltme Gerekli",
        "not_ready": "Hazır Değil",
        "budget": "Ekonomik",
        "mid_range": "Orta segment",
        "upper_mid": "Üst orta",
        "premium": "Premium",
        "irrational": "Mantıksız yüksek",
        "strong_conversion_area": "güçlü dönüşüm alanı",
        "critical_fix_area": "kritik düzeltme alanı",
        "hidden_risk_area": "gizli risk alanı",
        "low_priority_area": "düşük öncelik",
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
        "fashion_shoes": "Moda / Ayakkabı",
        "small_home_appliance": "Küçük Ev Aleti",
        "handmade_bag": "El Yapımı Çanta",
        "digital_service": "Dijital Hizmet",
        "online_course": "Online Kurs",
        "general_product": "Genel Ürün",
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
        "title": "Başlık",
        "price": "Fiyat",
        "hero_image": "Hero görseli",
        "description": "Açıklama",
        "value_proposition": "Değer önerisi",
        "warranty_or_return_policy": "Garanti / iade",
        "shipping_info": "Kargo bilgisi",
        "trust_signals": "Güven sinyalleri",
        "reviews_or_social_proof": "Yorumlar / sosyal kanıt",
        "call_to_action": "CTA",
    },
}

FIELD_KEY_LABELS_TR = {
    "title": "başlık",
    "price": "fiyat",
    "hero_image": "ana görsel",
    "description": "ürün açıklaması",
    "value_proposition": "değer önerisi",
    "warranty_or_return_policy": "garanti / iade",
    "shipping_info": "kargo bilgisi",
    "trust_signals": "güven sinyalleri",
    "reviews_or_social_proof": "yorumlar / sosyal kanıt",
    "call_to_action": "satın alma çağrısı",
}


def main() -> None:
    """Render the BuyerLab AI dashboard."""
    _load_env_for_indicator()
    st.set_page_config(
        page_title="BuyerLab AI Demo",
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
        st.divider()
        _render_product_brief_workspace()
    else:
        _render_product_brief_workspace()
        _render_empty_state()


def load_sample_products() -> list[dict[str, Any]]:
    """Load sample products for quick demos."""
    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _language_code() -> str:
    """Return the active UI language code."""
    return LANGUAGE_OPTIONS.get(st.session_state.get("language", "Türkçe"), "tr")


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


REPORT_TRANSLATIONS = {
    "This product page is not ready to launch.": "Bu ürün sayfası şu haliyle yayına hazır değil.",
    "This product page needs fixes before launch.": "Bu ürün sayfasının yayına çıkmadan önce düzeltilmesi gerekiyor.",
    "This product page is ready to launch in the simulated buyer assessment.": "Simüle müşteri değerlendirmesine göre bu ürün sayfası yayına çıkabilir.",
    "Main blocker:": "Ana engel:",
    "Business impact:": "İş etkisi:",
    "Recommendation:": "Öneri:",
    "Do not launch yet;": "Henüz yayına alma;",
    "Launch only after this fix is completed:": "Yalnızca bu düzeltme tamamlandıktan sonra yayına al:",
    "Launch can proceed, but keep monitoring real buyer behavior.": "Yayına çıkılabilir; ancak gerçek müşteri davranışı ayrıca takip edilmelidir.",
    "Trust proof is weak": "Güven kanıtları zayıf",
    "buyers need real proof assets, warranty clarity, and visible support.": "müşterilerin gerçek kanıtlara, net garanti bilgisine ve görünür desteğe ihtiyacı var.",
    "Missing category-critical information": "Kategori için kritik bilgiler eksik",
    "Price/value justification is weak for the heuristic local price perception band.": "Sezgisel yerel fiyat algısı bandına göre fiyat/değer gerekçesi zayıf.",
    "Competitor gap is not justified with proof or a clear comparison.": "Rakip farkı kanıt veya net karşılaştırma ile açıklanmamış.",
    "Return, warranty, shipping, or delivery terms do not reduce buyer risk enough.": "İade, garanti, kargo veya teslimat bilgileri müşteri riskini yeterince azaltmıyor.",
    "Product value and key claims are not clear enough for a fast launch decision.": "Ürün değeri ve ana iddialar hızlı bir yayın kararı için yeterince net değil.",
    "Emotional appeal or CTA strength is not compelling enough to create urgency.": "Duygusal çekicilik veya CTA gücü aciliyet yaratmak için yeterli değil.",
    "BuyerLab lost the": "BuyerLab şu müşteri profillerini kaybetti:",
    "persona(s). This creates business risk because buyers are likely to hesitate over:": "Bu iş riski yaratır çünkü müşteriler şu konularda tereddüt edebilir:",
    "Decision: Do not launch yet.": "Karar: Henüz yayına alma.",
    "Decision: Launch is possible only after required fixes before launch are completed.": "Karar: Yayına çıkmak ancak gerekli düzeltmeler tamamlandıktan sonra mümkün.",
    "Decision: Launch is possible.": "Karar: Yayına çıkmak mümkün.",
    "Fix trust proof, price justification, and category-critical details first.": "Önce güven kanıtları, fiyat gerekçesi ve kategori-kritik detaylar düzeltilmeli.",
    "Start with:": "İlk adım:",
    "Add real trust signals": "Gerçek güven sinyalleri ekle",
    "exact warranty duration": "net garanti süresi",
    "return conditions": "iade koşulları",
    "support contact": "destek iletişimi",
    "proof assets": "kanıt varlıkları",
    "visible purchase safety": "görünür satın alma güvenliği",
    "Add exact battery life": "Net batarya süresi ekle",
    "warranty period": "garanti süresi",
    "Bluetooth/version compatibility": "Bluetooth/sürüm uyumluluğu",
    "technical specs": "teknik özellikler",
    "return policy": "iade politikası",
    "real usage proof": "gerçek kullanım kanıtı",
    "microphone or sound quality proof": "mikrofon veya ses kalitesi kanıtı",
    "Explain why this product is worth": "Bu ürünün neden",
    "more than": "daha fazla etmeye değer olduğunu açıkla:",
    "Add a short comparison table against the alternative product.": "Alternatif ürüne karşı kısa bir karşılaştırma tablosu ekle.",
    "Collect real proof assets and place them near price and CTA.": "Gerçek kanıtları topla ve fiyat ile CTA yakınına yerleştir.",
    "Complete the category-critical product information.": "Kategori için kritik ürün bilgilerini tamamla.",
    "Rewrite price justification with concrete value proof.": "Fiyat gerekçesini somut değer kanıtlarıyla yeniden yaz.",
    "Add a short competitor comparison using seller-provided facts.": "Satıcının verdiği bilgilerle kısa rakip karşılaştırması ekle.",
    "Make shipping, warranty, and return terms visible.": "Kargo, garanti ve iade koşullarını görünür yap.",
    "Simulated conversion score": "Simüle dönüşüm skoru",
    "simulated conversion score": "simüle dönüşüm skoru",
    "not a real market prediction": "gerçek pazar tahmini değildir",
    "AI-assisted diagnostic score": "AI destekli tanı skoru",
    "AI-simulated buyer attention": "AI destekli müşteri dikkat analizi",
    "conversion friction": "dönüşüm sürtünmesi",
    "lacks category-critical proof": "kategori için kritik kanıtları eksik",
    "specific product evidence": "somut ürün kanıtı",
    "Claims need more concrete proof": "İddialar daha somut kanıt gerektiriyor",
    "needs clearer value proof for the": "şu fiyat bandı için daha net değer kanıtı gerektiriyor:",
    "Defend the price with concrete proof, total cost clarity, and competitor comparison.": "Fiyatı somut kanıt, toplam maliyet netliği ve rakip karşılaştırmasıyla savun.",
    "Price proof is not clear enough": "Fiyat kanıtı yeterince net değil",
    "does not create enough instant desire or visual confidence.": "yeterli anlık istek veya görsel güven oluşturmuyor.",
    "has a clearer emotional hook and CTA.": "daha net bir duygusal çekim ve CTA sunuyor.",
    "Emotional appeal is weak": "Duygusal çekicilik zayıf",
    "Stronger visual promise": "Daha güçlü görsel vaat",
    "Sharper CTA": "Daha net CTA",
    "Make the first screen more concrete, visual, and desire-led without exaggerating claims.": "İlk ekranı iddiaları abartmadan daha somut, görsel ve istek uyandıran hale getir.",
    "does not yet show enough trust proof, warranty clarity, or social proof.": "henüz yeterli güven kanıtı, garanti netliği veya sosyal kanıt göstermiyor.",
    "includes enough trust signals for a first simulated pass.": "ilk simüle değerlendirme için yeterli güven sinyali içeriyor.",
    "Weak trust proof": "Güven kanıtı zayıf",
    "Warranty or return policy needs clarity": "Garanti veya iade politikası netleşmeli",
    "Real trust signals": "Gerçek güven sinyalleri",
    "Warranty and support details": "Garanti ve destek detayları",
    "Add real trust signals, exact warranty/return terms, support details, and proof assets.": "Gerçek güven sinyalleri, net garanti/iade koşulları, destek bilgisi ve kanıt varlıkları ekle.",
    "Group signal": "Grup sinyali",
    "battery life": "pil süresi",
    "technical specifications": "teknik özellikler",
    "compatibility": "uyumluluk",
    "return policy": "iade politikası",
    "real usage proof": "gerçek kullanım kanıtı",
    "microphone or sound quality proof": "mikrofon veya ses kalitesi kanıtı",
    "warranty": "garanti",
    "size guide": "beden rehberi",
    "fit information": "kalıp bilgisi",
    "material": "materyal",
    "real product photos": "gerçek ürün fotoğrafları",
    "scope": "kapsam",
    "delivery time": "teslim süresi",
    "revision policy": "revizyon politikası",
    "portfolio proof": "portfolyo kanıtı",
    "support terms": "destek koşulları",
    "instructor credibility": "eğitmen güvenilirliği",
    "curriculum clarity": "müfredat netliği",
    "learning outcomes": "öğrenme çıktıları",
    "sample lesson/proof": "örnek ders/kanıt",
    "section lacks enough page content for simulated buyers.": "bölümünde simüle müşteriler için yeterli sayfa içeriği yok.",
    "Buyer feedback points to friction around": "Müşteri geri bildirimi şu bölümde sürtünmeye işaret ediyor:",
    "has limited simulated friction.": "sınırlı simüle sürtünme gösteriyor.",
    "Add clear": "Net bilgi ekle:",
    "content before launch.": "içeriği yayından önce tamamlanmalı.",
    "Reduce buyer uncertainty in": "Müşteri belirsizliğini azalt:",
    "Keep": "Koru:",
    "concise and easy to scan.": "kısa ve hızlı taranabilir olsun.",
    "No major remaining risk identified.": "Belirgin kalan risk tespit edilmedi.",
    "Highest simulated friction remains in": "En yüksek simüle sürtünme şu bölümde kalıyor:",
    "Simulated conversion score improved after optimization; rerun with real page constraints before launch.": "Optimizasyon sonrası simüle dönüşüm skoru arttı; yayına çıkmadan önce gerçek sayfa koşullarıyla tekrar test et.",
    "Simulated conversion score was unchanged; focus on remaining buyer risks before another test.": "Simüle dönüşüm skoru değişmedi; yeni testten önce kalan müşteri risklerine odaklan.",
    "Bargain Hunter evaluation failed.": "Fiyat Odaklı Müşteri değerlendirmesi temiz formatta gelmedi; BuyerLab güvenli yerel analiz kullandı.",
    "Impulsive Buyer evaluation failed.": "Dürtüsel Müşteri değerlendirmesi temiz formatta gelmedi; BuyerLab güvenli yerel analiz kullandı.",
    "Trust Seeker evaluation failed.": "Güven Arayan Müşteri değerlendirmesi temiz formatta gelmedi; BuyerLab güvenli yerel analiz kullandı.",
    "Simulation error": "Teknik değerlendirme hatası",
    "Retry this persona evaluation or enable BUYERLAB_MOCK_MODE=true.": "Bu teknik not kullanıcı raporundan gizlenmelidir.",
    "Add verified customer reviews when available.": "Gerçek müşteri yorumu varsa ekle.",
    "Show secure checkout and accepted payment methods.": "Güvenli ödeme ve kabul edilen ödeme yöntemlerini göster.",
    "Link to real return, warranty, and support policies.": "Gerçek iade, garanti ve destek politikalarına bağlantı ver.",
    "Add exact warranty duration and return conditions.": "Net garanti süresi ve iade koşullarını ekle.",
    "Add real support channel and service policy.": "Gerçek destek kanalı ve servis politikasını göster.",
    "Add real proof assets instead of generic quality claims.": "Genel kalite iddiaları yerine gerçek kanıt varlıkları ekle.",
    "Why is this product worth the price?": "Bu ürün neden bu fiyatı hak ediyor?",
    "What proof supports the price positioning?": "Fiyat konumlandırmasını hangi kanıt destekliyor?",
    "Added stronger price and value justification.": "Fiyat ve değer gerekçesi güçlendirildi.",
    "Focused the fix pack on required fixes before launch.": "Düzeltme paketi yayından önce gereken kritik maddelere odaklandı.",
    "Battery life is not specified clearly": "Pil süresi net belirtilmemiş",
    "Warranty is unclear": "Garanti net değil",
    "No microphone test proof": "Mikrofon test kanıtı yok",
    "No real trust signals": "Gerçek güven sinyali yok",
    "Competitor has more social proof": "Rakip daha görünür sosyal kanıta sahip",
    "No size guide": "Beden rehberi yok",
    "Fit information is unclear": "Kalıp bilgisi net değil",
    "Material details are weak": "Materyal detayları zayıf",
    "Return/exchange policy is vague": "İade/değişim politikası belirsiz",
    "Cleaning and maintenance details are missing": "Temizlik ve bakım detayları eksik",
    "Project scope is unclear": "Proje kapsamı net değil",
    "Revision policy is missing": "Revizyon politikası eksik",
    "Delivery timeline is vague": "Teslim süresi belirsiz",
    "Portfolio proof is weak": "Portfolyo kanıtı zayıf",
    "Support terms are unclear": "Destek koşulları net değil",
    "Learning outcomes are unclear": "Öğrenme çıktıları net değil",
    "Refund policy is missing": "İade politikası eksik",
    "No value proof": "Değer kanıtı yok",
    "No trust proof": "Güven kanıtı yok",
    "mid_range": "orta segment",
    "upper_mid": "üst orta",
    "premium": "premium",
    "budget": "ekonomik",
}


def _report_text(value: Any) -> str:
    """Localize generated dashboard text lightly without changing backend data."""
    text = " ".join(str(value or "").split())
    if _language_code() != "tr" or not text:
        return text

    text = _translate_price_gap(text)
    text = _replace_raw_field_names(text)
    for english, turkish in sorted(REPORT_TRANSLATIONS.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(english, turkish)
    text = _replace_raw_field_names(text)
    return text


def _translate_price_gap(text: str) -> str:
    """Translate the most common competitor price-gap sentence shape."""
    pattern = r"(?P<product>.+?) is (?P<gap>[0-9.]+) (?P<currency>[A-Z]+) above (?P<competitor>.+?); the page must prove the difference\."
    match = re.match(pattern, text)
    if not match:
        return text
    return (
        f"{match.group('product')}, {match.group('competitor')} ürününden "
        f"{match.group('gap')} {match.group('currency')} daha pahalı; sayfa bu farkı kanıtlamalı."
    )


def _replace_raw_field_names(text: str) -> str:
    """Replace backend field keys with seller-friendly Turkish labels."""
    for raw_key, label in FIELD_KEY_LABELS_TR.items():
        text = re.sub(rf"\b{re.escape(raw_key)}\b", label, text)
    return text


def _render_sidebar() -> None:
    """Render compact language, demo, and AI status controls."""
    language_names = list(LANGUAGE_OPTIONS)
    current_language = st.session_state.get("language", "Türkçe")
    if current_language not in language_names:
        current_language = "Türkçe"
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

    _render_environment_notice()


def _render_sample_loader() -> None:
    """Render a separate demo sample loader."""
    st.sidebar.markdown(f"### {_t('sample_loader')}")
    st.sidebar.caption(_t("sample_help"))
    all_samples = load_sample_products()
    samples = [sample for sample in all_samples if sample.get("id") in PRIMARY_SAMPLE_IDS] or all_samples
    sample_names = [_sample_display_name(sample) for sample in samples]
    if st.session_state.get("selected_sample") not in sample_names:
        st.session_state["selected_sample"] = sample_names[0]
    selected_sample = st.sidebar.selectbox(
        _t("sample_product"),
        sample_names,
        key="selected_sample",
    )

    if st.sidebar.button(_t("load_sample"), width="stretch"):
        sample = samples[sample_names.index(selected_sample)]
        _load_product_into_state(sample)
        st.session_state["last_error"] = ""
        st.rerun()
    st.sidebar.caption(_t("sample_report_help"))
    if st.sidebar.button(_t("run_sample_report"), type="primary", width="stretch"):
        sample = _primary_demo_sample(all_samples)
        _load_product_into_state(sample)
        st.session_state["last_error"] = ""
        product = _product_from_inputs()
        _run_dashboard_simulation(product)
        st.rerun()
    st.sidebar.divider()


def _render_quick_demo_panel() -> None:
    """Render one-click demo actions in the main workspace."""
    samples = load_sample_products()
    primary_sample = _sample_by_id(samples, "wireless-earbuds-demo")
    secondary_sample = _sample_by_id(samples, "online-course-demo")
    if not primary_sample:
        return

    st.markdown(f"#### {_t('quick_demo_title')}")
    st.caption(_t("quick_demo_help"))
    run_col, load_col = st.columns([1, 1])
    with run_col:
        if st.button(_t("run_primary_demo"), type="secondary", width="stretch"):
            _load_product_into_state(primary_sample)
            st.session_state["last_error"] = ""
            product = _product_from_inputs()
            _run_dashboard_simulation(product)
            st.rerun()
    with load_col:
        if secondary_sample and st.button(_t("run_secondary_demo"), width="stretch"):
            _load_product_into_state(secondary_sample)
            st.session_state["last_error"] = ""
            product = _product_from_inputs()
            _run_dashboard_simulation(product)
            st.rerun()


def _primary_demo_sample(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the strongest single-click demo case."""
    sample = _sample_by_id(samples, "wireless-earbuds-demo")
    if sample:
        return sample
    return samples[0] if samples else {}


def _sample_by_id(samples: list[dict[str, Any]], sample_id: str) -> dict[str, Any]:
    """Find a sample product by id."""
    for sample in samples:
        if sample.get("id") == sample_id:
            return sample
    return {}


def _sample_display_name(sample: dict[str, Any]) -> str:
    """Return a focused, local demo label without exposing every test fixture."""
    sample_id = sample.get("id", "")
    if _language_code() == "tr":
        labels = {
            "wireless-earbuds-demo": "Kablosuz Kulaklık (Demo)",
            "online-course-demo": "Online Kurs (Demo)",
        }
        if sample_id in labels:
            return labels[sample_id]
    return sample.get("name", sample.get("title", "Sample"))


def _input_mode_label(mode: str) -> str:
    """Return localized labels for the quick/advanced input mode switch."""
    return _t("advanced_mode") if mode == "advanced" else _t("quick_mode")


def _render_product_brief_workspace() -> None:
    """Render the main product test workspace instead of a long sidebar form."""
    st.markdown(f"### {_t('workspace_title')}")
    st.caption(_t("workspace_help"))

    _render_quick_demo_panel()

    st.markdown(
        f"""
        <div class="guide-strip">
          <span>{_escape(_t("three_step_1"))}</span>
          <span>{_escape(_t("three_step_2"))}</span>
          <span>{_escape(_t("three_step_3"))}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"#### {_t('core_details')}")
    left, right = st.columns([1.15, 0.85])
    with left:
        st.text_input(_t("product_title"), key="product_title")
        st.text_area(_t("product_description"), key="product_description", height=115)
        st.text_area(_t("value_proposition"), key="value_proposition", height=80)
    with right:
        _normalize_category_input()
        st.selectbox(
            _t("category"),
            CATEGORY_OPTIONS,
            key="product_category",
            format_func=_category_label,
        )
        price_col, currency_col = st.columns([1, 0.62])
        with price_col:
            st.number_input(_t("price"), min_value=0.0, step=1.0, key="product_price")
        with currency_col:
            _normalize_currency_input()
            st.selectbox(_t("currency"), _currency_options(), key="product_currency")

    with st.expander(_t("advanced_toggle"), expanded=False):
        identity_col, trust_col = st.columns(2)
        with identity_col:
            st.markdown(f"##### {_t('identity')}")
            st.text_input(_t("brand"), key="brand")
            st.text_input(_t("model"), key="model")
            st.text_input(_t("product_type"), key="product_type")
            st.text_area(_t("intended_use_case"), key="intended_use_case", height=70)
            _normalize_market_segment_input()
            st.selectbox(
                _t("market_segment"),
                MARKET_SEGMENTS,
                key="market_segment",
                format_func=_localized_label,
            )
            st.text_input(_t("local_market"), key="local_market")
        with trust_col:
            st.markdown(f"##### {_t('trust')}")
            st.text_area(_t("warranty"), key="warranty_or_return_policy", height=70)
            st.text_area(_t("shipping"), key="shipping_info", height=70)
            st.text_area(_t("trust_signals"), key="trust_signals", height=70)
            st.text_area(_t("proof_assets"), key="proof_assets", height=70)
            st.text_area(_t("known_limitations"), key="known_limitations", height=70)

        content_col, visual_col = st.columns(2)
        with content_col:
            st.markdown(f"##### {_t('content')}")
            st.text_area(_t("social_proof"), key="reviews_or_social_proof", height=70)
            st.text_input(_t("cta"), key="call_to_action")
        with visual_col:
            st.markdown("##### Sayfa notları" if _language_code() == "tr" else "##### Page notes")
            st.text_area(_t("image_notes"), key="image_notes", height=70)

    action_col, caption_col = st.columns([0.32, 0.68])
    with action_col:
        run_clicked = st.button(_t("run"), type="primary", width="stretch")
    with caption_col:
        st.caption(_t("flow_caption"))

    if run_clicked:
        product = _product_from_inputs()
        if not product.title:
            st.session_state["last_error"] = _t("missing_title")
            st.rerun()
        _run_dashboard_simulation(product)
        st.rerun()


def _render_header() -> None:
    """Render the dashboard header."""
    mock_mode = os.getenv("BUYERLAB_MOCK_MODE", "").strip().lower() == "true"
    mock_badge = f"<span class='status-pill mock'>{_escape(_t('mock_on'))}</span>" if mock_mode else ""
    st.markdown(
        f"""
        <section class="hero">
          <div class="hero-copy-block">
            <p class="eyebrow">{_escape(_t("app_eyebrow"))}</p>
            <h1>BuyerLab AI Demo</h1>
            <p class="tagline">{_escape(_t("tagline"))}</p>
            {_support_line(_t("tagline_support"))}
            <p class="hero-copy">{_escape(_t("hero_copy"))}</p>
            <p class="hero-disclaimer">{_escape(_t("hero_disclaimer"))}</p>
          </div>
          <div class="hero-badges">
            <span class="status-pill">{_escape(_t("badge_audit"))}</span>
            <span class="status-pill">{_escape(_t("badge_readiness"))}</span>
            <span class="status-pill">{_escape(_t("badge_price"))}</span>
            {mock_badge}
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _support_line(text: str) -> str:
    """Render optional short English support copy without crowding the header."""
    if not text:
        return ""
    return f"<p class='tagline-support'>{_escape(text)}</p>"


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
    st.caption(_t("flow_caption"))
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
            _t("tab_report"),
            _t("tab_buyers_simple"),
            _t("tab_friction"),
            _t("tab_fix_simple"),
        ]
    )

    with tabs[0]:
        _render_launch_readiness(before_state.final_report, comparison)
        with st.expander(_t("audit_details"), expanded=False):
            _render_market_context(before_state)
            st.divider()
            _render_category_audit(before_state)
    with tabs[1]:
        _render_persona_cards(before_state.first_round_responses, before_state.final_report)
        _render_buyer_loss_analysis(buyer_loss_analysis)
        with st.expander(_t("agent_discussion_details"), expanded=False):
            _render_debate_terminal(before_state)
    with tabs[2]:
        _render_attention_map(attention_map)
    with tabs[3]:
        _render_optimization(suggestion)
        st.divider()
        _render_before_after(comparison, after_state)


def _render_launch_readiness(
    final_report: SimulationReport | None,
    comparison: dict[str, Any],
) -> None:
    """Render launch readiness summary that is readable in under 10 seconds."""
    if final_report is None:
        st.info(_t("judge_missing"))
        return

    decision = final_report.launch_decision_summary or final_report.executive_verdict or final_report.summary
    first_fix = (final_report.required_fix_before_launch or final_report.next_best_actions or [""])[0]
    _render_executive_report(final_report, comparison, decision, first_fix)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"#### {_t('required_fixes')}")
        _render_list(final_report.required_fix_before_launch or final_report.top_conversion_blockers)
    with col2:
        st.markdown(f"#### {_t('next_actions')}")
        _render_list(final_report.next_best_actions or final_report.top_action_items)

    risk_cols = st.columns(4)
    risk_cols[0].metric("Güven Açığı" if _language_code() == "tr" else "Trust Gap", final_report.trust_risk_score)
    risk_cols[1].metric("Fiyat Gerekçesi Açığı" if _language_code() == "tr" else "Price Justification Gap", final_report.price_resistance_score)
    risk_cols[2].metric("Netlik" if _language_code() == "tr" else "Clarity", final_report.clarity_score)
    risk_cols[3].metric("İade Riski" if _language_code() == "tr" else "Return Risk", final_report.return_risk_score)

    with st.expander(_t("why_verdict"), expanded=False):
        _audit_panel(_t("report_meaning"), final_report.brief_quality_summary or final_report.summary)
        if final_report.verdict_reasoning:
            st.markdown(f"#### {_t('why_verdict')}")
            _render_list(final_report.verdict_reasoning)
        st.markdown(f"#### {_t('decision_sources')}")
        _render_pills(_decision_sources(final_report))

        missing_col, weakness_col = st.columns(2)
        with missing_col:
            st.markdown(f"#### {_t('missing_info_context')}")
            _render_list(
                final_report.missing_information_not_product_failure
                or final_report.missing_brief_fields
            )
        with weakness_col:
            st.markdown(f"#### {_t('page_weaknesses')}")
            _render_list(final_report.product_page_weaknesses or final_report.top_conversion_blockers)
        if final_report.buyer_loss_summary:
            _audit_panel(_t("buyer_loss_summary"), final_report.buyer_loss_summary)
        st.markdown(f"#### {_t('seller_questions')}")
        _render_list(final_report.seller_questions)

    st.caption(_t("score_caption"))


def _render_executive_report(
    final_report: SimulationReport,
    comparison: dict[str, Any],
    decision: str,
    first_fix: str,
) -> None:
    """Render a one-screen executive report for hackathon demos."""
    st.markdown(f"### {_t('executive_report')}")
    _render_one_minute_summary(final_report, decision, first_fix)
    metric_cols = st.columns(5)
    metric_cols[0].metric(_t("launch_status"), _localized_label(final_report.launch_status))
    metric_cols[1].metric(_t("launch_score"), final_report.launch_readiness_score)
    metric_cols[2].metric(_t("simulated_score"), comparison["before_score"])
    metric_cols[3].metric(_t("analysis_confidence"), f"{final_report.analysis_confidence_score}/100")
    metric_cols[4].metric(_t("expected_lift"), _expected_lift_value(comparison))

    summary_col, fix_col = st.columns([1.15, 0.85])
    with summary_col:
        _audit_panel(_t("launch_decision_summary"), decision)
        _audit_panel(_t("main_blocker"), final_report.main_blocker or final_report.summary)
    with fix_col:
        _audit_panel(_t("expected_lift"), _expected_lift_text(comparison))
        st.markdown(f"#### {_t('first_three_fixes')}")
        _render_list((final_report.required_fix_before_launch or [first_fix])[:3])


def _render_one_minute_summary(
    final_report: SimulationReport,
    decision: str,
    first_fix: str,
) -> None:
    """Show the product verdict in three plain-language blocks."""
    reason = final_report.main_blocker or final_report.summary
    action = first_fix or (final_report.next_best_actions or [""])[0]
    cols = st.columns(3)
    with cols[0]:
        _summary_card(
            _t("one_minute_decision"),
            _localized_label(final_report.launch_status),
            decision,
            final_report.launch_status,
        )
    with cols[1]:
        _summary_card(_t("one_minute_reason"), _short_metric(reason, limit=42), reason)
    with cols[2]:
        _summary_card(_t("one_minute_action"), _short_metric(action, limit=42), action)


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
                        f"<p>{_escape(_report_text(response.main_reason))}</p>"
                        f"<small><b>{_escape(_t('top_objection'))}:</b> {_escape(_report_text(objection))}</small>"
                        f"<small><b>{_escape(_t('suggested_fix'))}:</b> {_escape(_report_text(response.suggested_fix))}</small>"
                        f"<small><b>{_escape(_t('business_impact'))}:</b> {_escape(_localized_label(business_impact))}</small>"
                    ),
                )
            else:
                _html_card(
                    title=_persona_name(persona.name),
                    body=(
                        "<span class='badge pending'>pending</span>"
                        f"<p>{_escape(_report_text(persona.decision_style))}</p>"
                        f"<small>{_escape(_t('waiting'))}</small>"
                    ),
                )


def _render_buyer_loss_analysis(buyer_loss_analysis: list[dict[str, Any]]) -> None:
    """Render buyer loss analysis rows."""
    st.markdown(f"### {_t('buyer_loss')}")
    if not buyer_loss_analysis:
        st.info(_t("buyer_loss"))
        return

    cols = st.columns(2)
    for index, row in enumerate(buyer_loss_analysis[:4]):
        persona = _persona_name(row.get("persona_name", row.get("persona_id", "")))
        decision = _localized_label(row.get("final_decision", ""))
        intent = row.get("purchase_intent", 0)
        reason = _report_text(row.get("main_loss_reason", ""))
        fix = _report_text(row.get("suggested_fix", ""))
        impact = _localized_label(row.get("business_impact", ""))
        with cols[index % 2]:
            _audit_panel(
                f"{persona} - {decision}",
                (
                    f"Satın alma niyeti: {intent}/100. "
                    f"Ana neden: {reason or _t('not_provided')} "
                    f"Öneri: {fix or _t('not_provided')} "
                    f"İş etkisi: {impact or _t('not_provided')}."
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
            f"{_escape(_report_text(turn.message))}"
            "</div>"
        )
    st.markdown(f"<div class='terminal'>{''.join(lines)}</div>", unsafe_allow_html=True)


def _render_market_context(state: SimulationState) -> None:
    """Render local price perception without requiring competitor context."""
    product = state.product
    final_report = state.final_report
    price_report = analyze_local_price_perception(product)

    st.markdown(f"### {_t('market_price_title')}")
    st.caption(_t("market_note"))
    cols = st.columns(4)
    cols[0].metric(_t("local_market"), price_report.local_market)
    cols[1].metric(_t("price_band"), _localized_label(price_report.price_band))
    cols[2].metric(_t("value_risk"), price_report.perceived_value_risk)
    cols[3].metric(_t("currency"), price_report.currency)

    col1, col2 = st.columns([1, 1])
    with col1:
        _audit_panel(
            _t("price_justification_verdict"),
            (
                final_report.price_justification_verdict
                if final_report and final_report.price_justification_verdict
                else price_report.pricing_comment
            ),
        )
        _audit_panel(_t("price_positioning"), price_report.suggested_price_positioning)
    with col2:
        st.markdown(f"#### {_t('required_value_proofs')}")
        _render_pills(price_report.required_value_proofs or [_t("no_objection")])
        st.markdown(f"#### {_t('expected_questions')}")
        _render_list(price_report.expected_customer_questions)


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
                _t("required_field"): _report_text(row.get("field_name", row.get("field", ""))),
                _t("status"): _localized_label(row.get("status", "")),
                _t("impact"): _localized_label(row.get("impact", "")),
                _t("explanation"): _report_text(row.get("explanation", row.get("note", ""))),
                _t("suggested_fix"): _report_text(row.get("suggested_fix", "")),
            }
        )

    st.dataframe(rows, width="stretch", hide_index=True)
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
                _t("reason"): _report_text(score.reason),
                _t("suggested_fix"): _report_text(score.suggested_fix),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)

    cols = st.columns(3)
    cols[0].metric(_t("strongest"), _section_name(attention_map.strongest_section))
    cols[1].metric(_t("weakest"), _section_name(attention_map.weakest_section))
    cols[2].metric(_t("highest_friction"), _section_name(attention_map.highest_friction_section))
    st.info(_report_text(attention_map.summary))


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
            st.markdown(f"#### {_t('trust_proof_checklist')}")
            _render_list(trust_checklist)

    missing_checklist = getattr(suggestion, "missing_information_checklist", [])
    if missing_checklist:
        st.markdown(f"#### {_t('missing_information_checklist')}")
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

    st.info(_report_text(comparison["summary"]))
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
    intended_use_case = st.session_state["intended_use_case"].strip()

    return ProductInput(
        brand=st.session_state["brand"].strip(),
        model=st.session_state["model"].strip(),
        product_type=(
            st.session_state["product_type"].strip()
            or st.session_state["product_title"].strip()
            or _category_label(category)
        ),
        title=st.session_state["product_title"].strip(),
        category=category,
        normalized_category=category,
        market_segment=st.session_state["market_segment"],
        intended_use_case=intended_use_case or st.session_state["value_proposition"].strip(),
        local_market=st.session_state["local_market"].strip() or "Türkiye",
        price=float(st.session_state["product_price"]),
        currency=currency if currency != "UNKNOWN" else "TRY",
        description=st.session_state["product_description"].strip(),
        target_audience=intended_use_case or st.session_state["value_proposition"].strip(),
        value_proposition=st.session_state["value_proposition"].strip(),
        warranty_or_return_policy=st.session_state["warranty_or_return_policy"].strip(),
        shipping_info=st.session_state["shipping_info"].strip(),
        trust_signals=_parse_list(st.session_state["trust_signals"], limit=8),
        reviews_or_social_proof=st.session_state["reviews_or_social_proof"].strip(),
        call_to_action=st.session_state["call_to_action"].strip(),
        image_notes=st.session_state["image_notes"].strip() or None,
        competitor_context=None,
        proof_assets=_parse_list(st.session_state["proof_assets"], limit=8),
        known_limitations=_parse_list(st.session_state["known_limitations"], limit=8),
    )


def _load_product_into_state(sample: dict[str, Any]) -> None:
    """Load sample product values into sidebar session state."""
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
    st.session_state["our_differentiator"] = ""
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
    if st.session_state.get("app_state_version") != APP_STATE_VERSION:
        st.session_state.pop("results", None)
        st.session_state.pop("ai_test_status", None)
        st.session_state["last_error"] = ""
        st.session_state["app_state_version"] = APP_STATE_VERSION

    defaults = {
        "language": "Türkçe",
        "selected_sample": "",
        "input_mode": "quick",
        "brand": "",
        "model": "",
        "product_type": "",
        "product_category": "general_product",
        "market_segment": "mid_range",
        "intended_use_case": "",
        "product_price": 0.0,
        "product_currency": "TRY",
        "local_market": "Türkiye",
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
    st.sidebar.markdown(f"### {_t('ai_status')}")
    status = check_gemini_connection()
    status_message = status.get("message", "")
    mode = status.get("mode", "live")
    model = status.get("model", "")

    if mode == "mock":
        st.sidebar.success(_t("mock_on"))
    elif status.get("ok"):
        st.sidebar.success(_t("key_detected"))
    else:
        st.sidebar.warning(status_message or _t("missing_key"))

    st.sidebar.caption(f"{_t('ai_mode')}: {mode} · {_t('ai_model')}: {model}")

    if st.sidebar.button(_t("ai_test"), width="stretch"):
        try:
            result = generate_json(
                'Return only valid JSON: {"ok": true, "message": "BuyerLab AI connection is ready"}'
            )
            st.session_state["ai_test_status"] = {
                "ok": True,
                "message": result.get("message", _t("ai_test_passed")),
            }
        except Exception as exc:
            st.session_state["ai_test_status"] = {
                "ok": False,
                "message": f"{_t('ai_test_failed')} {str(exc).splitlines()[0]}",
            }

    ai_test_status = st.session_state.get("ai_test_status")
    if ai_test_status:
        if ai_test_status.get("ok"):
            st.sidebar.success(ai_test_status.get("message", _t("ai_test_passed")))
        else:
            st.sidebar.error(ai_test_status.get("message", _t("ai_test_failed")))


def _parse_list(raw_value: str, limit: int = 8) -> list[str]:
    """Parse a short list from newline or comma separated textarea input."""
    values: list[str] = []
    for line in str(raw_value or "").replace(",", "\n").splitlines():
        value = line.strip()
        if value and value not in values:
            values.append(value)
    return values[:limit]


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
    text = _report_text(value) or _t("not_provided")
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _expected_lift_value(comparison: dict[str, Any]) -> str:
    """Return compact before-after simulated score lift."""
    delta = int(comparison.get("score_delta", 0) or 0)
    return f"+{delta}" if delta > 0 else str(delta)


def _expected_lift_text(comparison: dict[str, Any]) -> str:
    """Explain the simulated before-after score delta in seller language."""
    before = int(comparison.get("before_score", 0) or 0)
    after = int(comparison.get("after_score", 0) or 0)
    delta = int(comparison.get("score_delta", after - before) or 0)
    if _language_code() == "tr":
        if delta > 0:
            return (
                f"Düzeltme paketi simülasyonda {before}/100 seviyesinden {after}/100 "
                f"seviyesine çıkarak +{delta} puan iyileşme gösteriyor."
            )
        if delta == 0:
            return "Düzeltme paketi simülasyonda skoru değiştirmedi; önce kalan riskleri azaltmak gerekir."
        return f"Düzeltme paketi simülasyonda {abs(delta)} puan düşüş gösterdi; öneriler yeniden kontrol edilmeli."

    if delta > 0:
        return f"The fix pack raises the simulated score from {before}/100 to {after}/100, a +{delta} point lift."
    if delta == 0:
        return "The fix pack did not change the simulated score; reduce remaining risks before retesting."
    return f"The fix pack reduced the simulated score by {abs(delta)} points; review the recommendations."


def _render_list(values: list[str] | tuple[str, ...]) -> None:
    """Render concise dashboard list items."""
    if not values:
        st.caption(_t("no_items"))
        return
    for item in values[:6]:
        st.markdown(f"- {_report_text(item)}")


def _render_pills(values: list[str]) -> None:
    """Render compact labels."""
    safe_values = values or [_t("not_provided")]
    html = "".join(
        f"<span class='pill'>{_escape(_report_text(_display_label(value)))}</span>" for value in safe_values
    )
    st.markdown(f"<div class='pill-row'>{html}</div>", unsafe_allow_html=True)


def _decision_sources(final_report: SimulationReport) -> list[str]:
    """Return compact source labels for the launch readiness verdict."""
    if _language_code() == "tr":
        sources = [
            "Ürün brief'i",
            "AI müşteri profilleri",
            "Kategori beklentileri",
            "Yerel TL fiyat algısı",
        ]
        if final_report.analysis_confidence_label:
            sources.append(f"Analiz güveni: {final_report.analysis_confidence_label}")
        return sources

    sources = [
        "Product brief",
        "AI buyer personas",
        "Category expectations",
        "Local price perception",
    ]
    if final_report.analysis_confidence_label:
        sources.append(f"Analysis confidence: {final_report.analysis_confidence_label}")
    return sources


def _audit_panel(title: str, body: str) -> None:
    """Render a concise text panel."""
    st.markdown(
        f"""
        <div class="audit-panel">
          <h4>{_escape(title)}</h4>
          <p>{_escape(_report_text(body) or _t("not_provided"))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _summary_card(
    label: str,
    value: str,
    body: str,
    status: str = "",
) -> None:
    """Render a large plain-language summary card for 60-second demos."""
    status_class = f" {status}" if status else ""
    st.markdown(
        f"""
        <div class="summary-card{_escape(status_class)}">
          <span>{_escape(label)}</span>
          <strong>{_escape(_report_text(value) or _t("not_provided"))}</strong>
          <p>{_escape(_report_text(body) or _t("not_provided"))}</p>
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
          .block-container {
            padding-top: 2rem;
          }
          [data-testid="stSidebar"] {
            background: #0f172a;
            border-right: 1px solid rgba(148, 163, 184, 0.18);
          }
          [data-testid="stSidebar"] h3 {
            margin-top: 18px;
            color: #f8fafc;
          }
          #MainMenu,
          [data-testid="stToolbar"],
          [data-testid="stDecoration"],
          [data-testid="stStatusWidget"],
          .stDeployButton {
            display: none;
            height: 0;
            visibility: hidden;
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
          .tagline-support {
            color: #7dd3fc;
            font-size: 14px;
            margin: -4px 0 12px;
          }
          .hero-copy {
            color: #94a3b8;
            max-width: 840px;
            margin: 0;
          }
          .hero-disclaimer {
            color: #cbd5e1;
            font-size: 13px;
            font-weight: 700;
            margin: 12px 0 0;
          }
          .hero-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            justify-content: flex-end;
            max-width: 360px;
          }
          .guide-strip {
            background: rgba(56, 189, 248, 0.08);
            border: 1px solid rgba(56, 189, 248, 0.18);
            border-radius: 8px;
            display: grid;
            gap: 10px;
            grid-template-columns: repeat(3, 1fr);
            margin: 18px 0;
            padding: 12px;
          }
          .guide-strip span {
            color: #dbeafe;
            font-size: 13px;
            font-weight: 800;
            text-align: center;
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
          .empty-panel, .audit-panel, .card, .summary-card {
            background: #111827;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 8px;
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.18);
            overflow-wrap: anywhere;
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
          .summary-card {
            border-color: rgba(56, 189, 248, 0.22);
            margin-bottom: 18px;
            min-height: 176px;
            padding: 18px;
          }
          .summary-card.ready {
            border-color: rgba(34, 197, 94, 0.42);
          }
          .summary-card.needs_fixes {
            border-color: rgba(245, 158, 11, 0.42);
          }
          .summary-card.not_ready {
            border-color: rgba(239, 68, 68, 0.42);
          }
          .summary-card span {
            color: #38bdf8;
            display: block;
            font-size: 12px;
            font-weight: 900;
            margin-bottom: 10px;
            text-transform: uppercase;
          }
          .summary-card strong {
            color: #f8fafc;
            display: block;
            font-size: 24px;
            line-height: 1.15;
            margin-bottom: 12px;
          }
          .summary-card p {
            color: #cbd5e1;
            font-size: 14px;
            line-height: 1.42;
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
            white-space: nowrap;
          }
          div[data-testid="stTabs"] [role="tablist"] {
            overflow-x: auto;
          }
          h3, h4 {
            color: #f8fafc;
          }
          @media (max-width: 900px) {
            .block-container {
              padding-left: 0.9rem;
              padding-right: 0.9rem;
              padding-top: 1rem;
            }
            .hero {
              display: block;
              margin-bottom: 16px;
              padding: 18px;
            }
            .hero h1 {
              font-size: 32px;
            }
            .tagline {
              font-size: 16px;
            }
            .hero-copy {
              font-size: 13px;
            }
            .hero-badges {
              justify-content: flex-start;
              margin-top: 18px;
              max-width: none;
            }
            .guide-strip {
              grid-template-columns: 1fr;
            }
            .summary-card,
            .audit-panel,
            .card {
              min-height: 0;
              padding: 14px;
            }
            .summary-card strong {
              font-size: 20px;
            }
            .card p {
              min-height: 0;
            }
            div[data-testid="stHorizontalBlock"] {
              gap: 0.75rem;
            }
            div[data-testid="stMetric"] {
              margin-bottom: 8px;
            }
            div[data-testid="stDataFrame"] {
              overflow-x: auto;
            }
          }
          @media (max-width: 520px) {
            .hero h1 {
              font-size: 28px;
            }
            .status-pill,
            .pill {
              font-size: 11px;
            }
            .summary-card span {
              font-size: 11px;
            }
            .summary-card p,
            .audit-panel p,
            .card p {
              font-size: 13px;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
