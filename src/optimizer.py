"""Before-after optimization helpers for BuyerLab AI simulations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from src.gemini_client import generate_json
from src.launch_readiness import build_category_expectation_check
from src.price_intelligence import analyze_competitor_gap, analyze_local_price_perception
from src.state import (
    AttentionMapReport,
    PAGE_SECTION_NAMES,
    ProductInput,
    SimulationReport,
    SimulationState,
)


@dataclass(slots=True)
class OptimizedProductSuggestion:
    """Dashboard-ready product page copy improvements for a second simulation."""

    title: str = ""
    description: str = ""
    value_proposition: str = ""
    warranty_or_return_policy: str = ""
    shipping_info: str = ""
    trust_signals: list[str] = field(default_factory=list)
    trust_proof_checklist: list[str] = field(default_factory=list)
    faq_items: list[str] = field(default_factory=list)
    competitor_comparison_suggestion: str = ""
    missing_information_checklist: list[str] = field(default_factory=list)
    call_to_action: str = ""
    change_summary: list[str] = field(default_factory=list)


def build_optimization_prompt(
    product: ProductInput,
    final_report: SimulationReport,
    attention_map: AttentionMapReport | None = None,
    buyer_loss_analysis: list[dict[str, Any]] | None = None,
) -> str:
    """Build a Gemini prompt for realistic product page optimization."""
    context = {
        "important_note": (
            "The current score is a simulated conversion score, not a real market "
            "prediction."
        ),
        "product": asdict(product),
        "final_report": asdict(final_report),
        "attention_map": asdict(attention_map) if attention_map else None,
        "buyer_loss_analysis": buyer_loss_analysis or [],
    }

    return f"""
You are optimizing a pre-launch e-commerce product page for BuyerLab AI.
Use the simulation report to propose realistic copy improvements that can be
tested in a second simulation.

Context:
{json.dumps(context, ensure_ascii=True, indent=2)}

Return only valid JSON with this exact shape:
{{
  "title": "",
  "description": "",
  "value_proposition": "",
  "warranty_or_return_policy": "",
  "shipping_info": "",
  "trust_signals": [],
  "trust_proof_checklist": [],
  "faq_items": [],
  "competitor_comparison_suggestion": "",
  "missing_information_checklist": [],
  "call_to_action": "",
  "change_summary": []
}}

Rules:
- Write all user-facing copy and checklist items in Turkish.
- Keep all text concise, practical, and seller-focused.
- Prioritize trust blockers first.
- Then address missing product information.
- Then improve price/value justification.
- Then improve emotional appeal and CTA.
- Do not exaggerate product claims.
- Do not invent fake reviews, fake certifications, fake awards, or fake guarantees.
- If trust signals are missing, suggest adding real trust signals the seller can verify.
- If proof is missing, say "add real proof" instead of fabricating facts.
- Include a missing_information_checklist with category-critical fields the seller
  must add before launch.
- Include competitor_comparison_suggestion only from seller-provided competitor data.
- Mention "simulated conversion score" only as a testing signal, not a prediction.
""".strip()


def generate_optimized_product(
    product: ProductInput,
    final_report: SimulationReport,
    attention_map: AttentionMapReport | None = None,
    buyer_loss_analysis: list[dict[str, Any]] | None = None,
) -> OptimizedProductSuggestion:
    """Generate optimized product copy, falling back safely on invalid Gemini output."""
    try:
        raw_suggestion = generate_json(
            build_optimization_prompt(
                product=product,
                final_report=final_report,
                attention_map=attention_map,
                buyer_loss_analysis=buyer_loss_analysis,
            )
        )
        return _suggestion_from_json(raw_suggestion, product, final_report, attention_map)
    except Exception:
        return _fallback_suggestion(product, final_report, attention_map)


def apply_optimization_to_product(
    product: ProductInput,
    suggestion: OptimizedProductSuggestion,
) -> ProductInput:
    """Apply optimized copy suggestions to a ProductInput for re-simulation."""
    return replace(
        product,
        title=suggestion.title or product.title,
        description=suggestion.description or product.description,
        value_proposition=suggestion.value_proposition or product.value_proposition,
        warranty_or_return_policy=(
            suggestion.warranty_or_return_policy
            or product.warranty_or_return_policy
        ),
        shipping_info=suggestion.shipping_info or product.shipping_info,
        trust_signals=suggestion.trust_signals or product.trust_signals,
        call_to_action=suggestion.call_to_action or product.call_to_action,
    )


def compare_before_after(
    before_state: SimulationState,
    after_state: SimulationState,
) -> dict[str, Any]:
    """Compare two simulation states using simulated conversion score signals."""
    before_score = _state_score(before_state)
    after_score = _state_score(after_state)
    score_delta = after_score - before_score

    return {
        "before_score": before_score,
        "after_score": after_score,
        "score_delta": score_delta,
        "improved_sections": _improved_sections(before_state, after_state),
        "remaining_risks": _remaining_risks(after_state),
        "summary": _comparison_summary(before_score, after_score, score_delta),
    }


def _suggestion_from_json(
    raw_suggestion: dict[str, Any],
    product: ProductInput,
    final_report: SimulationReport,
    attention_map: AttentionMapReport | None,
) -> OptimizedProductSuggestion:
    """Convert Gemini JSON into a safe OptimizedProductSuggestion."""
    fallback = _fallback_suggestion(product, final_report, attention_map)
    suggestion = OptimizedProductSuggestion(
        title=_short_text(raw_suggestion.get("title")) or product.title,
        description=_seller_text_or_fallback(
            raw_suggestion.get("description"),
            fallback.description,
            limit=320,
        ),
        value_proposition=_seller_text_or_fallback(
            raw_suggestion.get("value_proposition"),
            fallback.value_proposition,
            limit=220,
        ),
        warranty_or_return_policy=_seller_text_or_fallback(
            raw_suggestion.get("warranty_or_return_policy"),
            fallback.warranty_or_return_policy,
            limit=180,
        ),
        shipping_info=_seller_text_or_fallback(
            raw_suggestion.get("shipping_info"),
            fallback.shipping_info,
            limit=180,
        ),
        trust_signals=_seller_list_or_fallback(
            raw_suggestion.get("trust_signals"),
            fallback.trust_signals,
            limit=5,
        ),
        trust_proof_checklist=_seller_list_or_fallback(
            raw_suggestion.get("trust_proof_checklist"),
            fallback.trust_proof_checklist,
            limit=6,
        ),
        faq_items=_seller_list_or_fallback(
            raw_suggestion.get("faq_items"),
            fallback.faq_items,
            limit=5,
        ),
        competitor_comparison_suggestion=_short_text(
            raw_suggestion.get("competitor_comparison_suggestion"),
            limit=240,
        )
        or fallback.competitor_comparison_suggestion,
        missing_information_checklist=_seller_list_or_fallback(
            raw_suggestion.get("missing_information_checklist"),
            fallback.missing_information_checklist,
            limit=8,
        ),
        call_to_action=_seller_text_or_fallback(
            raw_suggestion.get("call_to_action"),
            fallback.call_to_action,
            limit=80,
        ),
        change_summary=_seller_list_or_fallback(
            raw_suggestion.get("change_summary"),
            fallback.change_summary,
            limit=5,
        ),
    )
    return suggestion


def _fallback_suggestion(
    product: ProductInput,
    final_report: SimulationReport,
    attention_map: AttentionMapReport | None,
) -> OptimizedProductSuggestion:
    """Create safe optimization suggestions when Gemini is unavailable or invalid."""
    return OptimizedProductSuggestion(
        title=_fallback_title(product),
        description=_fallback_description(product),
        value_proposition=_fallback_value_proposition(product),
        warranty_or_return_policy=_fallback_warranty(product),
        shipping_info=_fallback_shipping(product),
        trust_signals=_fallback_trust_signals(product),
        trust_proof_checklist=_fallback_trust_proof_checklist(product, final_report),
        faq_items=_fallback_faq_items(product, final_report),
        competitor_comparison_suggestion=_fallback_competitor_comparison(product),
        missing_information_checklist=_fallback_missing_information_checklist(product),
        call_to_action=_fallback_call_to_action(product),
        change_summary=_fallback_change_summary(final_report, attention_map),
    )


def _fallback_title(product: ProductInput) -> str:
    """Return a clearer title without inventing unsupported claims."""
    identity = " ".join(
        part
        for part in [product.brand, product.model, product.product_type]
        if part
    )
    if identity:
        return _short_text(identity)
    if product.title and product.category:
        return _short_text(f"{product.title} - {product.intended_use_case or product.category}")
    return product.title or "Yayın öncesi test edilecek ürün"


def _fallback_description(product: ProductInput) -> str:
    """Return concise description copy grounded in the original product input."""
    base = product.description or "Ürünün ne işe yaradığını ve kime hitap ettiğini net anlat."
    audience = product.intended_use_case or product.target_audience or "hedef müşteri"
    missing = _fallback_missing_information_checklist(product)
    missing_text = f" Şunlar için gerçek kanıt ekle: {', '.join(missing[:3])}." if missing else ""
    return _short_text(
        f"{base} Kullanım bağlamı: {audience}. Ana faydayı, somut kanıtı, fiyat gerekçesini ve sonraki adımı açık yaz.{missing_text}",
        limit=320,
    )


def _fallback_value_proposition(product: ProductInput) -> str:
    """Return a sharper value proposition using existing product facts."""
    price_report = analyze_local_price_perception(product)
    if product.value_proposition:
        return _short_text(
            f"{product.value_proposition} Bu fiyat bandını somut kanıt, risk azaltıcı bilgiler ve kategoriye özel detaylarla destekle.",
            limit=220,
        )
    return "Ana faydayı, neden önemli olduğunu ve fiyatı hangi gerçek kanıtların desteklediğini açıkla."


def _fallback_warranty(product: ProductInput) -> str:
    """Suggest real warranty or return policy content without fabricating terms."""
    if product.warranty_or_return_policy:
        return product.warranty_or_return_policy
    return "Gerçek iade süresini, iade koşullarını ve varsa garanti kapsamını ekle."


def _fallback_shipping(product: ProductInput) -> str:
    """Suggest realistic shipping information without inventing logistics."""
    if product.shipping_info:
        return product.shipping_info
    return "Gerçek kargo ücretini, teslimat süresini, hizmet verilen bölgeleri ve teslimat notlarını ekle."


def _fallback_trust_signals(product: ProductInput) -> list[str]:
    """Suggest verifiable trust signals instead of fabricated social proof."""
    signals = list(product.trust_signals)
    suggestions = [
        "Gerçek müşteri yorumu varsa ekle.",
        "Güvenli ödeme ve kabul edilen ödeme yöntemlerini göster.",
        "Gerçek iade, garanti ve destek politikalarına bağlantı ver.",
    ]
    for suggestion in suggestions:
        if suggestion not in signals:
            signals.append(suggestion)
    return signals[:5]


def _fallback_trust_proof_checklist(
    product: ProductInput,
    final_report: SimulationReport,
) -> list[str]:
    """Create a concrete trust proof checklist without inventing fake claims."""
    checklist = [
        "Net garanti süresi ve iade koşullarını ekle.",
        "Gerçek destek kanalı ve servis politikasını göster.",
        "Genel kalite iddiaları yerine gerçek kanıt varlıkları ekle.",
    ]
    if final_report.trust_risk_score >= 50:
        checklist.append("Güven kanıtlarını fiyat ve satın alma çağrısının yakınına yerleştir.")
    if not product.reviews_or_social_proof.strip():
        checklist.append("Yalnızca gerçekse doğrulanmış yorum veya önceki iş örneği ekle.")
    return _dedupe_short_list(checklist, limit=6)


def _fallback_faq_items(
    product: ProductInput,
    final_report: SimulationReport,
) -> list[str]:
    """Create FAQ suggestions from report risks and product gaps."""
    faq_items: list[str] = []
    if final_report.return_risk_score >= 40 or not product.warranty_or_return_policy:
        faq_items.append("Gerçek iade veya garanti koşulları nedir?")
    if final_report.price_resistance_score >= 40:
        faq_items.append("Bu ürün neden bu fiyatı hak ediyor?")
    if final_report.price_justification_verdict:
        faq_items.append("Fiyat konumlandırmasını hangi kanıt destekliyor?")
    if final_report.trust_risk_score >= 40:
        faq_items.append("Hangi gerçek kanıt veya müşteri geri bildirimi mevcut?")
    if not product.shipping_info:
        faq_items.append("Kargo ücreti ve teslimat süresi nedir?")
    if not faq_items:
        faq_items.append("Bu ürün en çok kimler için uygun?")
    return faq_items[:5]


def _fallback_competitor_comparison(product: ProductInput) -> str:
    """Suggest a competitor comparison from seller-provided context only."""
    competitor = product.competitor_context
    if competitor is None or not (
        competitor.competitor_name
        or competitor.competitor_price
        or competitor.our_differentiator
    ):
        return "Rakip bilgisi girilmedi; karşılaştırma yazmadan önce bir alternatif ürün bilgisi eklenebilir."

    gap = analyze_competitor_gap(product)
    name = competitor.competitor_name or "the competitor"
    if gap.price_gap and gap.price_gap > 0:
        return (
            f"{name} ile kısa bir karşılaştırma tablosu ekle: fiyat farkı, iddia edilen ayrışma, "
            "kanıtlar, garanti/iade koşulları ve yüksek fiyatın gerekçesi."
        )
    return (
        f"{name} ile kısa bir karşılaştırma tablosu ekle: fiyat, ayrışma, kanıtlar, "
        "destek koşulları ve müşteri riskini azaltan bilgiler."
    )


def _fallback_missing_information_checklist(product: ProductInput) -> list[str]:
    """Return category-critical missing information for the fix pack."""
    missing = [
        row["field_name"]
        for row in build_category_expectation_check(product)
        if row["status"] in {"missing", "weak"}
    ]
    missing.extend(product.known_limitations)
    return _dedupe_short_list(missing, limit=8)


def _fallback_call_to_action(product: ProductInput) -> str:
    """Return a clearer CTA grounded in the existing page."""
    if product.call_to_action:
        return product.call_to_action
    return "Ürünü incele"


def _fallback_change_summary(
    final_report: SimulationReport,
    attention_map: AttentionMapReport | None,
) -> list[str]:
    """Summarize optimization priorities by expected simulated business impact."""
    changes: list[str] = []
    if final_report.trust_risk_score >= 40:
        changes.append("Öncelik gerçek güven sinyallerine ve satın alma riskini azaltan bilgilere verildi.")
    if final_report.clarity_score < 70:
        changes.append("Ürün detayları ve değer önerisi daha net hale getirildi.")
    if final_report.price_resistance_score >= 40:
        changes.append("Fiyat ve değer gerekçesi güçlendirildi.")
    if final_report.required_fix_before_launch:
        changes.append("Düzeltme paketi yayından önce gereken kritik maddelere odaklandı.")
    if final_report.launch_decision_summary:
        changes.append(final_report.launch_decision_summary)
    if attention_map and attention_map.highest_friction_section:
        changes.append(f"En yüksek sürtünme görülen bölüm için düzeltme önerisi üretildi: {attention_map.highest_friction_section}.")
    if not changes:
        changes.append("Metin, daha net simüle dönüşüm skoru testi için sadeleştirildi.")
    return changes[:5]


def _state_score(state: SimulationState) -> int:
    """Read a state's simulated conversion score safely."""
    if state.final_report is not None:
        return _safe_score(state.final_report.simulated_conversion_score)
    if state.after_score is not None:
        return _safe_score(state.after_score)
    if state.before_score is not None:
        return _safe_score(state.before_score)
    return 0


def _improved_sections(
    before_state: SimulationState,
    after_state: SimulationState,
) -> list[str]:
    """Identify attention-map sections with improved simulated friction or attention."""
    if not before_state.attention_map or not after_state.attention_map:
        return []

    before_sections = {
        score.section_name: score for score in before_state.attention_map.section_scores
    }
    improved: list[str] = []
    for after_score in after_state.attention_map.section_scores:
        before_score = before_sections.get(after_score.section_name)
        if before_score is None:
            continue
        friction_improved = after_score.friction_score < before_score.friction_score
        attention_improved = after_score.attention_score > before_score.attention_score
        if friction_improved or attention_improved:
            improved.append(after_score.section_name)

    return [section for section in PAGE_SECTION_NAMES if section in improved]


def _remaining_risks(state: SimulationState) -> list[str]:
    """Summarize remaining dashboard-ready risks after optimization."""
    risks: list[str] = []
    if state.final_report:
        risks.extend(state.final_report.buyer_loss_reasons[:3])
        if state.final_report.trust_risk_score >= 50:
            risks.append("Güven riski hâlâ yüksek.")
        if state.final_report.price_resistance_score >= 50:
            risks.append("Fiyat direnci hâlâ yüksek.")
        if state.final_report.return_risk_score >= 50:
            risks.append("Kargo veya iade endişesi hâlâ yüksek.")

    if state.attention_map:
        risks.append(
            f"En yüksek simüle sürtünme hâlâ {state.attention_map.highest_friction_section} bölümünde."
        )

    return _dedupe_short_list(risks, limit=5) or ["Belirgin kalan risk tespit edilmedi."]


def _comparison_summary(before_score: int, after_score: int, score_delta: int) -> str:
    """Create a concise before-after summary using simulated score language."""
    if score_delta > 0:
        return (
            "Düzeltme paketi sonrası simüle dönüşüm skoru iyileşti; yayından önce gerçek sayfa içeriğiyle tekrar test et."
        )
    if score_delta < 0:
        return (
            "Düzeltme paketi sonrası simüle dönüşüm skoru düştü; bu metni kullanmadan önce önerileri gözden geçir."
        )
    return (
        "Simüle dönüşüm skoru değişmedi; yeni testten önce kalan müşteri risklerine odaklan."
    )


def _short_list(value: Any, limit: int = 5) -> list[str]:
    """Normalize JSON list fields into concise string lists."""
    if isinstance(value, list):
        return _dedupe_short_list([str(item) for item in value], limit=limit)
    if isinstance(value, str) and value.strip():
        return [_short_text(value)]
    return []


def _dedupe_short_list(values: list[str], limit: int = 5) -> list[str]:
    """Deduplicate dashboard text while preserving order."""
    items: list[str] = []
    for value in values:
        text = _short_text(value)
        if text and text not in items:
            items.append(text)
    return items[:limit]


def _safe_score(value: Any) -> int:
    """Normalize a score into an integer from 0 to 100."""
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, score))


def _short_text(value: Any, limit: int = 180) -> str:
    """Keep optimization text concise and dashboard-ready."""
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _seller_text_or_fallback(value: Any, fallback: str, limit: int = 180) -> str:
    """Use Gemini text only when it looks seller-ready and not English/template-like."""
    text = _short_text(value, limit=limit)
    if not text or _looks_like_english_template(text):
        return fallback
    return text


def _seller_list_or_fallback(value: Any, fallback: list[str], limit: int = 5) -> list[str]:
    """Use Gemini list items only when they are dashboard-ready for Turkish sellers."""
    items = _short_list(value, limit=limit)
    if not items or any(_looks_like_english_template(item) for item in items):
        return fallback[:limit]
    return items


def _looks_like_english_template(text: str) -> bool:
    """Detect common English/template output that should not be shown in the Turkish UI."""
    lowered = f" {text.lower()} "
    markers = [
        " add ",
        " built for ",
        " state ",
        " proof ",
        " warranty ",
        " shipping ",
        " customer ",
        " before launch ",
        " price ",
        " value ",
        " support ",
        " review ",
        " when available ",
        " positioned ",
        " checkout ",
        " delivery ",
    ]
    return any(marker in lowered for marker in markers)
