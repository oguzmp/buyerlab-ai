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
from src.state import (
    AgentResponse,
    AttentionMapReport,
    ProductInput,
    SimulationReport,
    SimulationState,
    get_default_personas,
)


DATA_PATH = Path(__file__).parent / "data" / "sample_products.json"


def load_sample_products() -> list[dict[str, Any]]:
    """Load sample products for quick demos."""
    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


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
    st.sidebar.markdown("## Product Input")
    _render_environment_notice()

    samples = load_sample_products()
    sample_names = [sample.get("name", sample.get("title", "Sample")) for sample in samples]
    selected_sample = st.sidebar.selectbox("Sample product", sample_names)

    if st.sidebar.button("Load sample product", use_container_width=True):
        sample = samples[sample_names.index(selected_sample)]
        _load_product_into_state(sample)
        st.session_state["last_error"] = ""
        st.rerun()

    st.sidebar.text_input("Product title", key="product_title")
    st.sidebar.text_input("Category", key="product_category")
    st.sidebar.number_input("Price", min_value=0.0, step=1.0, key="product_price")
    st.sidebar.text_input("Currency", key="product_currency")
    st.sidebar.text_input("Target audience", key="target_audience")
    st.sidebar.text_area("Value proposition", key="value_proposition", height=90)
    st.sidebar.text_area("Product description", key="product_description", height=120)
    st.sidebar.text_area("Warranty / return policy", key="warranty_or_return_policy", height=80)
    st.sidebar.text_area("Shipping info", key="shipping_info", height=80)
    st.sidebar.text_area(
        "Trust signals",
        key="trust_signals",
        height=90,
        help="Use one signal per line, such as secure checkout or real reviews.",
    )
    st.sidebar.text_area("Reviews or social proof", key="reviews_or_social_proof", height=80)
    st.sidebar.text_input("Call to action", key="call_to_action")
    st.sidebar.text_area("Image notes", key="image_notes", height=80)

    run_clicked = st.sidebar.button(
        "Run Simulation",
        type="primary",
        use_container_width=True,
    )

    if run_clicked:
        product = _product_from_inputs()
        if not product.title:
            st.session_state["last_error"] = "Add a product title before running a simulation."
            st.rerun()
        _run_dashboard_simulation(product)
        st.rerun()


def _render_header() -> None:
    """Render the dashboard header."""
    st.markdown(
        """
        <section class="hero">
          <div>
            <p class="eyebrow">Pre-launch conversion lab</p>
            <h1>BuyerLab AI</h1>
            <p class="tagline">Test your product with AI buyers before launch.</p>
            <p class="hero-copy">
              BuyerLab AI simulates different buyer personas to identify conversion
              blockers before a product page goes live.
            </p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_empty_state() -> None:
    """Render the pre-run dashboard preview."""
    st.markdown("### Simulation Flow")
    _render_persona_cards([])

    st.markdown(
        """
        <div class="empty-panel">
          <h3>Ready for first simulation</h3>
          <p>
            Add product details in the left panel, then run the AI buyer test.
            The dashboard will show buyer decisions, debate, conversion blockers,
            AI-simulated attention friction, and optimization suggestions.
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
    st.markdown("### Simulation Flow")
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
        st.metric("Simulated conversion score", before_score)
    with col2:
        st.metric("After optimization", after_score, delta=score_delta)
    with col3:
        st.metric("Trust risk", final_report.trust_risk_score if final_report else 0)
    with col4:
        st.metric("Price resistance", final_report.price_resistance_score if final_report else 0)


def _render_persona_cards(responses: list[AgentResponse]) -> None:
    """Render persona decision cards."""
    personas = get_default_personas()
    responses_by_persona = {response.persona_id: response for response in responses}
    cols = st.columns(4)

    for index, persona in enumerate(personas):
        response = responses_by_persona.get(persona.id)
        with cols[index]:
            if response:
                objection = response.objections[0] if response.objections else "No major objection."
                _html_card(
                    title=persona.name,
                    body=(
                        f"<span class='badge {response.decision}'>{response.decision}</span>"
                        f"<div class='intent'>{response.purchase_intent}/100 intent</div>"
                        f"<p>{_escape(response.main_reason)}</p>"
                        f"<small>Top objection: {_escape(objection)}</small>"
                    ),
                )
            else:
                _html_card(
                    title=persona.name,
                    body=(
                        "<span class='badge pending'>pending</span>"
                        "<div class='intent'>Run simulation</div>"
                        f"<p>{_escape(persona.decision_style)}</p>"
                        "<small>Waiting for buyer evaluation.</small>"
                    ),
                )


def _render_debate_terminal(state: SimulationState) -> None:
    """Render debate history as a compact market terminal."""
    st.markdown("### Live Debate / Market Terminal")
    if not state.debate_history:
        st.info("No debate history yet.")
        return

    lines = []
    for turn in state.debate_history:
        lines.append(
            "<div class='terminal-line'>"
            f"<span>{_escape(turn.speaker)}</span>"
            f"<em>{_escape(turn.stance)}</em>"
            f"{_escape(turn.message)}"
            "</div>"
        )
    st.markdown(f"<div class='terminal'>{''.join(lines)}</div>", unsafe_allow_html=True)


def _render_judge_report(final_report: SimulationReport | None) -> None:
    """Render the Judge report."""
    st.markdown("### Judge Report")
    if final_report is None:
        st.info("Judge report is not available yet.")
        return

    cols = st.columns(5)
    metrics = [
        ("Simulated conversion score", final_report.simulated_conversion_score),
        ("Trust risk", final_report.trust_risk_score),
        ("Price resistance", final_report.price_resistance_score),
        ("Clarity", final_report.clarity_score),
        ("Return risk", final_report.return_risk_score),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, value)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Winning personas**")
        _render_pills(final_report.winning_personas or ["None yet"])
    with col2:
        st.markdown("**Lost personas**")
        _render_pills(final_report.lost_personas or ["None yet"])

    st.markdown("**Top action items**")
    for item in final_report.top_action_items:
        st.markdown(f"- {item}")

    st.info(final_report.summary)
    st.caption("Scores are simulated buyer testing signals, not real market predictions.")


def _render_buyer_loss_analysis(buyer_loss_analysis: list[dict[str, Any]]) -> None:
    """Render buyer loss rows."""
    st.markdown("### Lost Buyer Analysis")
    if not buyer_loss_analysis:
        st.info("No buyer loss analysis available.")
        return

    rows = [
        {
            "Persona": row.get("persona_name", row.get("persona_id", "")),
            "Decision": row.get("final_decision", ""),
            "Intent": row.get("purchase_intent", 0),
            "Impact": row.get("business_impact", ""),
            "Main reason": row.get("main_loss_reason", ""),
            "Suggested fix": row.get("suggested_fix", ""),
        }
        for row in buyer_loss_analysis
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_attention_map(attention_map: AttentionMapReport) -> None:
    """Render AI-simulated attention and conversion friction map."""
    st.markdown("### Conversion Friction / Attention Map")
    st.caption(
        "AI-simulated buyer attention and conversion friction analysis. "
        "This is not real eye-tracking."
    )

    rows = []
    for score in attention_map.section_scores:
        rows.append(
            {
                "Section": score.section_name,
                "Attention": score.attention_score,
                "Friction": score.friction_score,
                "Priority": get_section_priority_label(score.attention_score, score.friction_score),
                "Sentiment": score.sentiment,
                "Reason": score.reason,
                "Suggested fix": score.suggested_fix,
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    cols = st.columns(3)
    cols[0].metric("Strongest section", attention_map.strongest_section)
    cols[1].metric("Weakest section", attention_map.weakest_section)
    cols[2].metric("Highest friction", attention_map.highest_friction_section)
    st.info(attention_map.summary)


def _render_optimization(suggestion: Any) -> None:
    """Render optimized product copy suggestions."""
    st.markdown("### Optimization Plan")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Optimized title**")
        st.write(suggestion.title)
        st.markdown("**Improved value proposition**")
        st.write(suggestion.value_proposition)
        st.markdown("**CTA suggestion**")
        st.write(suggestion.call_to_action)
    with col2:
        st.markdown("**Optimized description**")
        st.write(suggestion.description)
        st.markdown("**Improved trust signals**")
        _render_pills(suggestion.trust_signals)

    st.markdown("**FAQ suggestions**")
    for item in suggestion.faq_items:
        st.markdown(f"- {item}")

    st.markdown("**Change summary**")
    for item in suggestion.change_summary:
        st.markdown(f"- {item}")


def _render_before_after(comparison: dict[str, Any], after_state: SimulationState) -> None:
    """Render before-after simulation comparison."""
    st.markdown("### Before / After Comparison")
    col1, col2, col3 = st.columns(3)
    col1.metric("Before score", comparison["before_score"])
    col2.metric("After score", comparison["after_score"])
    col3.metric("Score delta", comparison["score_delta"])

    col4, col5 = st.columns(2)
    with col4:
        st.markdown("**Improved sections**")
        _render_pills(comparison["improved_sections"] or ["No section lift yet"])
    with col5:
        st.markdown("**Remaining risks**")
        for risk in comparison["remaining_risks"]:
            st.markdown(f"- {risk}")

    st.info(comparison["summary"])
    if after_state.final_report:
        st.caption(
            "After score is still a simulated conversion score, not a market prediction."
        )


def _run_dashboard_simulation(product: ProductInput) -> None:
    """Run the full before-after dashboard workflow with readable error handling."""
    st.session_state["last_error"] = ""
    try:
        with st.spinner("Running AI buyer simulation..."):
            before_state = run_simulation(product)
            before_report = before_state.final_report or SimulationReport(
                summary="Simulated conversion score unavailable."
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
            f"Simulation failed safely: {str(exc).splitlines()[0]}"
        )


def _product_from_inputs() -> ProductInput:
    """Create ProductInput from sidebar fields."""
    return ProductInput(
        title=st.session_state["product_title"].strip(),
        category=st.session_state["product_category"].strip(),
        price=float(st.session_state["product_price"]),
        currency=st.session_state["product_currency"].strip() or "USD",
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
        st.sidebar.success("Mock mode is enabled.")
    elif not has_api_key:
        st.sidebar.warning(
            "GEMINI_API_KEY is missing. Add it or set BUYERLAB_MOCK_MODE=true "
            "for demo-safe deterministic output."
        )
    else:
        st.sidebar.caption("Gemini API key detected.")


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
    html = "".join(f"<span class='pill'>{_escape(value)}</span>" for value in values)
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
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
