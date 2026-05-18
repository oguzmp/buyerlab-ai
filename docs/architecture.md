# BuyerLab AI Architecture

BuyerLab AI is a Streamlit-based pre-launch product-page testing system. It uses Gemini API and structured JSON outputs to simulate how different buyer personas may react to an e-commerce product page before launch.

The system is designed as a diagnostic tool. It does not claim to predict real market performance, and its attention map is not real eye-tracking.

## System Overview

The application accepts a `ProductInput`, runs multiple AI buyer personas, summarizes their reactions through a Judge Agent, scores page sections for AI-simulated attention and conversion friction, and generates optimization suggestions for a second simulated run.

High-level flow:

```text
ProductInput
  -> Buyer persona agents
  -> First-round AgentResponse list
  -> DebateTurn history
  -> Judge SimulationReport
  -> AttentionMapReport
  -> OptimizedProductSuggestion
  -> Re-simulation and before-after comparison
```

## Module Responsibilities

- `app.py`: Streamlit dashboard, product input form, result rendering, and workflow orchestration for the demo experience.
- `src/state.py`: Dataclass domain models including `ProductInput`, `BuyerPersona`, `AgentResponse`, `DebateTurn`, `SimulationReport`, `PageSectionScore`, `AttentionMapReport`, and `SimulationState`.
- `src/prompts.py`: Prompt templates for buyer personas and the Judge Agent.
- `src/gemini_client.py`: Gemini API wrapper, mock mode, JSON extraction, and error handling.
- `src/agents.py`: Persona evaluation and concise debate generation.
- `src/graph.py`: Core simulation orchestration through `run_simulation(product)`.
- `src/judge.py`: Buyer loss analysis, objection clustering, risk scoring, prioritized action items, and final Judge report creation.
- `src/attention_map.py`: AI-simulated buyer attention and conversion friction scoring for product page sections.
- `src/optimizer.py`: Product page optimization suggestions, optimized `ProductInput` generation, and before-after comparison.
- `data/sample_products.json`: Demo product examples.

## Agent Roles

BuyerLab AI currently uses four default buyer personas:

- Skeptic Buyer: Looks for technical details, proof, clear claims, return policy, and warranty information.
- Bargain Hunter: Evaluates price, value for money, shipping cost, discounts, and whether the offer feels justified.
- Impulsive Buyer: Reacts to emotional appeal, urgency, visual attractiveness, FOMO, and excitement.
- Trust Seeker: Looks for seller credibility, reviews, social proof, guarantees, professional language, and trust signals.

The Judge Agent reads persona decisions and debate history, then produces a dashboard-ready report with a simulated conversion score, loss reasons, risk scores, and action items.

## Simulation State

`SimulationState` is the main object passed through the simulation flow. It contains:

- `product`: Product or service page being tested.
- `personas`: Buyer personas used in the simulation.
- `first_round_responses`: One `AgentResponse` per persona.
- `debate_history`: Concise persona reactions to the group signal.
- `final_report`: Judge output as `SimulationReport`.
- `attention_map`: AI-simulated attention and friction report.
- `optimized_product_copy`: Reserved field for optimized copy text.
- `before_score` and `after_score`: Simulated conversion scores for comparison.

## Data Flow

1. The seller enters product details in the Streamlit dashboard.
2. `run_simulation(product)` creates an empty state with default personas.
3. `src/agents.py` runs each persona prompt through `generate_json()`.
4. Each JSON response is converted into an `AgentResponse`.
5. A short debate is generated as `DebateTurn` entries.
6. `src/judge.py` builds enhanced judge context from responses and debate history.
7. The Judge Agent returns or falls back to a `SimulationReport`.
8. `src/attention_map.py` scores required product page sections.
9. `src/optimizer.py` generates practical copy improvements.
10. The optimized product is simulated again for before-after comparison.

## Product Page Sections

The attention and friction scoring layer evaluates:

- `title`
- `price`
- `hero_image`
- `description`
- `value_proposition`
- `warranty_or_return_policy`
- `shipping_info`
- `trust_signals`
- `reviews_or_social_proof`
- `call_to_action`

Scores are AI-simulated buyer attention and conversion friction signals. They are not real heatmaps, real gaze data, or real analytics.

## Reliability Choices

- All Gemini calls are expected to return structured JSON.
- `src/gemini_client.py` extracts JSON from normal text or markdown code blocks.
- Missing API keys produce clear errors unless mock mode is enabled.
- Persona failures do not crash the full simulation; the system creates a hesitant fallback response.
- Judge failures produce a deterministic fallback report.
- Attention map and optimizer failures produce safe fallback outputs.
- Dashboard wording consistently uses “simulated conversion score.”

## Mock Mode

Mock mode is controlled by:

```text
BUYERLAB_MOCK_MODE=true
```

When enabled, Gemini calls return deterministic mock JSON instead of calling the API. This is useful for:

- Hackathon demos without internet or API setup
- Local UI testing
- Reproducible screenshots
- Avoiding accidental API usage

Live Gemini responses require:

```text
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```
