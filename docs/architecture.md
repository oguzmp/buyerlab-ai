# BuyerLab AI Architecture

BuyerLab AI starts as a small Streamlit app that runs a placeholder buyer simulation over sample product data.

## Initial Components

- `app.py`: Streamlit UI entrypoint.
- `src/state.py`: Typed simulation state shared across the graph.
- `src/prompts.py`: Prompt builders for buyer and judge steps.
- `src/agents.py`: Gemini client helper and buyer-response agent placeholder.
- `src/graph.py`: LangGraph flow with a local fallback for early tests.
- `src/judge.py`: Deterministic placeholder judge.
- `data/sample_products.json`: Demo product inputs.

## Planned Flow

1. A product is selected in Streamlit.
2. A buyer persona is attached to the simulation state.
3. The buyer agent reacts to the product.
4. The judge summarizes buyer intent and objections.
5. Results are shown in the app.

The Gemini API key must be supplied through `GEMINI_API_KEY`.
