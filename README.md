# BuyerLab AI

Test your product with AI buyers before launch.

BuyerLab AI is a pre-launch e-commerce testing dashboard that simulates multiple AI buyer personas reviewing a product page before it goes live.

BuyerLab AI uses a Turkish-first interface because it is prepared for a Turkish hackathon, while keeping key AI/SaaS terms in English for technical clarity.

## Problem

E-commerce sellers often find out that a product page is weak only after spending money on ads, running a failed launch, or receiving avoidable customer returns. Missing product details, weak trust signals, unclear value, or confusing shipping and return policies can quietly reduce buyer confidence.

## Solution

BuyerLab AI runs a structured AI buyer simulation before launch. It evaluates a product page through different buyer personas and produces:

- Simulated conversion score
- Buyer loss reasons
- Conversion friction map
- AI-simulated attention map
- Optimization action plan
- Before-after comparison

## Honesty Note

BuyerLab AI does not claim to statistically predict real market performance. It provides an AI-simulated buyer reaction score to help sellers identify product-page weaknesses before launch.

The attention map is also AI-simulated buyer attention and conversion friction analysis. It is not real eye-tracking or real analytics data.

## Key Features

- Multi-agent buyer simulation
- Skeptic Buyer
- Bargain Hunter
- Impulsive Buyer
- Trust Seeker
- Judge Agent
- Buyer loss analysis
- Conversion friction map
- AI-simulated attention map
- Before-after optimization
- Streamlit dashboard
- Mock mode for demo and testing

## Tech Stack

- Python
- Streamlit
- Gemini API via `google-genai`
- Python dataclasses for typed domain models
- JSON-based structured outputs
- Lightweight Python orchestration for the multi-agent simulation flow

## How It Works

1. Product input
2. Buyer personas evaluate the product page
3. Personas produce a concise debate
4. Judge Agent creates a business report
5. Attention and friction map scores product page sections
6. Optimizer generates improved product copy
7. Product is re-simulated for a before-after comparison

Flow:

```text
Product input -> Buyer personas -> Debate -> Judge report -> Attention/friction map -> Optimization -> Re-simulation
```

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd buyerlab-ai
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local environment file:

```bash
copy .env.example .env
```

Add your Gemini API key to `.env`:

```text
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

For demo or testing without an API key, enable mock mode:

```text
BUYERLAB_MOCK_MODE=true
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## cPanel Deployment

A cPanel upload package can be prepared as `dist/buyerlab-ai-cpanel.zip`.

BuyerLab AI is a Streamlit app, so cPanel must support a long-running Python process or reverse proxy setup. Classic static/PHP-only cPanel hosting cannot run Streamlit directly. See `deployment/cpanel/README.md` for the exact upload notes.

## Environment Variables

- `GEMINI_API_KEY`: Gemini API key used for live AI responses.
- `BUYERLAB_MOCK_MODE`: Set to `true` for deterministic demo output without calling Gemini.
- `GEMINI_MODEL`: Optional Gemini model name. Defaults to `gemini-2.5-flash`.

## Project Structure

- `app.py`: Streamlit dashboard.
- `src/`: Core simulation, agents, judge, attention map, optimizer, Gemini client, and typed models.
- `data/`: Sample product inputs.
- `docs/`: Architecture notes and demo script.
- `tests/`: Initial test scaffolding.

## Demo Scenario

Recommended primary demo:

- Wireless Earbuds (Demo)

This demo works well because it shows a brand/model-level product audit, heuristic local TRY price perception, a competitor price gap, missing technical proof, trust risk, and before-after improvement.

## Hackathon Submission Notes

- Main AI provider: Gemini API
- App type: Streamlit web app
- Core idea: AI buyer simulation for pre-launch e-commerce testing
- Main differentiator: multi-agent buyer debate + simulated conversion score + friction map + before-after optimization

## Limitations

- Uses simulated buyer reactions, not real analytics.
- Does not provide real eye-tracking.
- Does not replace real A/B testing.
- Intended as a pre-launch diagnostic tool for product-page weaknesses.

## Roadmap

- Image understanding with Gemini Vision
- More buyer personas
- Category-specific persona weighting
- Exportable PDF reports
- Team collaboration
- Real marketplace integrations
