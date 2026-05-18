#!/usr/bin/env bash
set -euo pipefail

export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8501}"

python -m streamlit run app.py \
  --server.address "$HOST" \
  --server.port "$PORT" \
  --server.headless true \
  --browser.gatherUsageStats false
