"""Streamlit entrypoint for BuyerLab AI."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.graph import run_sample_simulation


DATA_PATH = Path(__file__).parent / "data" / "sample_products.json"


def load_sample_products() -> list[dict]:
    """Load sample products for the first demo screen."""
    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    st.set_page_config(page_title="BuyerLab AI", layout="wide")
    st.title("BuyerLab AI")
    st.caption("Virtual buyer simulation for pre-launch e-commerce testing.")

    products = load_sample_products()
    product_names = [product["name"] for product in products]
    selected_name = st.selectbox("Sample product", product_names)
    selected_product = next(product for product in products if product["name"] == selected_name)

    st.subheader("Product")
    st.json(selected_product)

    if st.button("Run placeholder simulation"):
        result = run_sample_simulation(selected_product)
        st.subheader("Simulation result")
        st.json(result)


if __name__ == "__main__":
    main()
