"""LangGraph wiring for the BuyerLab AI simulation flow."""

from __future__ import annotations

from typing import Callable

from src.agents import simulate_buyer_response
from src.judge import judge_buyer_response
from src.state import Product, SimulationState, create_initial_state


def build_graph() -> Callable[[SimulationState], SimulationState]:
    """Build the simulation graph, with a lightweight fallback for local tests."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        return _run_linear_simulation

    graph = StateGraph(SimulationState)
    graph.add_node("buyer", simulate_buyer_response)
    graph.add_node("judge", judge_buyer_response)
    graph.add_edge(START, "buyer")
    graph.add_edge("buyer", "judge")
    graph.add_edge("judge", END)
    return graph.compile().invoke


def _run_linear_simulation(state: SimulationState) -> SimulationState:
    """Run the placeholder flow without LangGraph installed."""
    state = simulate_buyer_response(state)
    return judge_buyer_response(state)


def run_sample_simulation(product: Product) -> SimulationState:
    """Run one sample buyer simulation for a product."""
    state = create_initial_state(product)
    graph = build_graph()
    return graph(state)
