"""
OmniMind AI — Sidebar Controls
Renders the Streamlit sidebar and returns user-selected parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from config import (
    MONTE_CARLO_DEFAULT_RUNS,
    MONTE_CARLO_MAX_RUNS,
    MONTE_CARLO_MIN_RUNS,
)


@dataclass
class ControlState:
    """All values read from the sidebar controls."""

    company_size: str          # "Small", "Medium", "Large", "Enterprise"
    market_growth: float       # slider 0–0.30
    competition_level: float   # slider 0–1
    technology_disruption: float  # slider 0–1
    economic_pressure: float   # slider 0–1
    n_simulations: int         # slider 100–1000
    generate_clicked: bool


def render_sidebar() -> ControlState:
    """Render the sidebar and return the current control state."""
    defaults = {
        "company_size": "All",
        "market_growth": 0.08,
        "competition_level": 0.50,
        "technology_disruption": 0.50,
        "economic_pressure": 0.40,
        "n_simulations": MONTE_CARLO_DEFAULT_RUNS,
    }

    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/artificial-intelligence.png",
            width=64,
        )
        st.title("⚙️ Control Panel")
        st.caption("Tune the environment, then generate strategy recommendations.")

        if st.button("↺ Reset to defaults", use_container_width=True):
            for state_key, state_val in defaults.items():
                st.session_state[state_key] = state_val
            st.rerun()

        st.markdown("---")

        st.subheader("🏢 Company Filters")
        company_size = st.selectbox(
            "Company Size",
            options=["All", "Small (<500)", "Medium (500–5k)", "Large (5k–50k)", "Enterprise (>50k)"],
            index=0,
            key="company_size",
            help="Filter analysis to a company-size band.",
        )

        st.markdown("---")
        st.subheader("📈 Market Parameters")

        market_growth = st.slider(
            "Market Growth Rate",
            min_value=0.0,
            max_value=0.30,
            value=0.08,
            step=0.01,
            format="%.2f",
            key="market_growth",
            help="Annual market growth rate (0 = flat, 0.30 = 30% growth)",
        )

        competition_level = st.slider(
            "Competition Level",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05,
            key="competition_level",
            help="Intensity of market competition (0 = none, 1 = extreme)",
        )

        technology_disruption = st.slider(
            "Technology Disruption",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05,
            key="technology_disruption",
            help="Speed of technological change in the market",
        )

        economic_pressure = st.slider(
            "Economic Pressure",
            min_value=0.0,
            max_value=1.0,
            value=0.40,
            step=0.05,
            key="economic_pressure",
            help="Macro-economic headwinds (0 = benign, 1 = severe recession)",
        )

        st.markdown("---")
        st.subheader("🎲 Simulation Settings")

        n_simulations = st.slider(
            "Monte Carlo Runs",
            min_value=MONTE_CARLO_MIN_RUNS,
            max_value=MONTE_CARLO_MAX_RUNS,
            value=MONTE_CARLO_DEFAULT_RUNS,
            step=50,
            key="n_simulations",
            help="Number of stochastic simulation runs (more = more accurate, slower)",
        )

        st.markdown("---")
        generate_clicked = st.button(
            "🚀 Generate AI Strategy",
            use_container_width=True,
            type="primary",
        )

        st.markdown("---")
        st.caption("OmniMind AI v1.0 · Autonomous Decision Intelligence")

    return ControlState(
        company_size=company_size,
        market_growth=market_growth,
        competition_level=competition_level,
        technology_disruption=technology_disruption,
        economic_pressure=economic_pressure,
        n_simulations=n_simulations,
        generate_clicked=generate_clicked,
    )
