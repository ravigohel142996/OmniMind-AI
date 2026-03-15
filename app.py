"""
OmniMind AI — Main Application Entry Point
Run with: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="OmniMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd

from config import PLATFORM_NAME, PLATFORM_SUBTITLE, PLATFORM_VERSION
from data.company_generator import generate_companies
from data.market_environment import MarketEnvironment
from models.growth_model import GrowthModel
from models.risk_model import RiskModel
from models.hiring_model import HiringModel
from simulation.market_simulator import MarketSimulator
from simulation.monte_carlo_engine import MonteCarloEngine
from simulation.future_scenarios import generate_scenarios
from analytics.company_analysis import CompanyAnalyser
from agents.strategy_consensus import StrategyConsensus
from ui.controls import render_sidebar
from ui import dashboard


# ── Custom CSS ─────────────────────────────────────────────────────────────────

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* Platform header */
        .omnimind-header {
            background: linear-gradient(135deg, #0e1117 0%, #1a2744 100%);
            border-bottom: 2px solid #4F8BF9;
            padding: 1.2rem 2rem;
            margin-bottom: 1.5rem;
            border-radius: 8px;
        }
        .omnimind-header h1 {
            color: #4F8BF9;
            font-size: 2.2rem;
            margin: 0;
            font-weight: 700;
            letter-spacing: 1px;
        }
        .omnimind-header p {
            color: #8899aa;
            margin: 0.3rem 0 0 0;
            font-size: 1rem;
        }
        /* Metric cards */
        [data-testid="metric-container"] {
            background-color: #1a2030;
            border: 1px solid #2a3a50;
            border-radius: 10px;
            padding: 0.8rem;
        }
        /* Section dividers */
        hr {
            border-color: #2a3a50;
        }
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #0d1117;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Caching helpers ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Generating company dataset…")
def _load_companies() -> pd.DataFrame:
    return generate_companies()


@st.cache_resource(show_spinner="Training ML models…")
def _train_models(df_json: str) -> tuple[GrowthModel, RiskModel, HiringModel]:
    df = pd.read_json(df_json, orient="records")

    growth_model = GrowthModel()
    risk_model = RiskModel()

    # Inject default economic_pressure for training
    df_risk = df.copy()
    df_risk["economic_pressure"] = 0.40

    growth_model.train(df)
    risk_model.train(df_risk)

    # Hiring model needs growth_score — use growth predictions
    df_hiring = df.copy()
    df_hiring["growth_score"] = growth_model.predict(df)

    hiring_model = HiringModel()
    hiring_model.train(df_hiring)

    return growth_model, risk_model, hiring_model


@st.cache_data(show_spinner="Running market simulation…")
def _run_market_simulation(
    df_json: str,
    economic_pressure: float,
    technology_disruption: float,
    market_growth: float,
) -> str:
    df = pd.read_json(df_json, orient="records")
    market = MarketEnvironment(
        economic_pressure=economic_pressure,
        technology_disruption=technology_disruption,
        market_growth_rate=market_growth,
    )
    simulator = MarketSimulator(n_rounds=12)
    sim_df = simulator.simulate(df, market)
    return sim_df.to_json(orient="records")


# ── Filter helpers ─────────────────────────────────────────────────────────────

def _filter_by_size(df: pd.DataFrame, size_label: str) -> pd.DataFrame:
    if size_label == "All":
        return df
    bands = {
        "Small (<500)": (0, 499),
        "Medium (500–5k)": (500, 4_999),
        "Large (5k–50k)": (5_000, 49_999),
        "Enterprise (>50k)": (50_000, 10_000_000),
    }
    lo, hi = bands[size_label]
    return df[(df["employees"] >= lo) & (df["employees"] <= hi)]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    _inject_css()

    # ── Header ──────────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="omnimind-header">
            <h1>🧠 {PLATFORM_NAME}</h1>
            <p>{PLATFORM_SUBTITLE} &nbsp;·&nbsp; v{PLATFORM_VERSION}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar controls ─────────────────────────────────────────────────────────
    controls = render_sidebar()

    # ── Data loading & model training ────────────────────────────────────────────
    companies_df = _load_companies()
    df_json = companies_df.to_json(orient="records")
    growth_model, risk_model, hiring_model = _train_models(df_json)

    # ── Market environment ────────────────────────────────────────────────────────
    market = MarketEnvironment(
        economic_pressure=controls.economic_pressure,
        technology_disruption=controls.technology_disruption,
        market_growth_rate=controls.market_growth,
        global_demand=max(0.0, min(1.0, 1 - controls.economic_pressure * 0.5)),
    )

    # ── Enrich company data with model scores ─────────────────────────────────────
    analyser = CompanyAnalyser(growth_model, risk_model, hiring_model)
    enriched_df = analyser.enrich(companies_df, market)

    # Apply size filter
    filtered_df = _filter_by_size(enriched_df, controls.company_size)
    if filtered_df.empty:
        st.warning("No companies match the selected size filter. Showing all.")
        filtered_df = enriched_df

    # ── Market simulation ─────────────────────────────────────────────────────────
    sim_json = _run_market_simulation(
        df_json,
        controls.economic_pressure,
        controls.technology_disruption,
        controls.market_growth,
    )
    sim_df = pd.read_json(sim_json, orient="records")

    # ── Navigation tabs ───────────────────────────────────────────────────────────
    tab_market, tab_company, tab_strategy, tab_simulation, tab_intelligence = st.tabs(
        [
            "🌐 Market Overview",
            "🏢 Company Intelligence",
            "🤖 Strategy Advisor",
            "🔮 Future Simulation",
            "📊 Market Intelligence",
        ]
    )

    # ── Tab 1: Market Overview ────────────────────────────────────────────────────
    with tab_market:
        dashboard.render_market_overview(market)

    # ── Company selector (shared across tabs 2, 3, 4) ────────────────────────────
    company_names = sorted(filtered_df["company_name"].tolist())
    selected_company = st.sidebar.selectbox(
        "🏢 Select Company",
        options=company_names,
        index=0,
    )

    profile = analyser.get_profile(filtered_df, selected_company)
    radar_data = analyser.radar_data(profile)

    # ── Tab 2: Company Intelligence ────────────────────────────────────────────────
    with tab_company:
        dashboard.render_company_intelligence(
            profile, radar_data, sim_df, filtered_df
        )

    # ── Strategy & simulation (computed on demand or always) ─────────────────────
    company_row = filtered_df[
        filtered_df["company_name"] == selected_company
    ].iloc[0]

    mc_engine = MonteCarloEngine(n_runs=controls.n_simulations)
    mc_result = mc_engine.run(
        company_row=company_row,
        market=market,
        base_growth=profile.growth_score,
        base_risk=profile.risk_probability,
    )

    consensus_engine = StrategyConsensus()
    consensus = consensus_engine.evaluate(
        company=company_row,
        market=market,
        growth_score=profile.growth_score,
        risk_score=profile.risk_probability,
        mc_roi_mean=mc_result.roi_mean,
    )

    scenarios = generate_scenarios(
        base_growth=profile.growth_score,
        base_risk=profile.risk_probability,
        market=market,
    )

    # ── Tab 3: Strategy Advisor ────────────────────────────────────────────────────
    with tab_strategy:
        dashboard.render_strategy_advisor(consensus, consensus.agent_breakdown)

    # ── Tab 4: Future Simulation ───────────────────────────────────────────────────
    with tab_simulation:
        dashboard.render_future_simulation(mc_result, scenarios, sim_df)

    # ── Tab 5: Market Intelligence ─────────────────────────────────────────────────
    with tab_intelligence:
        dashboard.render_market_intelligence(filtered_df, sim_df)


if __name__ == "__main__":
    main()
