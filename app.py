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
        :root {
            --om-bg: #070b14;
            --om-bg-soft: #101a2b;
            --om-surface: #111e33;
            --om-card: #17233a;
            --om-border: #2b3f5f;
            --om-text: #e9f1ff;
            --om-text-muted: #9fb2d0;
            --om-primary: #4f8bf9;
            --om-accent: #22d3ee;
            --om-success: #34d399;
        }

        .stApp {
            background: radial-gradient(circle at top right, #10213f 0%, var(--om-bg) 45%);
            color: var(--om-text);
        }

        /* Platform header */
        .omnimind-header {
            background: linear-gradient(125deg, #0f1d34 0%, #1d3257 50%, #0f1d34 100%);
            border: 1px solid var(--om-border);
            box-shadow: 0 10px 30px rgba(4, 10, 22, 0.45);
            padding: 1.2rem 2rem;
            margin-bottom: 1.25rem;
            border-radius: 14px;
        }
        .omnimind-header h1 {
            color: #dce9ff;
            font-size: 2.1rem;
            margin: 0;
            font-weight: 800;
            letter-spacing: .6px;
        }
        .omnimind-header p {
            color: var(--om-text-muted);
            margin: 0.35rem 0 0 0;
            font-size: 0.98rem;
        }

        /* Metric cards */
        [data-testid="metric-container"] {
            background: linear-gradient(180deg, #1a2a46 0%, #152238 100%);
            border: 1px solid var(--om-border);
            border-radius: 12px;
            padding: 0.85rem;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, .02);
        }

        /* Section dividers */
        hr {
            border-color: #20314f;
        }

        /* Sidebar shell */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #080e1a 0%, #0b1322 100%);
            border-right: 1px solid #16243f;
        }

        [data-testid="stSidebar"] * {
            color: var(--om-text);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] .stCaption {
            color: var(--om-text-muted);
        }

        /* Inputs */
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #15243d;
            border: 1px solid #2c4469;
            border-radius: 10px;
        }

        .stSlider [data-baseweb="slider"] > div > div {
            background: linear-gradient(90deg, var(--om-primary), var(--om-accent));
        }

        .stSlider [role="slider"] {
            background-color: var(--om-primary);
            border-color: #c7d9ff;
            box-shadow: 0 0 0 4px rgba(79, 139, 249, 0.2);
        }

        /* Buttons */
        .stButton > button {
            border-radius: 10px;
            border: 1px solid #325184;
            background: #13213a;
            color: #eaf2ff;
            transition: all .2s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            border-color: #4f8bf9;
            box-shadow: 0 6px 18px rgba(79, 139, 249, 0.22);
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #2d67d1 0%, #3d84ff 100%);
            border-color: #77a8ff;
            color: white;
            font-weight: 700;
        }

        /* Tabs */
        [data-testid="stTabs"] [role="tab"] {
            background: #0f1a2d;
            border: 1px solid #233858;
            border-radius: 8px 8px 0 0;
            margin-right: 0.25rem;
            color: #c7d9fb;
        }

        [data-testid="stTabs"] [aria-selected="true"] {
            background: #193056;
            border-color: #4f8bf9;
            color: #ffffff;
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
    # global_demand is inversely proportional to economic_pressure:
    # strong pressure halves demand; the relationship is clamped to [0, 1].
    _DEMAND_PRESSURE_FACTOR: float = 0.5
    market = MarketEnvironment(
        economic_pressure=controls.economic_pressure,
        technology_disruption=controls.technology_disruption,
        market_growth_rate=controls.market_growth,
        global_demand=max(
            0.0, min(1.0, 1 - controls.economic_pressure * _DEMAND_PRESSURE_FACTOR)
        ),
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
