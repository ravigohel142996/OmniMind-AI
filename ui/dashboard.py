"""
OmniMind AI — Dashboard Sections
Renders each major section of the Streamlit dashboard.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from analytics.company_analysis import CompanyProfile
from agents.strategy_consensus import ConsensusResult
from simulation.monte_carlo_engine import MonteCarloResult
from simulation.future_scenarios import Scenario
from data.market_environment import MarketEnvironment
from utils.helpers import (
    format_currency,
    format_percentage,
    confidence_label,
    risk_label,
)
from ui import charts, animations


# ── Helper ─────────────────────────────────────────────────────────────────────

def _metric_card(col, label: str, value: str, delta: str | None = None) -> None:
    with col:
        if delta:
            st.metric(label, value, delta)
        else:
            st.metric(label, value)


# ── 1. Global Market Overview ──────────────────────────────────────────────────

def render_market_overview(market: MarketEnvironment) -> None:
    st.header("🌐 Global Market Overview")

    c1, c2, c3, c4 = st.columns(4)
    _metric_card(c1, "📈 Market Growth Rate",
                 format_percentage(market.market_growth_rate))
    _metric_card(c2, "💼 Economic Pressure",
                 format_percentage(market.economic_pressure),
                 delta=f"{'↑ High' if market.economic_pressure > 0.6 else '↓ Manageable'}")
    _metric_card(c3, "⚡ Tech Disruption",
                 format_percentage(market.technology_disruption))
    _metric_card(c4, "🌍 Global Demand",
                 format_percentage(market.global_demand))

    st.markdown("---")
    c5, c6 = st.columns(2)
    with c5:
        st.info(
            f"**Opportunity Index:** {market.opportunity_index:.3f}  \n"
            f"A composite score reflecting current market upside potential."
        )
    with c6:
        st.warning(
            f"**Threat Index:** {market.threat_index:.3f}  \n"
            f"A composite score reflecting macro headwinds and competitive pressure."
        )


# ── 2. Company Intelligence ────────────────────────────────────────────────────

def render_company_intelligence(
    profile: CompanyProfile,
    radar_data: dict[str, float],
    sim_df: pd.DataFrame,
    enriched_df: pd.DataFrame,
) -> None:
    st.header(f"🏢 Company Intelligence — {profile.name}")

    # Key metrics row
    c1, c2, c3, c4 = st.columns(4)
    _metric_card(c1, "💰 Revenue", format_currency(profile.revenue))
    _metric_card(c2, "👥 Employees", f"{profile.employees:,}")
    _metric_card(c3, "🏭 Industry", profile.industry)
    _metric_card(c4, "🔬 R&D Spend", format_percentage(profile.r_and_d_spend))

    st.markdown("---")

    # Score cards
    c5, c6, c7 = st.columns(3)
    _metric_card(c5, "🚀 Growth Score",
                 f"{profile.growth_score:.3f}",
                 delta=f"{profile.growth_score * 100:.1f}%")
    _metric_card(c6, "⚠️ Risk Probability",
                 risk_label(profile.risk_probability))
    _metric_card(c7, "🤝 Hiring Probability",
                 f"{profile.hiring_probability:.3f}")

    st.markdown("---")

    # Charts
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(
            charts.radar_chart(radar_data, profile.name),
            use_container_width=True,
        )
    with col_right:
        st.plotly_chart(
            charts.revenue_trend_chart(sim_df, profile.name),
            use_container_width=True,
        )

    st.plotly_chart(
        charts.risk_heatmap(enriched_df),
        use_container_width=True,
    )


# ── 3. Strategy Advisor ────────────────────────────────────────────────────────

def render_strategy_advisor(
    consensus: ConsensusResult,
    agent_results: list,
) -> None:
    st.header("🤖 AI Strategy Advisor")

    # Headline recommendation
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1e3a5f, #0e1117);
            border: 1px solid #4F8BF9;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
        ">
            <h2 style="color:#4F8BF9; margin:0;">
                🎯 {consensus.recommended_strategy}
            </h2>
            <p style="color:#aaa; margin:8px 0 0 0;">
                Recommended by OmniMind AI Consensus Engine
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    _metric_card(
        c1,
        "🎯 Confidence",
        confidence_label(consensus.confidence_score),
        delta=f"{consensus.confidence_score * 100:.1f}%",
    )
    _metric_card(
        c2,
        "📊 Expected ROI",
        format_percentage(consensus.expected_roi),
    )
    _metric_card(
        c3,
        "⚠️ Risk Probability",
        risk_label(consensus.risk_probability),
    )

    st.markdown("---")

    # Agent breakdown
    st.subheader("🧠 Agent Analysis Breakdown")
    st.plotly_chart(
        charts.agent_score_chart(consensus.agent_breakdown),
        use_container_width=True,
    )

    with st.expander("📋 Agent Rationales", expanded=False):
        for rec in consensus.agent_breakdown:
            st.markdown(f"**{rec.agent_name}** → *{rec.strategy}*")
            st.caption(rec.rationale)
            st.divider()

    # Strategy vote scores
    st.subheader("🗳️ Strategy Vote Scores")
    sorted_votes = sorted(
        consensus.strategy_vote_scores.items(), key=lambda x: x[1], reverse=True
    )
    for strategy, score in sorted_votes:
        icon = "⭐" if strategy == consensus.recommended_strategy else "  "
        st.progress(
            float(score),
            text=f"{icon} {strategy}: {score * 100:.1f}%",
        )


# ── 4. Future Simulation ───────────────────────────────────────────────────────

def render_future_simulation(
    mc_result: MonteCarloResult,
    scenarios: list[Scenario],
    sim_df: pd.DataFrame,
) -> None:
    st.header("🔮 Future Simulation")

    # Summary metrics
    c1, c2, c3 = st.columns(3)
    _metric_card(c1, "📈 Expected Growth (mean)", f"{mc_result.growth_mean:.3f}")
    _metric_card(c2, "⚠️ Expected Risk (mean)", f"{mc_result.risk_mean:.3f}")
    _metric_card(c3, "💰 Expected ROI (mean)", format_percentage(mc_result.roi_mean))

    c4, c5 = st.columns(2)
    with c4:
        st.info(
            f"**Growth 90% CI:** [{mc_result.growth_p5:.3f}, {mc_result.growth_p95:.3f}]"
        )
    with c5:
        st.info(
            f"**ROI 90% CI:** [{mc_result.roi_p5:.1%}, {mc_result.roi_p95:.1%}]"
        )

    st.markdown("---")

    # Charts
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Risk Distribution", "📈 Growth Distribution",
         "🎯 Strategy Outcomes", "🔮 Growth Forecast"]
    )

    with tab1:
        st.plotly_chart(
            charts.monte_carlo_distribution(mc_result.risk_samples, "Risk Probability"),
            use_container_width=True,
        )
    with tab2:
        st.plotly_chart(
            charts.monte_carlo_distribution(mc_result.growth_samples, "Growth Score"),
            use_container_width=True,
        )
    with tab3:
        st.plotly_chart(
            charts.strategy_outcomes_chart(mc_result.strategy_outcomes),
            use_container_width=True,
        )
    with tab4:
        st.plotly_chart(
            charts.mc_growth_cone(mc_result),
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("🌍 Strategic Scenario Analysis")
    st.plotly_chart(
        animations.scenario_comparison_bar(scenarios),
        use_container_width=True,
    )

    with st.expander("📋 Scenario Descriptions", expanded=False):
        for s in scenarios:
            direction = "📈" if s.growth_delta > 0 else "📉"
            st.markdown(
                f"**{direction} {s.name}**  \n"
                f"{s.description}  \n"
                f"Growth Δ: `{s.growth_delta:+.3f}` · "
                f"Risk Δ: `{s.risk_delta:+.3f}` · "
                f"ROI: `{s.roi_estimate:.2f}×`"
            )
            st.divider()


# ── 5. Market Intelligence ─────────────────────────────────────────────────────

def render_market_intelligence(
    enriched_df: pd.DataFrame,
    sim_df: pd.DataFrame,
) -> None:
    st.header("📊 Market Intelligence")

    from analytics.market_analysis import MarketAnalyser
    analyser = MarketAnalyser()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🫧 Growth vs Competition",
            "💡 Innovation vs Revenue",
            "🗺️ Opportunity Map",
            "🎬 Market Evolution",
        ]
    )

    with tab1:
        st.plotly_chart(
            charts.growth_vs_competition_bubble(
                analyser.growth_vs_competition(enriched_df)
            ),
            use_container_width=True,
        )

    with tab2:
        st.plotly_chart(
            charts.innovation_vs_revenue_scatter(
                analyser.innovation_vs_revenue(enriched_df)
            ),
            use_container_width=True,
        )

    with tab3:
        pivot = analyser.market_heatmap(enriched_df)
        st.plotly_chart(
            charts.market_opportunity_heatmap(pivot),
            use_container_width=True,
        )
        st.subheader("📋 Industry Opportunity Ranking")
        opp_df = analyser.opportunity_map(enriched_df)[
            ["industry", "avg_growth", "avg_risk", "avg_innovation",
             "avg_revenue", "opportunity_score", "company_count"]
        ].round(3)
        st.dataframe(opp_df, use_container_width=True)

    with tab4:
        st.plotly_chart(
            animations.animated_market_evolution(sim_df),
            use_container_width=True,
        )
