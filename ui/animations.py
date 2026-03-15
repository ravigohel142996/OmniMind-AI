"""
OmniMind AI — Animated Visualisations
Plotly Express animated charts for scenario and market evolution visuals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import CHART_HEIGHT, CHART_TEMPLATE


def animated_market_evolution(sim_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Animated bubble chart showing company growth and risk evolution
    across simulation rounds.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Long-format simulation output (company_name, round, revenue,
        growth_score, risk_score).
    top_n : int
        Number of top companies (by final revenue) to include.
    """
    # Keep only top_n companies by final revenue
    final = sim_df[sim_df["round"] == sim_df["round"].max()]
    top_companies = (
        final.nlargest(top_n, "revenue")["company_name"].tolist()
    )
    filtered = sim_df[sim_df["company_name"].isin(top_companies)].copy()

    # Clamp sizes
    filtered["bubble_size"] = np.clip(
        np.log1p(filtered["revenue"]) / np.log1p(filtered["revenue"].max()) * 50 + 5,
        5,
        55,
    )

    fig = px.scatter(
        filtered,
        x="growth_score",
        y="risk_score",
        animation_frame="round",
        animation_group="company_name",
        size="bubble_size",
        color="company_name",
        hover_name="company_name",
        template=CHART_TEMPLATE,
        title="Market Evolution — Growth vs Risk (Animated)",
        labels={"growth_score": "Growth Score", "risk_score": "Risk Score"},
        range_x=[0, 1],
        range_y=[0, 1],
        height=CHART_HEIGHT + 80,
    )
    fig.update_layout(showlegend=False)
    return fig


def scenario_comparison_bar(scenarios: list) -> go.Figure:
    """
    Grouped bar chart comparing scenarios across growth, risk and ROI.
    """
    names = [s.name for s in scenarios]
    growth_deltas = [s.growth_delta for s in scenarios]
    risk_deltas = [s.risk_delta for s in scenarios]
    roi_estimates = [s.roi_estimate for s in scenarios]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Growth Δ", x=names, y=growth_deltas, marker_color="#4F8BF9")
    )
    fig.add_trace(
        go.Bar(name="Risk Δ", x=names, y=risk_deltas, marker_color="#F97B4F")
    )
    fig.add_trace(
        go.Bar(name="ROI Estimate", x=names, y=roi_estimates, marker_color="#2ECC71")
    )
    fig.update_layout(
        barmode="group",
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT,
        title=dict(text="Scenario Comparison — Growth, Risk & ROI", x=0.5),
        yaxis_title="Delta / Multiplier",
        xaxis_tickangle=-30,
    )
    return fig
