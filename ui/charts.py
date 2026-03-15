"""
OmniMind AI — Plotly Chart Components
All chart-building functions return a plotly Figure object.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import CHART_HEIGHT, CHART_TEMPLATE, RANDOM_SEED


# ── Radar chart ────────────────────────────────────────────────────────────────

def radar_chart(radar_data: dict[str, float], company_name: str) -> go.Figure:
    """Render a radar (spider) chart for a single company."""
    categories = list(radar_data.keys())
    values = list(radar_data.values())
    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(
        go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(79, 139, 249, 0.25)",
            line=dict(color="#4F8BF9", width=2),
            name=company_name,
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10))
        ),
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
        title=dict(text=f"Company Intelligence Radar — {company_name}", x=0.5),
    )
    return fig


# ── Revenue trend ──────────────────────────────────────────────────────────────

def revenue_trend_chart(sim_df: pd.DataFrame, company_name: str) -> go.Figure:
    """Line chart of simulated revenue over rounds for a single company."""
    cdf = sim_df[sim_df["company_name"] == company_name].copy()
    fig = px.line(
        cdf,
        x="round",
        y="revenue",
        markers=True,
        template=CHART_TEMPLATE,
        title=f"Simulated Revenue Trend — {company_name}",
        labels={"round": "Simulation Round", "revenue": "Revenue ($)"},
        height=CHART_HEIGHT,
    )
    fig.update_traces(line=dict(color="#4F8BF9", width=2.5))
    return fig


# ── Monte Carlo distribution ───────────────────────────────────────────────────

def monte_carlo_distribution(samples: np.ndarray, label: str) -> go.Figure:
    """Histogram of Monte Carlo samples with a KDE overlay."""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=samples,
            nbinsx=50,
            name=label,
            marker_color="#4F8BF9",
            opacity=0.75,
            histnorm="probability density",
        )
    )
    # KDE via Plotly's violin
    fig.add_trace(
        go.Violin(
            x=samples,
            name="KDE",
            line_color="#F97B4F",
            fillcolor="rgba(249,123,79,0.10)",
            visible=True,
            showlegend=False,
            side="positive",
            width=1.5,
        )
    )
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT,
        title=dict(text=f"Monte Carlo Distribution — {label}", x=0.5),
        xaxis_title=label,
        yaxis_title="Density",
        bargap=0.02,
    )
    return fig


# ── Strategy outcomes bar chart ────────────────────────────────────────────────

def strategy_outcomes_chart(outcomes: dict[str, float]) -> go.Figure:
    """Horizontal bar chart of Monte Carlo strategy win probabilities."""
    strategies = list(outcomes.keys())
    probs = [outcomes[s] * 100 for s in strategies]
    colours = [
        "#4F8BF9" if p == max(probs) else "#6C757D" for p in probs
    ]

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=strategies,
            orientation="h",
            marker_color=colours,
            text=[f"{p:.1f}%" for p in probs],
            textposition="outside",
        )
    )
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT,
        title=dict(text="Strategy Success Probabilities (Monte Carlo)", x=0.5),
        xaxis_title="Win Probability (%)",
        xaxis_range=[0, 110],
        margin=dict(l=160, r=60, t=60, b=40),
    )
    return fig


# ── Industry bubble chart ──────────────────────────────────────────────────────

def growth_vs_competition_bubble(df: pd.DataFrame) -> go.Figure:
    """Bubble chart: growth score vs competition level, sized by revenue."""
    fig = px.scatter(
        df,
        x="competition_level",
        y="growth_score",
        size="revenue",
        color="industry",
        hover_name="company_name",
        template=CHART_TEMPLATE,
        title="Growth vs Competition (bubble = revenue)",
        labels={
            "competition_level": "Competition Level",
            "growth_score": "Growth Score",
        },
        height=CHART_HEIGHT,
        size_max=45,
    )
    return fig


# ── Innovation vs revenue ──────────────────────────────────────────────────────

def innovation_vs_revenue_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter plot of innovation index vs revenue coloured by industry."""
    fig = px.scatter(
        df,
        x="innovation_index",
        y="revenue",
        color="industry",
        size="r_and_d_spend",
        hover_name="company_name",
        template=CHART_TEMPLATE,
        title="Innovation vs Revenue",
        labels={
            "innovation_index": "Innovation Index",
            "revenue": "Revenue ($)",
        },
        log_y=True,
        height=CHART_HEIGHT,
        size_max=35,
    )
    return fig


# ── Market opportunity heatmap ─────────────────────────────────────────────────

def market_opportunity_heatmap(pivot: pd.DataFrame) -> go.Figure:
    """Heatmap of average growth score by industry × technology adoption tier."""
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="Blues",
            text=pivot.values.round(3),
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(title="Avg Growth Score"),
        )
    )
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT,
        title=dict(text="Market Opportunity Heatmap (Growth by Industry × Tech Adoption)", x=0.5),
        xaxis_title="Technology Adoption Tier",
        yaxis_title="Industry",
        margin=dict(l=140, r=40, t=60, b=60),
    )
    return fig


# ── Risk heatmap ───────────────────────────────────────────────────────────────

def risk_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of risk probability across industries and competition bands."""
    out = df.copy()
    out["competition_band"] = pd.cut(
        out["competition_level"],
        bins=[0, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    pivot = out.pivot_table(
        values="risk_probability",
        index="industry",
        columns="competition_band",
        aggfunc="mean",
    ).fillna(0)

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="Reds",
            text=pivot.values.round(3),
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(title="Avg Risk Probability"),
        )
    )
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT,
        title=dict(text="Risk Heatmap (by Industry × Competition Level)", x=0.5),
        xaxis_title="Competition Band",
        yaxis_title="Industry",
        margin=dict(l=140, r=40, t=60, b=60),
    )
    return fig


# ── Agent score breakdown bar chart ───────────────────────────────────────────

def agent_score_chart(agent_results: list) -> go.Figure:
    """Grouped bar chart of agent names vs their scores and confidences."""
    names = [r.agent_name for r in agent_results]
    scores = [r.score for r in agent_results]
    confidences = [r.confidence for r in agent_results]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Agent Score", x=names, y=scores, marker_color="#4F8BF9")
    )
    fig.add_trace(
        go.Bar(name="Confidence", x=names, y=confidences, marker_color="#F97B4F")
    )
    fig.update_layout(
        barmode="group",
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT,
        title=dict(text="AI Agent Scores & Confidence", x=0.5),
        yaxis_range=[0, 1.1],
        yaxis_title="Score / Confidence",
    )
    return fig


# ── Monte Carlo timeline ───────────────────────────────────────────────────────

def mc_growth_cone(mc_result) -> go.Figure:
    """
    Fan / cone chart showing the percentile range of simulated growth over
    the simulation horizon.
    """
    rounds = list(range(1, 13))
    rng = np.random.default_rng(RANDOM_SEED)
    # Build a simple fan from the MC summary
    mean_path = np.linspace(mc_result.growth_mean * 0.8, mc_result.growth_mean, 12)
    p5_path = np.linspace(mc_result.growth_p5 * 0.7, mc_result.growth_p5, 12)
    p95_path = np.linspace(mc_result.growth_p95 * 0.85, mc_result.growth_p95, 12)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rounds + rounds[::-1],
            y=list(p95_path) + list(p5_path[::-1]),
            fill="toself",
            fillcolor="rgba(79, 139, 249, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="90% Confidence Band",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rounds,
            y=mean_path,
            line=dict(color="#4F8BF9", width=2.5),
            name="Mean Growth Path",
        )
    )
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT,
        title=dict(text="Monte Carlo Growth Forecast (12-Round Horizon)", x=0.5),
        xaxis_title="Simulation Round",
        yaxis_title="Growth Score",
        yaxis_range=[0, 1],
    )
    return fig
