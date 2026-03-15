"""
OmniMind AI — Growth Agent
Evaluates a company from a growth-maximisation perspective and proposes
actionable strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data.market_environment import MarketEnvironment


@dataclass
class AgentRecommendation:
    """Container for a single agent's recommendation."""

    agent_name: str
    strategy: str
    confidence: float   # 0–1
    rationale: str
    score: float        # normalised agent score (0–1)


class GrowthAgent:
    """
    Maximises long-term expansion opportunities.

    The agent scores a company on its growth potential and recommends
    the most appropriate expansion strategy.
    """

    name: str = "Growth Agent"

    def evaluate(
        self,
        company: pd.Series,
        market: MarketEnvironment,
        growth_score: float,
        risk_score: float,
    ) -> AgentRecommendation:
        """Evaluate the company and return a growth-focused recommendation."""
        # Composite growth signal
        signal = (
            growth_score * 0.45
            + company.get("innovation_index", 0.5) * 0.20
            + market.opportunity_index * 0.20
            + company.get("market_growth", 0.08) / 0.30 * 0.15
        )
        signal = float(min(max(signal, 0), 1))

        if signal >= 0.70:
            strategy = "Aggressive Expansion"
            rationale = (
                "High growth score and favourable market opportunity support "
                "accelerated expansion into new markets."
            )
        elif signal >= 0.50:
            strategy = "Steady Growth"
            rationale = (
                "Moderate growth potential — invest incrementally while "
                "monitoring competitive dynamics."
            )
        elif signal >= 0.35:
            strategy = "Market Consolidation"
            rationale = (
                "Growth is constrained; focus on deepening existing market share "
                "before pursuing expansion."
            )
        else:
            strategy = "Hold & Observe"
            rationale = (
                "Weak growth signals suggest delaying expansion until market "
                "conditions improve."
            )

        confidence = float(min(max(signal * 0.9 + 0.05, 0), 1))

        return AgentRecommendation(
            agent_name=self.name,
            strategy=strategy,
            confidence=round(confidence, 4),
            rationale=rationale,
            score=round(signal, 4),
        )
