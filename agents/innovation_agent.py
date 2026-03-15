"""
OmniMind AI — Innovation Agent
Evaluates a company from an R&D investment perspective.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data.market_environment import MarketEnvironment


@dataclass
class AgentRecommendation:
    agent_name: str
    strategy: str
    confidence: float
    rationale: str
    score: float


class InnovationAgent:
    """
    Drives R&D investment and technology adoption.
    """

    name: str = "Innovation Agent"

    def evaluate(
        self,
        company: pd.Series,
        market: MarketEnvironment,
        growth_score: float,
        risk_score: float,
    ) -> AgentRecommendation:
        """Evaluate the company and return an innovation-focused recommendation."""
        innovation = company.get("innovation_index", 0.5)
        r_and_d = company.get("r_and_d_spend", 0.05)
        tech_adoption = company.get("technology_adoption", 0.5)

        signal = (
            innovation * 0.35
            + tech_adoption * 0.30
            + r_and_d / 0.25 * 0.20   # normalise by max plausible r&d ratio
            + market.technology_disruption * 0.15
        )
        signal = float(min(max(signal, 0), 1))

        if signal >= 0.70:
            strategy = "Innovation Focus"
            rationale = (
                "Strong innovation posture — increase R&D budget to maintain "
                "technological leadership and capture disruption opportunities."
            )
        elif signal >= 0.50:
            strategy = "Steady Growth"
            rationale = (
                "Moderate innovation capability — sustain current R&D levels "
                "and selectively adopt emerging technologies."
            )
        elif signal >= 0.30:
            strategy = "Strategic Pivot"
            rationale = (
                "Innovation gap detected — consider partnering with tech "
                "firms or acquiring capabilities externally."
            )
        else:
            strategy = "Cost Optimisation"
            rationale = (
                "Low innovation index; prioritise foundational technology "
                "investments before scaling R&D spend."
            )

        confidence = float(min(max(signal * 0.85 + 0.10, 0), 1))

        return AgentRecommendation(
            agent_name=self.name,
            strategy=strategy,
            confidence=round(confidence, 4),
            rationale=rationale,
            score=round(signal, 4),
        )
