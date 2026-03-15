"""
OmniMind AI — Risk Agent
Evaluates a company from a risk-minimisation perspective.
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


class RiskAgent:
    """
    Minimises exposure to competitive and macro-economic threats.
    """

    name: str = "Risk Agent"

    def evaluate(
        self,
        company: pd.Series,
        market: MarketEnvironment,
        growth_score: float,
        risk_score: float,
    ) -> AgentRecommendation:
        """Evaluate the company and return a risk-focused recommendation."""
        threat = (
            risk_score * 0.45
            + market.threat_index * 0.30
            + company.get("competition_level", 0.5) * 0.25
        )
        threat = float(min(max(threat, 0), 1))

        if threat >= 0.70:
            strategy = "Risk Mitigation"
            rationale = (
                "Elevated risk exposure demands immediate defensive measures: "
                "reduce leverage, hedge positions, and delay non-essential hiring."
            )
        elif threat >= 0.50:
            strategy = "Cost Optimisation"
            rationale = (
                "Moderate risk levels — tighten operational costs and build "
                "cash reserves as a buffer."
            )
        elif threat >= 0.30:
            strategy = "Steady Growth"
            rationale = (
                "Risk is manageable; maintain current trajectory with periodic "
                "risk monitoring."
            )
        else:
            strategy = "Aggressive Expansion"
            rationale = (
                "Low risk environment provides a safe window for bolder strategic moves."
            )

        confidence = float(min(max(1 - threat * 0.4 + 0.2, 0), 1))

        return AgentRecommendation(
            agent_name=self.name,
            strategy=strategy,
            confidence=round(confidence, 4),
            rationale=rationale,
            score=round(1 - threat, 4),  # higher score = lower risk
        )
