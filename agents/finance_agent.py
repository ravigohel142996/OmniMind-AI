"""
OmniMind AI — Finance Agent
Evaluates a company from a cost-control and financial-health perspective.
"""

from __future__ import annotations

import math
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


class FinanceAgent:
    """
    Controls costs and ensures financial sustainability.
    """

    name: str = "Finance Agent"

    def evaluate(
        self,
        company: pd.Series,
        market: MarketEnvironment,
        growth_score: float,
        risk_score: float,
    ) -> AgentRecommendation:
        """Evaluate the company and return a finance-focused recommendation."""
        revenue = float(company.get("revenue", 1_000_000))
        employees = int(company.get("employees", 100))
        r_and_d = float(company.get("r_and_d_spend", 0.05))

        # Revenue-per-employee as a cost-efficiency proxy (log-scaled 0-1)
        rev_per_emp = revenue / max(employees, 1)
        efficiency = float(
            min(math.log1p(rev_per_emp) / math.log1p(5_000_000), 1)
        )

        # Cost pressure: high R&D + high risk = more financial stress
        cost_pressure = r_and_d * 0.3 + risk_score * 0.4 + market.economic_pressure * 0.3
        cost_pressure = float(min(max(cost_pressure, 0), 1))

        # Financial health = efficiency vs cost pressure
        health = (efficiency * 0.6 + (1 - cost_pressure) * 0.4)
        health = float(min(max(health, 0), 1))

        if health >= 0.70:
            strategy = "Aggressive Expansion"
            rationale = (
                "Solid financial health and low cost pressure allow for "
                "capital deployment into growth initiatives."
            )
        elif health >= 0.55:
            strategy = "Steady Growth"
            rationale = (
                "Healthy financials support moderate investment; monitor "
                "burn rate as expansion progresses."
            )
        elif health >= 0.40:
            strategy = "Cost Optimisation"
            rationale = (
                "Rising cost pressure warrants a cost-reduction programme: "
                "streamline operations and defer discretionary spend."
            )
        else:
            strategy = "Risk Mitigation"
            rationale = (
                "Financial stress is elevated — prioritise liquidity, reduce "
                "debt exposure, and pause hiring."
            )

        confidence = float(min(max(health * 0.90 + 0.05, 0), 1))

        return AgentRecommendation(
            agent_name=self.name,
            strategy=strategy,
            confidence=round(confidence, 4),
            rationale=rationale,
            score=round(health, 4),
        )
