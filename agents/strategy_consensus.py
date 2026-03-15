"""
OmniMind AI — Strategy Consensus Engine
Combines the four agent recommendations via weighted voting to produce
a single recommended strategy with confidence and ROI estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from config import AGENT_WEIGHTS, STRATEGIES
from data.market_environment import MarketEnvironment

from agents.growth_agent import GrowthAgent
from agents.risk_agent import RiskAgent
from agents.innovation_agent import InnovationAgent
from agents.finance_agent import FinanceAgent

if TYPE_CHECKING:
    from agents.growth_agent import AgentRecommendation


@dataclass
class ConsensusResult:
    """Final consensus output from all agents."""

    recommended_strategy: str
    confidence_score: float        # 0–1
    expected_roi: float            # fractional ROI (e.g. 0.20 = 20 %)
    risk_probability: float        # 0–1
    agent_breakdown: list["AgentRecommendation"]
    strategy_vote_scores: dict[str, float]


class StrategyConsensus:
    """
    Runs all four AI agents and combines their outputs through a
    weighted voting scheme.
    """

    def __init__(self) -> None:
        self._growth_agent = GrowthAgent()
        self._risk_agent = RiskAgent()
        self._innovation_agent = InnovationAgent()
        self._finance_agent = FinanceAgent()

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        company: pd.Series,
        market: MarketEnvironment,
        growth_score: float,
        risk_score: float,
        mc_roi_mean: float = 0.10,
    ) -> ConsensusResult:
        """
        Run all agents and return a :class:`ConsensusResult`.

        Parameters
        ----------
        company : pd.Series
            Company record.
        market : MarketEnvironment
            Current market environment.
        growth_score : float
            ML-predicted growth score.
        risk_score : float
            ML-predicted risk probability.
        mc_roi_mean : float
            Mean expected ROI from Monte Carlo simulation.
        """
        agents = [
            self._growth_agent,
            self._risk_agent,
            self._innovation_agent,
            self._finance_agent,
        ]
        weights = [
            AGENT_WEIGHTS["growth"],
            AGENT_WEIGHTS["risk"],
            AGENT_WEIGHTS["innovation"],
            AGENT_WEIGHTS["finance"],
        ]

        recommendations = [
            a.evaluate(company, market, growth_score, risk_score)
            for a in agents
        ]

        # Build strategy → weighted_score map
        strategy_scores: dict[str, float] = {s: 0.0 for s in STRATEGIES}
        for rec, weight in zip(recommendations, weights):
            if rec.strategy in strategy_scores:
                strategy_scores[rec.strategy] += rec.score * weight

        # Normalise
        total = sum(strategy_scores.values())
        if total > 0:
            strategy_scores = {k: v / total for k, v in strategy_scores.items()}

        best_strategy = max(strategy_scores, key=lambda k: strategy_scores[k])

        # Consensus confidence = weighted average of agent confidences
        confidence = float(
            sum(r.confidence * w for r, w in zip(recommendations, weights))
        )

        # ROI estimate: blend MC result with consensus signal
        consensus_signal = strategy_scores[best_strategy]
        expected_roi = float(
            mc_roi_mean * 0.60 + consensus_signal * 0.40 * mc_roi_mean * 2
        )

        return ConsensusResult(
            recommended_strategy=best_strategy,
            confidence_score=round(confidence, 4),
            expected_roi=round(expected_roi, 4),
            risk_probability=round(risk_score, 4),
            agent_breakdown=recommendations,
            strategy_vote_scores=strategy_scores,
        )
