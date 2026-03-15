"""
OmniMind AI — Monte Carlo Engine
Simulates *n_runs* stochastic futures for a given company and market,
returning percentile statistics used by the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import MONTE_CARLO_DEFAULT_RUNS, RANDOM_SEED
from data.market_environment import MarketEnvironment


@dataclass
class MonteCarloResult:
    """Aggregated results from a Monte Carlo simulation."""

    growth_mean: float
    growth_p5: float
    growth_p95: float
    growth_samples: np.ndarray

    risk_mean: float
    risk_p5: float
    risk_p95: float
    risk_samples: np.ndarray

    roi_mean: float
    roi_p5: float
    roi_p95: float
    roi_samples: np.ndarray

    strategy_outcomes: dict[str, float]  # strategy_label -> win_probability

    def to_summary_dict(self) -> dict[str, float]:
        return {
            "growth_mean": self.growth_mean,
            "growth_p5": self.growth_p5,
            "growth_p95": self.growth_p95,
            "risk_mean": self.risk_mean,
            "risk_p5": self.risk_p5,
            "risk_p95": self.risk_p95,
            "roi_mean": self.roi_mean,
            "roi_p5": self.roi_p5,
            "roi_p95": self.roi_p95,
        }


class MonteCarloEngine:
    """
    Runs Monte Carlo simulations to model future uncertainty.

    Parameters
    ----------
    n_runs : int
        Number of stochastic simulation runs.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_runs: int = MONTE_CARLO_DEFAULT_RUNS,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.n_runs = n_runs
        self.rng = np.random.default_rng(seed)

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        company_row: pd.Series,
        market: MarketEnvironment,
        base_growth: float,
        base_risk: float,
    ) -> MonteCarloResult:
        """
        Simulate *n_runs* futures for *company_row* given market conditions.

        Parameters
        ----------
        company_row : pd.Series
            Single company record.
        market : MarketEnvironment
            Current market environment.
        base_growth : float
            ML-predicted growth score (0–1).
        base_risk : float
            ML-predicted risk probability (0–1).
        """
        n = self.n_runs

        # Perturb base estimates with Gaussian noise
        growth_noise = self.rng.normal(0, 0.08, size=n)
        risk_noise = self.rng.normal(0, 0.06, size=n)

        # Market opportunity / threat adjustments
        market_boost = market.opportunity_index * self.rng.uniform(0, 0.15, size=n)
        market_drag = market.threat_index * self.rng.uniform(0, 0.15, size=n)

        # Innovation boost
        innovation_factor = company_row.get("innovation_index", 0.5) * 0.10

        growth_samples = np.clip(
            base_growth + growth_noise + market_boost - market_drag + innovation_factor,
            0,
            1,
        )
        risk_samples = np.clip(
            base_risk + risk_noise - market_boost + market_drag,
            0,
            1,
        )

        # ROI: driven by growth and inverse-risk
        roi_base = growth_samples * (1 - risk_samples * 0.5)
        roi_samples = np.clip(
            roi_base * self.rng.uniform(0.8, 1.4, size=n),
            -0.5,
            2.0,
        )

        strategy_outcomes = self._simulate_strategy_outcomes(
            growth_samples, risk_samples
        )

        return MonteCarloResult(
            growth_mean=float(growth_samples.mean()),
            growth_p5=float(np.percentile(growth_samples, 5)),
            growth_p95=float(np.percentile(growth_samples, 95)),
            growth_samples=growth_samples,
            risk_mean=float(risk_samples.mean()),
            risk_p5=float(np.percentile(risk_samples, 5)),
            risk_p95=float(np.percentile(risk_samples, 95)),
            risk_samples=risk_samples,
            roi_mean=float(roi_samples.mean()),
            roi_p5=float(np.percentile(roi_samples, 5)),
            roi_p95=float(np.percentile(roi_samples, 95)),
            roi_samples=roi_samples,
            strategy_outcomes=strategy_outcomes,
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    def _simulate_strategy_outcomes(
        self,
        growth_samples: np.ndarray,
        risk_samples: np.ndarray,
    ) -> dict[str, float]:
        """Return the win-probability for each strategy across simulations."""
        strategies = {
            "Aggressive Expansion": growth_samples > 0.65,
            "Steady Growth": (growth_samples > 0.45) & (risk_samples < 0.55),
            "Risk Mitigation": risk_samples < 0.35,
            "Innovation Focus": growth_samples > 0.55,
            "Cost Optimisation": risk_samples < 0.50,
            "Market Consolidation": (growth_samples > 0.40) & (risk_samples < 0.45),
        }
        return {k: float(v.mean()) for k, v in strategies.items()}
