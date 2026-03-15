"""
OmniMind AI — Future Scenarios
Generates named strategic scenarios with outcome projections.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import RANDOM_SEED
from data.market_environment import MarketEnvironment


@dataclass
class Scenario:
    """Represents a single named future scenario."""

    name: str
    description: str
    growth_delta: float   # expected change in growth score
    risk_delta: float     # expected change in risk probability
    roi_estimate: float   # estimated ROI multiplier


_BASE_SCENARIOS: list[dict] = [
    {
        "name": "Bull Market Boom",
        "description": "Strong global demand and low economic pressure drive rapid growth.",
        "growth_delta": +0.18,
        "risk_delta": -0.12,
        "roi_estimate": 1.35,
    },
    {
        "name": "Tech Disruption Wave",
        "description": "Rapid technology adoption reshapes competitive dynamics.",
        "growth_delta": +0.10,
        "risk_delta": +0.08,
        "roi_estimate": 1.15,
    },
    {
        "name": "Recessionary Pressure",
        "description": "Economic contraction reduces demand and tightens margins.",
        "growth_delta": -0.15,
        "risk_delta": +0.20,
        "roi_estimate": 0.75,
    },
    {
        "name": "Stable Expansion",
        "description": "Moderate, consistent growth with manageable competition.",
        "growth_delta": +0.06,
        "risk_delta": -0.03,
        "roi_estimate": 1.10,
    },
    {
        "name": "Hyper-Competition",
        "description": "New entrants flood the market, compressing margins.",
        "growth_delta": -0.05,
        "risk_delta": +0.15,
        "roi_estimate": 0.90,
    },
    {
        "name": "Global Expansion",
        "description": "Opening of new geographic markets accelerates revenue.",
        "growth_delta": +0.14,
        "risk_delta": +0.04,
        "roi_estimate": 1.25,
    },
]


def generate_scenarios(
    base_growth: float,
    base_risk: float,
    market: MarketEnvironment,
    seed: int = RANDOM_SEED,
) -> list[Scenario]:
    """
    Adjust base scenarios using the current market environment and
    company-specific baselines.

    Returns a list of :class:`Scenario` objects.
    """
    rng = np.random.default_rng(seed)
    scenarios: list[Scenario] = []

    # Market modifiers
    opportunity_mod = (market.opportunity_index - 0.5) * 0.10
    threat_mod = (market.threat_index - 0.5) * 0.10

    for s in _BASE_SCENARIOS:
        noise_g = float(rng.normal(0, 0.02))
        noise_r = float(rng.normal(0, 0.02))

        adj_growth = float(
            np.clip(base_growth + s["growth_delta"] + opportunity_mod + noise_g, 0, 1)
        )
        adj_risk = float(
            np.clip(base_risk + s["risk_delta"] + threat_mod + noise_r, 0, 1)
        )
        adj_roi = float(s["roi_estimate"] * (1 + opportunity_mod - threat_mod))

        scenarios.append(
            Scenario(
                name=s["name"],
                description=s["description"],
                growth_delta=round(adj_growth - base_growth, 4),
                risk_delta=round(adj_risk - base_risk, 4),
                roi_estimate=round(adj_roi, 4),
            )
        )

    return scenarios
