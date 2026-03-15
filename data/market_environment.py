"""
OmniMind AI — Market Environment
Defines and updates the global macro-economic market parameters that
influence company performance during simulation rounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MarketEnvironment:
    """
    Encapsulates the current market conditions.

    All attributes are floats in [0, 1] unless noted otherwise.
    """

    economic_pressure: float = 0.40       # 0 = benign, 1 = severe recession
    technology_disruption: float = 0.50   # 0 = stable, 1 = rapid disruption
    global_demand: float = 0.60           # 0 = weak, 1 = very strong
    market_growth_rate: float = 0.08      # annual growth rate (can be negative)

    # Internal RNG
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(42), repr=False
    )

    # ── Derived properties ─────────────────────────────────────────────────────

    @property
    def opportunity_index(self) -> float:
        """Composite market opportunity (higher is better)."""
        return (
            self.global_demand * 0.4
            + (1 - self.economic_pressure) * 0.3
            + (1 - self.technology_disruption) * 0.15
            + min(max(self.market_growth_rate / 0.30, 0), 1) * 0.15
        )

    @property
    def threat_index(self) -> float:
        """Composite market threat (higher is worse)."""
        return (
            self.economic_pressure * 0.4
            + self.technology_disruption * 0.35
            + (1 - self.global_demand) * 0.25
        )

    # ── Simulation helpers ─────────────────────────────────────────────────────

    def apply_shock(self, magnitude: float = 0.05) -> "MarketEnvironment":
        """
        Apply a random shock to all market parameters and return *self*
        to allow chaining.
        """
        noise = self._rng.normal(0, magnitude, size=4)
        self.economic_pressure = float(
            np.clip(self.economic_pressure + noise[0], 0, 1)
        )
        self.technology_disruption = float(
            np.clip(self.technology_disruption + noise[1], 0, 1)
        )
        self.global_demand = float(np.clip(self.global_demand + noise[2], 0, 1))
        self.market_growth_rate = float(
            np.clip(self.market_growth_rate + noise[3] * 0.02, -0.20, 0.40)
        )
        return self

    def to_dict(self) -> dict[str, float]:
        """Return parameters as a plain dictionary."""
        return {
            "economic_pressure": self.economic_pressure,
            "technology_disruption": self.technology_disruption,
            "global_demand": self.global_demand,
            "market_growth_rate": self.market_growth_rate,
            "opportunity_index": self.opportunity_index,
            "threat_index": self.threat_index,
        }

    @classmethod
    def from_dict(cls, params: dict[str, float]) -> "MarketEnvironment":
        """Construct a MarketEnvironment from a dictionary."""
        return cls(
            economic_pressure=params.get("economic_pressure", 0.40),
            technology_disruption=params.get("technology_disruption", 0.50),
            global_demand=params.get("global_demand", 0.60),
            market_growth_rate=params.get("market_growth_rate", 0.08),
        )
