"""
OmniMind AI — Market Simulator
Steps through multiple simulation rounds, applying market shocks and
updating company performance metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import RANDOM_SEED
from data.market_environment import MarketEnvironment


class MarketSimulator:
    """
    Iterates market rounds, applying environmental shocks and tracking
    how company key metrics evolve over time.
    """

    def __init__(
        self,
        n_rounds: int = 12,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.n_rounds = n_rounds
        self.rng = np.random.default_rng(seed)

    def simulate(
        self,
        df: pd.DataFrame,
        market: MarketEnvironment,
    ) -> pd.DataFrame:
        """
        Simulate *n_rounds* market rounds for all companies in *df*.

        Returns
        -------
        pd.DataFrame
            Long-format dataframe with columns:
            company_name, round, revenue, growth_score, risk_score
        """
        records: list[dict] = []
        market_copy = MarketEnvironment(
            economic_pressure=market.economic_pressure,
            technology_disruption=market.technology_disruption,
            global_demand=market.global_demand,
            market_growth_rate=market.market_growth_rate,
        )

        revenues = df["revenue"].values.copy().astype(float)

        for rnd in range(self.n_rounds):
            market_copy.apply_shock(magnitude=0.04)

            # Company-level growth factors
            growth_factor = (
                1
                + market_copy.market_growth_rate
                + df["innovation_index"].values * 0.05
                - df["competition_level"].values * 0.03
                + market_copy.opportunity_index * 0.04
                - market_copy.threat_index * 0.04
                + self.rng.normal(0, 0.02, size=len(df))
            )
            revenues = revenues * np.clip(growth_factor, 0.80, 1.40)

            growth_scores = np.clip(
                (revenues / revenues.max()) * 0.6
                + df["innovation_index"].values * 0.4,
                0,
                1,
            )
            risk_scores = np.clip(
                df["competition_level"].values * 0.4
                + market_copy.threat_index * 0.4
                + self.rng.normal(0, 0.05, size=len(df)),
                0,
                1,
            )

            for i, name in enumerate(df["company_name"]):
                records.append(
                    {
                        "company_name": name,
                        "round": rnd + 1,
                        "revenue": float(revenues[i]),
                        "growth_score": float(growth_scores[i]),
                        "risk_score": float(risk_scores[i]),
                    }
                )

        return pd.DataFrame(records)
