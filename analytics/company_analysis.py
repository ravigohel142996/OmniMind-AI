"""
OmniMind AI — Company Analysis
Enriches a company record with model scores and produces
company-level analytics used by the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.market_environment import MarketEnvironment
from models.growth_model import GrowthModel
from models.risk_model import RiskModel
from models.hiring_model import HiringModel


@dataclass
class CompanyProfile:
    """Enriched company record with model predictions."""

    name: str
    industry: str
    employees: int
    revenue: float
    growth_score: float
    risk_probability: float
    hiring_probability: float
    innovation_index: float
    technology_adoption: float
    market_growth: float
    competition_level: float
    r_and_d_spend: float


class CompanyAnalyser:
    """
    Runs all ML models on a company dataset and produces per-company
    enriched profiles.
    """

    def __init__(
        self,
        growth_model: GrowthModel,
        risk_model: RiskModel,
        hiring_model: HiringModel,
    ) -> None:
        self._growth = growth_model
        self._risk = risk_model
        self._hiring = hiring_model

    def enrich(
        self,
        df: pd.DataFrame,
        market: MarketEnvironment,
    ) -> pd.DataFrame:
        """
        Add model predictions to *df* and return an enriched DataFrame.

        New columns: growth_score, risk_probability, hiring_probability
        """
        out = df.copy()

        # Inject market economic_pressure for risk model
        out["economic_pressure"] = market.economic_pressure

        out["growth_score"] = self._growth.predict(out)
        out["risk_probability"] = self._risk.predict_proba(out)

        # Hiring model needs growth_score
        out["hiring_probability"] = self._hiring.predict_proba(out)

        # Clean up temporary column
        out.drop(columns=["economic_pressure"], inplace=True, errors="ignore")

        return out

    def get_profile(
        self,
        enriched_df: pd.DataFrame,
        company_name: str,
    ) -> CompanyProfile:
        """Return a :class:`CompanyProfile` for the named company."""
        row = enriched_df[enriched_df["company_name"] == company_name].iloc[0]
        return CompanyProfile(
            name=row["company_name"],
            industry=row["industry"],
            employees=int(row["employees"]),
            revenue=float(row["revenue"]),
            growth_score=float(row["growth_score"]),
            risk_probability=float(row["risk_probability"]),
            hiring_probability=float(row["hiring_probability"]),
            innovation_index=float(row["innovation_index"]),
            technology_adoption=float(row["technology_adoption"]),
            market_growth=float(row["market_growth"]),
            competition_level=float(row["competition_level"]),
            r_and_d_spend=float(row["r_and_d_spend"]),
        )

    def radar_data(self, profile: CompanyProfile) -> dict[str, float]:
        """Return normalised values suitable for a radar chart."""
        return {
            "Growth Score": profile.growth_score,
            "Innovation": profile.innovation_index,
            "Tech Adoption": profile.technology_adoption,
            "Market Growth": float(
                np.clip((profile.market_growth + 0.10) / 0.40, 0, 1)
            ),
            "Hire Potential": profile.hiring_probability,
            "Safety (inv. risk)": 1 - profile.risk_probability,
        }
