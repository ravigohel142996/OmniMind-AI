"""
OmniMind AI — Market Analysis
Aggregate and cross-sectional analytics across the company dataset
for the Market Intelligence dashboard section.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.market_environment import MarketEnvironment


class MarketAnalyser:
    """Generates cross-company and industry-level analytics."""

    def industry_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate key metrics by industry.

        Returns a DataFrame indexed by industry with columns:
        avg_growth, avg_risk, avg_innovation, avg_revenue,
        avg_competition, company_count
        """
        group = df.groupby("industry")
        summary = pd.DataFrame(
            {
                "avg_growth": group["growth_score"].mean(),
                "avg_risk": group["risk_probability"].mean(),
                "avg_innovation": group["innovation_index"].mean(),
                "avg_revenue": group["revenue"].mean(),
                "avg_competition": group["competition_level"].mean(),
                "company_count": group["company_name"].count(),
            }
        ).reset_index()
        return summary

    def opportunity_map(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a market opportunity map combining growth potential and
        risk exposure per industry.
        """
        summary = self.industry_summary(df)
        summary["opportunity_score"] = (
            summary["avg_growth"] * 0.5
            + summary["avg_innovation"] * 0.3
            + (1 - summary["avg_risk"]) * 0.2
        )
        return summary.sort_values("opportunity_score", ascending=False)

    def top_companies(
        self,
        df: pd.DataFrame,
        metric: str = "growth_score",
        n: int = 10,
    ) -> pd.DataFrame:
        """Return the top *n* companies ranked by *metric*."""
        return (
            df[["company_name", "industry", metric]]
            .sort_values(metric, ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def growth_vs_competition(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a scatter-ready DataFrame of growth vs competition."""
        return df[
            [
                "company_name",
                "industry",
                "growth_score",
                "competition_level",
                "revenue",
                "innovation_index",
            ]
        ].copy()

    def innovation_vs_revenue(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a scatter-ready DataFrame of innovation vs revenue."""
        return df[
            [
                "company_name",
                "industry",
                "innovation_index",
                "revenue",
                "r_and_d_spend",
                "growth_score",
            ]
        ].copy()

    def market_heatmap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot table of average growth by industry × technology_adoption_tier.
        """
        out = df.copy()
        out["adoption_tier"] = pd.cut(
            out["technology_adoption"],
            bins=[0, 0.33, 0.66, 1.0],
            labels=["Low", "Medium", "High"],
            include_lowest=True,
        )
        pivot = out.pivot_table(
            values="growth_score",
            index="industry",
            columns="adoption_tier",
            aggfunc="mean",
        ).fillna(0)
        return pivot
