"""
OmniMind AI — Growth Prediction Model
Trains a Gradient Boosting regressor to predict a company's growth score.

Features : revenue, innovation_index, market_growth, technology_adoption
Target   : growth_score  (0–1, derived from synthetic labels)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import RANDOM_SEED

FEATURES: list[str] = [
    "revenue",
    "innovation_index",
    "market_growth",
    "technology_adoption",
]


def _build_labels(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Derive synthetic growth_score labels from company features."""
    score = (
        0.30 * (df["revenue"] / df["revenue"].max())
        + 0.25 * df["innovation_index"]
        + 0.25 * ((df["market_growth"] - df["market_growth"].min())
                  / (df["market_growth"].max() - df["market_growth"].min() + 1e-9))
        + 0.20 * df["technology_adoption"]
        + rng.normal(0, 0.03, size=len(df))
    )
    return np.clip(score.values, 0, 1)


class GrowthModel:
    """Encapsulates training and inference for the growth prediction model."""

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        self.seed = seed
        self._pipeline: Pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "gbr",
                    GradientBoostingRegressor(
                        n_estimators=150,
                        max_depth=4,
                        learning_rate=0.08,
                        random_state=seed,
                    ),
                ),
            ]
        )
        self._trained: bool = False

    def train(self, df: pd.DataFrame) -> "GrowthModel":
        """Fit the model on *df*."""
        rng = np.random.default_rng(self.seed)
        X = df[FEATURES].copy()
        y = _build_labels(df, rng)
        self._pipeline.fit(X, y)
        self._trained = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return growth scores for each row in *df* (values in [0, 1])."""
        if not self._trained:
            raise RuntimeError("Call .train() before .predict()")
        raw = self._pipeline.predict(df[FEATURES])
        return np.clip(raw, 0, 1)

    def predict_single(self, features: dict[str, float]) -> float:
        """Predict for a single observation passed as a dict."""
        row = pd.DataFrame([features])
        return float(self.predict(row)[0])
