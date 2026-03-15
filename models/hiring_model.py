"""
OmniMind AI — Hiring Forecast Model
Trains a Gradient Boosting classifier to predict hiring probability.

Features : growth_score, market_growth, technology_adoption
Target   : hiring_probability  (0–1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import RANDOM_SEED

FEATURES: list[str] = [
    "growth_score",
    "market_growth",
    "technology_adoption",
]


def _build_labels(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Derive binary hiring labels."""
    score = (
        0.50 * df["growth_score"]
        + 0.30 * ((df["market_growth"] - df["market_growth"].min())
                  / (df["market_growth"].max() - df["market_growth"].min() + 1e-9))
        + 0.20 * df["technology_adoption"]
        + rng.normal(0, 0.04, size=len(df))
    )
    threshold = np.median(np.clip(score, 0, 1))
    return (score >= threshold).astype(int)


class HiringModel:
    """Encapsulates training and inference for the hiring forecast model."""

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        self.seed = seed
        self._pipeline: Pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "gbc",
                    GradientBoostingClassifier(
                        n_estimators=100,
                        max_depth=3,
                        learning_rate=0.10,
                        random_state=seed,
                    ),
                ),
            ]
        )
        self._trained: bool = False

    def train(self, df: pd.DataFrame) -> "HiringModel":
        """Fit the model.  *df* must contain a 'growth_score' column."""
        rng = np.random.default_rng(self.seed)
        X = df[FEATURES].copy()
        y = _build_labels(X, rng)
        self._pipeline.fit(X, y)
        self._trained = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return hiring probabilities (values in [0, 1]) for each row."""
        if not self._trained:
            raise RuntimeError("Call .train() before .predict_proba()")
        return self._pipeline.predict_proba(df[FEATURES])[:, 1]

    def predict_single(self, features: dict[str, float]) -> float:
        """Predict hiring probability for a single observation dict."""
        row = pd.DataFrame([features])
        return float(self.predict_proba(row)[0])
