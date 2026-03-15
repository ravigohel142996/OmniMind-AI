"""
OmniMind AI — Risk Prediction Model
Trains a Gradient Boosting classifier to estimate risk probability.

Features : competition_level, economic_pressure, revenue, employees
Target   : risk_probability  (0–1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import RANDOM_SEED

FEATURES: list[str] = [
    "competition_level",
    "economic_pressure",
    "revenue",
    "employees",
]


def _build_labels(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Derive binary risk labels from company features."""
    risk_score = (
        0.35 * df["competition_level"]
        + 0.30 * df["economic_pressure"]
        + 0.20 * (1 - df["revenue"] / df["revenue"].max())
        + 0.15 * (1 - np.log1p(df["employees"]) / np.log1p(df["employees"].max()))
        + rng.normal(0, 0.05, size=len(df))
    )
    # Binarise at median for classification
    threshold = np.median(np.clip(risk_score, 0, 1))
    return (risk_score >= threshold).astype(int)


class RiskModel:
    """Encapsulates training and inference for the risk prediction model."""

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        self.seed = seed
        self._pipeline: Pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "gbc",
                    GradientBoostingClassifier(
                        n_estimators=150,
                        max_depth=4,
                        learning_rate=0.08,
                        random_state=seed,
                    ),
                ),
            ]
        )
        self._trained: bool = False

    def train(self, df: pd.DataFrame) -> "RiskModel":
        """Fit the model.  *df* must contain an 'economic_pressure' column."""
        rng = np.random.default_rng(self.seed)

        # economic_pressure may not be in the company df — inject a default
        X = df[FEATURES].copy() if "economic_pressure" in df.columns else _inject_ep(df)
        y = _build_labels(X, rng)
        self._pipeline.fit(X, y)
        self._trained = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return risk probabilities (values in [0, 1]) for each row."""
        if not self._trained:
            raise RuntimeError("Call .train() before .predict_proba()")
        X = df[FEATURES].copy() if "economic_pressure" in df.columns else _inject_ep(df)
        return self._pipeline.predict_proba(X)[:, 1]

    def predict_single(self, features: dict[str, float]) -> float:
        """Predict risk probability for a single observation dict."""
        row = pd.DataFrame([features])
        return float(self.predict_proba(row)[0])


def _inject_ep(df: pd.DataFrame, default: float = 0.40) -> pd.DataFrame:
    """Add a default economic_pressure column when missing."""
    out = df.copy()
    out["economic_pressure"] = default
    return out[FEATURES]
