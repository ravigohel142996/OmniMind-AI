"""
OmniMind AI — Shared Utilities
Helper functions used across the platform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalise(series: pd.Series) -> pd.Series:
    """Min-max normalise a pandas Series to [0, 1]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mn) / (mx - mn)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp *value* to [*low*, *high*]."""
    return max(low, min(high, value))


def format_currency(value: float, symbol: str = "$") -> str:
    """Format a numeric value as a human-readable currency string."""
    if value >= 1e9:
        return f"{symbol}{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{symbol}{value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{symbol}{value / 1e3:.2f}K"
    return f"{symbol}{value:.2f}"


def format_percentage(value: float) -> str:
    """Format a float (0–1 range) as a percentage string."""
    return f"{value * 100:.1f}%"


def weighted_average(values: list[float], weights: list[float]) -> float:
    """Return the weighted average of *values* using *weights*."""
    total_w = sum(weights)
    if total_w == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_w


def confidence_label(score: float) -> str:
    """Convert a 0–1 confidence score to a human-readable label."""
    if score >= 0.80:
        return "Very High"
    if score >= 0.65:
        return "High"
    if score >= 0.50:
        return "Moderate"
    if score >= 0.35:
        return "Low"
    return "Very Low"


def risk_label(probability: float) -> str:
    """Convert a 0–1 risk probability to a colour-coded label."""
    if probability >= 0.70:
        return "🔴 Critical"
    if probability >= 0.50:
        return "🟠 High"
    if probability >= 0.30:
        return "🟡 Moderate"
    return "🟢 Low"
