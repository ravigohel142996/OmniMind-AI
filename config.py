"""
OmniMind AI — Platform Configuration
Centralised constants and settings used across the application.
"""

from __future__ import annotations

# ── Platform identity ──────────────────────────────────────────────────────────
PLATFORM_NAME: str = "OmniMind AI"
PLATFORM_SUBTITLE: str = "Autonomous Decision Intelligence System"
PLATFORM_VERSION: str = "1.0.0"

# ── Dataset ────────────────────────────────────────────────────────────────────
NUM_COMPANIES: int = 120
RANDOM_SEED: int = 42

# ── Industries ─────────────────────────────────────────────────────────────────
INDUSTRIES: list[str] = [
    "Technology",
    "Healthcare",
    "Finance",
    "Manufacturing",
    "Retail",
    "Energy",
    "Telecommunications",
    "Automotive",
    "Aerospace",
    "Pharmaceuticals",
]

# ── Monte Carlo ────────────────────────────────────────────────────────────────
MONTE_CARLO_DEFAULT_RUNS: int = 500
MONTE_CARLO_MIN_RUNS: int = 100
MONTE_CARLO_MAX_RUNS: int = 1000

# ── Agent weights (must sum to 1.0) ────────────────────────────────────────────
AGENT_WEIGHTS: dict[str, float] = {
    "growth": 0.30,
    "risk": 0.25,
    "innovation": 0.25,
    "finance": 0.20,
}

# ── Strategy labels ────────────────────────────────────────────────────────────
STRATEGIES: list[str] = [
    "Aggressive Expansion",
    "Steady Growth",
    "Risk Mitigation",
    "Innovation Focus",
    "Cost Optimisation",
    "Market Consolidation",
    "Strategic Pivot",
    "Hold & Observe",
]

# ── UI ─────────────────────────────────────────────────────────────────────────
THEME_PRIMARY_COLOR: str = "#4F8BF9"
THEME_BACKGROUND: str = "#0E1117"
CHART_HEIGHT: int = 400
CHART_TEMPLATE: str = "plotly_dark"
