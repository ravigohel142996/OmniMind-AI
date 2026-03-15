"""
OmniMind AI — Synthetic Company Generator
Generates a realistic dataset of companies used throughout the platform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import INDUSTRIES, NUM_COMPANIES, RANDOM_SEED


# ── Name components ────────────────────────────────────────────────────────────
_PREFIXES: list[str] = [
    "Apex", "Nova", "Zenith", "Vertex", "Nexus", "Orbit", "Prism",
    "Atlas", "Comet", "Pulse", "Sigma", "Delta", "Alpha", "Omega",
    "Titan", "Lyra", "Quasar", "Helix", "Vector", "Cipher",
]
_SUFFIXES: list[str] = [
    "Systems", "Technologies", "Solutions", "Dynamics", "Ventures",
    "Industries", "Labs", "Analytics", "Innovations", "Enterprises",
    "Networks", "Group", "Corp", "Partners", "Digital",
]


def _generate_names(n: int, rng: np.random.Generator) -> list[str]:
    """Generate *n* unique company names."""
    names: set[str] = set()
    while len(names) < n:
        p = _PREFIXES[rng.integers(0, len(_PREFIXES))]
        s = _SUFFIXES[rng.integers(0, len(_SUFFIXES))]
        names.add(f"{p} {s}")
    return list(names)


def generate_companies(
    n: int = NUM_COMPANIES,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic company dataset.

    Returns
    -------
    pd.DataFrame
        One row per company with the following columns:

        company_name, industry, employees, revenue, innovation_index,
        r_and_d_spend, market_growth, competition_level,
        technology_adoption
    """
    rng = np.random.default_rng(seed)

    industries = rng.choice(INDUSTRIES, size=n, replace=True)

    # Employee count — log-normal distribution.
    # mean=7.5 ≈ e^7.5 ≈ 1 800 (median), sigma=1.4 gives a realistic
    # range from ~10 (small start-ups) to ~500 000 (large enterprises).
    employees = (rng.lognormal(mean=7.5, sigma=1.4, size=n)).astype(int)
    employees = np.clip(employees, 10, 500_000)

    # Revenue — positively correlated with employees
    base_revenue = employees * rng.uniform(50_000, 300_000, size=n)
    revenue = np.clip(base_revenue, 1_000_000, 5e10)

    # Innovation index (0–1)
    innovation_index = np.clip(rng.beta(2, 3, size=n) + rng.uniform(-0.1, 0.1, size=n), 0, 1)

    # R&D spend as fraction of revenue (0–0.25)
    r_and_d_spend = np.clip(rng.beta(1.5, 5, size=n) * 0.3, 0, 0.25)

    # Market growth rate (-0.10 → +0.30)
    market_growth = np.clip(rng.normal(0.08, 0.06, size=n), -0.10, 0.30)

    # Competition level (0–1)
    competition_level = np.clip(rng.beta(3, 2, size=n), 0, 1)

    # Technology adoption (0–1)
    technology_adoption = np.clip(
        innovation_index * 0.6 + rng.uniform(0, 0.4, size=n), 0, 1
    )

    df = pd.DataFrame(
        {
            "company_name": _generate_names(n, rng),
            "industry": industries,
            "employees": employees,
            "revenue": revenue.round(2),
            "innovation_index": innovation_index.round(4),
            "r_and_d_spend": r_and_d_spend.round(4),
            "market_growth": market_growth.round(4),
            "competition_level": competition_level.round(4),
            "technology_adoption": technology_adoption.round(4),
        }
    )

    return df.reset_index(drop=True)
