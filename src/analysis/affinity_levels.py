"""Canonical affinity-level classification."""

from __future__ import annotations

from typing import Any

import pandas as pd


def affinity_level_fixed(score: Any) -> Any:
    """Map a 0-120 affinity score to Low, Moderate, or High."""
    if pd.isna(score):
        return pd.NA

    try:
        value = float(score)
    except (TypeError, ValueError):
        return pd.NA

    if 0 <= value <= 40:
        return "Low"
    if 40 < value <= 80:
        return "Moderate"
    if value > 80:
        return "High"
    return pd.NA
