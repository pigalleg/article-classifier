"""Canonical affinity-level classification."""

from __future__ import annotations

from typing import Any

import pandas as pd


def affinity_level_fixed(score: Any, six_bands: bool = False) -> Any:
    """Map a score to three affinity levels or, optionally, six 20-point bands."""
    if pd.isna(score):
        return pd.NA

    try:
        value = float(score)
    except (TypeError, ValueError):
        return pd.NA

    if six_bands:
        if 0 <= value < 20:
            return "Low"
        if 20 <= value < 40:
            return "Low+"
        if 40 <= value < 60:
            return "Moderate"
        if 60 <= value < 80:
            return "Moderate+"
        if 80 <= value < 100:
            return "High"
        if 100 <= value:
            return "High+"
    else:
        if 0 <= value < 40:
            return "Low"
        if 40 <= value < 80:
            return "Moderate"
        if 80 <= value:
            return "High"
    return pd.NA
