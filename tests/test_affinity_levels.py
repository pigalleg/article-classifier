import pandas as pd

from src.analysis.affinity_levels import affinity_level_fixed


def test_affinity_level_fixed_six_bands_uses_20_point_boundaries():
    """Bands are left-closed: a score landing on a cut belongs to the band above."""
    scores = [-1, 0, 20, 20.1, 40, 40.1, 60, 60.1, 80, 80.1, 100, 100.1, None]

    levels = [affinity_level_fixed(score, six_bands=True) for score in scores]

    assert levels == [
        pd.NA,
        "Low",
        "Low+",
        "Low+",
        "Moderate",
        "Moderate",
        "Moderate+",
        "Moderate+",
        "High",
        "High",
        "High+",
        "High+",
        pd.NA,
    ]