"""Assign abstracts to PRPs from adjudicated RA question scores.

Every High programme is retained, and Moderate programmes are retained when
their strongest question is within the configured margin of the abstract's
strongest programme. When no Moderate or High programme exists, Low programmes
within that margin are retained as fallbacks. One strong question is sufficient
to assign a programme.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import yaml

from .affinity_levels import affinity_level_fixed

RA_PROGRAMME_MAPPING_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "ra_grouping_ra2025.csv"
)
SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"

LowLevelStrategy = Literal["fallback", "report_all"]


def _load_relative_margin() -> float:
    settings = yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8"))
    return float(settings["runtime"]["affinity"]["prp_assignment"]["relative_margin"])


def _load_programme_mapping() -> pd.DataFrame:
    return pd.read_csv(RA_PROGRAMME_MAPPING_PATH)


def _merge_programme_mapping(
    ra_adjudication_df: pd.DataFrame,
    programme_mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    mapped_rows = ra_adjudication_df.copy()
    mapped_rows["RA2025_ID"] = pd.to_numeric(
        mapped_rows["RA2025_ID"], errors="coerce"
    ).astype("Int64")

    programme_mapping = (
        programme_mapping_df[["RA2025", "Grouping"]]
        .dropna(subset=["RA2025", "Grouping"])
        .drop_duplicates()
        .rename(columns={"RA2025": "RA2025_ID", "Grouping": "PRP_Name"})
    )
    # programme_mapping = (
    #         programme_mapping_df[["RA2025", "Subgrouping_2"]]
    #         .dropna(subset=["RA2025", "Subgrouping_2"])
    #         .drop_duplicates()
    #         .rename(columns={"RA2025": "RA2025_ID", "Subgrouping_2": "PRP_Name"})
    #     )

    programme_mapping["RA2025_ID"] = pd.to_numeric(
        programme_mapping["RA2025_ID"], errors="coerce"
    ).astype("Int64")

    return mapped_rows.merge(programme_mapping, on="RA2025_ID", how="left")


def map_questions_to_programmes(ra_adjudication_df: pd.DataFrame) -> pd.DataFrame:
    """Add PRP programmes using the canonical RA programme mapping."""
    return _merge_programme_mapping(
        ra_adjudication_df,
        _load_programme_mapping(),
    )


def apply_affinity_level_fixed(
    affinity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add affinity levels using the adjudication pipeline's fixed scale."""
    result = affinity_df.copy()
    result["Final_Affinity"] = pd.to_numeric(
        result["Final_Affinity"], errors="coerce"
    )
    result["Affinity_Level"] = result["Final_Affinity"].apply(
        affinity_level_fixed
    )
    return result


def select_prp_assignments(
    mapped_question_scores: pd.DataFrame,
    low_level_strategy: LowLevelStrategy = "fallback",
) -> pd.DataFrame:
    """Select one or more PRPs for each abstract using maximum question scores.

    High-affinity programmes are always selected. Moderate programmes are
    selected when their maximum is within ``relative_margin`` of the abstract's
    best programme. If an abstract has no Moderate or High programme, Low
    programmes within the margin are selected as fallbacks. ``report_all``
    retains all Low programmes and adds an ``Is_Selected`` indicator;
    ``fallback`` retains only selected Low programmes. Both strategies apply
    the same selection rules to Moderate and High programmes.
    """
    if low_level_strategy not in {"fallback", "report_all"}:
        raise ValueError(
            "low_level_strategy must be either 'fallback' or 'report_all'"
        )

    question_scores = mapped_question_scores.copy()
    question_scores["Final_Affinity"] = pd.to_numeric(
        question_scores["Final_Affinity"], errors="coerce"
    )
    question_scores = question_scores.dropna(
        subset=["Abstract_Index", "PRP_Name", "RA2025_ID", "Final_Affinity"]
    )

    winning_columns = [
        "Abstract_Index",
        "PRP_Name",
        "RA2025_ID",
        "Final_Affinity",
        "Affinity_Level",
        "RA_Question",
    ]

    programme_scores = (
        question_scores.sort_values(
            ["Abstract_Index", "PRP_Name", "Final_Affinity", "RA2025_ID"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["Abstract_Index", "PRP_Name"], keep="first")[
            winning_columns
        ]
        .rename(
            columns={
                "RA2025_ID": "Highest_Question_ID",
                "RA_Question": "Highest_Question_Text",
                "Final_Affinity": "Highest_Question_Affinity",
            }
        )
    )

    programme_scores["Best_Abstract_Affinity"] = programme_scores.groupby(
        "Abstract_Index"
    )[
        "Highest_Question_Affinity"
    ].transform("max")
    programme_scores["Affinity_Gap"] = (
        programme_scores["Best_Abstract_Affinity"]
        - programme_scores["Highest_Question_Affinity"]
    )
    relative_margin = _load_relative_margin()
    within_margin = programme_scores["Affinity_Gap"] <= relative_margin
    high_affinity = programme_scores["Affinity_Level"].eq("High")
    moderate_affinity = programme_scores["Affinity_Level"].eq("Moderate")
    low_affinity = programme_scores["Affinity_Level"].eq("Low")
    has_non_low = (high_affinity | moderate_affinity).groupby(
        programme_scores["Abstract_Index"]
    ).transform("any")
    low_fallback = low_affinity & ~has_non_low & within_margin
    is_selected = high_affinity | (moderate_affinity & within_margin) | low_fallback

    programme_scores["Selection_Reason"] = "not_selected"
    programme_scores.loc[
        moderate_affinity & within_margin,
        "Selection_Reason",
    ] = "within_relative_margin"
    programme_scores.loc[high_affinity, "Selection_Reason"] = "high_affinity"
    programme_scores.loc[
        high_affinity & within_margin,
        "Selection_Reason",
    ] = "high_affinity_and_within_margin"
    programme_scores.loc[
        low_fallback,
        "Selection_Reason",
    ] = "low_fallback_within_relative_margin"
    programme_scores["Is_Selected"] = is_selected

    result = (
        programme_scores[is_selected | low_affinity]
        if low_level_strategy == "report_all"
        else programme_scores[is_selected].drop(columns="Is_Selected")
    )
    return result.sort_values(
        ["Abstract_Index", "Highest_Question_Affinity", "PRP_Name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def label_all_prp_affinities(
    mapped_question_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Label every abstract-programme pair by its strongest question score.

    This function does not select or discard programmes. The canonical affinity
    scale labels each programme Low, Moderate, or High using the maximum affinity
    among its related RA questions.
    """
    question_scores = mapped_question_scores.copy()
    question_scores["Final_Affinity"] = pd.to_numeric(
        question_scores["Final_Affinity"], errors="coerce"
    )
    question_scores = question_scores.dropna(
        subset=["Abstract_Index", "PRP_Name", "RA2025_ID", "Final_Affinity"]
    )

    programme_affinities = (
        question_scores.sort_values(
            ["Abstract_Index", "PRP_Name", "Final_Affinity", "RA2025_ID"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["Abstract_Index", "PRP_Name"], keep="first")[
            [
                "Abstract_Index",
                "PRP_Name",
                "RA2025_ID",
                "RA_Question",
                "Final_Affinity",
            ]
        ]
        .rename(
            columns={
                "RA2025_ID": "Highest_Question_ID",
                "RA_Question": "Highest_Question_Text",
                "Final_Affinity": "Highest_Question_Affinity",
            }
        )
    )
    programme_affinities["Affinity_Level"] = programme_affinities[
        "Highest_Question_Affinity"
    ].apply(affinity_level_fixed)

    return programme_affinities.sort_values(
        ["Abstract_Index", "Highest_Question_Affinity", "PRP_Name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def assign_prps(
    ra_adjudication_df: pd.DataFrame,
    low_level_strategy: LowLevelStrategy = "report_all",
) -> pd.DataFrame:
    """Map, label, and assign PRPs from adjudicated RA question scores."""
    mapped_questions = map_questions_to_programmes(ra_adjudication_df)
    levelled_questions = apply_affinity_level_fixed(mapped_questions)
    # return select_prp_assignments(levelled_questions, low_level_strategy)
    return label_all_prp_affinities(levelled_questions)