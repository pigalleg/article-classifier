import pandas as pd
import pytest

from src.analysis import prp_assignment
from src.analysis.prp_assignment import (
    apply_affinity_level_fixed,
    assign_prps,
    label_all_prp_affinities,
    map_questions_to_programmes,
    select_prp_assignments,
)

# Selection depends on runtime.affinity.prp_assignment.relative_margin. Pin it so
# these expectations stay stable when settings.yaml is retuned.
TEST_RELATIVE_MARGIN = 5.0


@pytest.fixture(autouse=True)
def pinned_relative_margin(monkeypatch):
    monkeypatch.setattr(
        prp_assignment, "_load_relative_margin", lambda: TEST_RELATIVE_MARGIN
    )


def test_map_questions_to_programmes_reads_canonical_mapping(monkeypatch):
    adjudication_df = pd.DataFrame(
        {"Abstract_Index": ["a", "a"], "RA2025_ID": [1, 2], "Final_Affinity": [90, 30]}
    )
    mapping_df = pd.DataFrame(
        {"RA2025": [1, 2], "Grouping": ["Planning", "Stability"]}
    )
    monkeypatch.setattr(pd, "read_csv", lambda path: mapping_df)

    result = map_questions_to_programmes(adjudication_df)

    assert result["PRP_Name"].tolist() == ["Planning", "Stability"]
    assert result["Final_Affinity"].tolist() == [90, 30]


def test_apply_affinity_level_fixed_uses_canonical_boundaries():
    """Bands are left-closed: 40 is Moderate and 80 is High, not the band below."""
    affinity_df = pd.DataFrame(
        {"Final_Affinity": [-1, 0, 39.9, 40, 79.9, 80, 120, None]}
    )

    result = apply_affinity_level_fixed(affinity_df)

    assert result["Affinity_Level"].astype("string").tolist() == [
        pd.NA,
        "Low",
        "Low",
        "Moderate",
        "Moderate",
        "High",
        "High",
        pd.NA,
    ]


def test_apply_affinity_level_fixed_can_use_six_bands():
    affinity_df = pd.DataFrame(
        {"Final_Affinity": [20, 20.1, 40, 40.1, 60, 60.1, 80, 80.1, 100, 100.1]}
    )

    result = apply_affinity_level_fixed(affinity_df, six_bands=True)

    assert result["Affinity_Level"].tolist() == [
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
    ]


def test_apply_affinity_level_fixed_calls_canonical_function(monkeypatch):
    observed_scores = []

    def fake_affinity_level_fixed(score):
        observed_scores.append(score)
        return "canonical"

    monkeypatch.setattr(
        prp_assignment,
        "affinity_level_fixed",
        fake_affinity_level_fixed,
    )

    result = apply_affinity_level_fixed(
        pd.DataFrame({"Final_Affinity": [10, 90]})
    )

    assert observed_scores == [10, 90]
    assert result["Affinity_Level"].tolist() == ["canonical", "canonical"]


def test_select_prp_assignments_uses_single_best_question_and_margin():
    mapped_scores = pd.DataFrame(
        {
            "Abstract_Index": ["a"] * 6 + ["b"] * 2,
            "PRP_Name": ["Planning", "Planning", "CROF", "DER", "IBR", "Stability", "CROF", "DER"],
            "RA2025_ID": [1, 2, 3, 4, 5, 6, 7, 8],
            "RA_Question": [f"Question {question_id}" for question_id in range(1, 9)],
            "Final_Affinity": [110, 10, 102, 85, 70, 40, 40, 20],
        }
    )
    mapped_scores = apply_affinity_level_fixed(mapped_scores)

    result = select_prp_assignments(mapped_scores)

    assert result[["Abstract_Index", "PRP_Name"]].values.tolist() == [
        ["a", "Planning"],
        ["a", "CROF"],
        ["a", "DER"],
        ["b", "CROF"],
    ]
    assert result["Highest_Question_ID"].tolist() == [1, 3, 4, 7]
    assert result["Highest_Question_Affinity"].tolist() == [110, 102, 85, 40]
    assert result["Affinity_Gap"].tolist() == [0, 8, 25, 0]
    assert result["Selection_Reason"].tolist() == [
        "high_affinity_and_within_margin",
        "high_affinity",
        "high_affinity",
        "within_relative_margin",
    ]


def test_select_prp_assignments_can_report_all_programmes():
    mapped_scores = pd.DataFrame(
        {
            "Abstract_Index": ["a", "a", "a", "b", "b", "b"],
            "PRP_Name": ["Planning", "IBR", "CROF", "DER", "IBR", "Stability"],
            "RA2025_ID": [1, 2, 3, 4, 5, 6],
            "RA_Question": [f"Question {question_id}" for question_id in range(1, 7)],
            "Final_Affinity": [90, 70, 39, 40, 36, 20],
        }
    )
    mapped_scores = apply_affinity_level_fixed(mapped_scores)

    result = select_prp_assignments(mapped_scores, low_level_strategy="report_all")

    assert result[["Abstract_Index", "PRP_Name"]].values.tolist() == [
        ["a", "Planning"],
        ["a", "CROF"],
        ["b", "DER"],
        ["b", "IBR"],
        ["b", "Stability"],
    ]
    assert result["Is_Selected"].tolist() == [True, False, True, False, False]
    assert result["Selection_Reason"].tolist() == [
        "high_affinity_and_within_margin",
        "not_selected",
        "within_relative_margin",
        "not_selected",
        "not_selected",
    ]


def test_select_prp_assignments_filters_with_canonical_levels():
    mapped_scores = pd.DataFrame(
        {
            "Abstract_Index": ["a", "a", "a", "a"],
            "PRP_Name": ["Planning", "CROF", "DER", "IBR"],
            "RA2025_ID": [1, 2, 3, 4],
            "RA_Question": ["Question 1", "Question 2", "Question 3", "Question 4"],
            "Final_Affinity": [110, 85, 70, 39],
        }
    )
    mapped_scores = apply_affinity_level_fixed(mapped_scores)

    result = select_prp_assignments(mapped_scores)

    assert result["PRP_Name"].tolist() == ["Planning", "CROF"]
    assert result["Affinity_Level"].tolist() == ["High", "High"]
    assert result["Selection_Reason"].tolist() == [
        "high_affinity_and_within_margin",
        "high_affinity",
    ]


def test_label_all_prp_affinities_keeps_every_programme_and_uses_maximum():
    mapped_scores = pd.DataFrame(
        {
            "Abstract_Index": ["a"] * 6,
            "PRP_Name": ["Planning", "Planning", "CROF", "CROF", "DER", "DER"],
            "RA2025_ID": [1, 2, 3, 4, 5, 6],
            "RA_Question": [f"Question {question_id}" for question_id in range(1, 7)],
            "Final_Affinity": [81, 20, 80, 45, 40, 10],
        }
    )

    result = label_all_prp_affinities(mapped_scores)

    assert result["PRP_Name"].tolist() == ["Planning", "CROF", "DER"]
    assert result["Highest_Question_ID"].tolist() == [1, 3, 5]
    assert result["Highest_Question_Affinity"].tolist() == [81, 80, 40]
    assert result["Affinity_Level"].tolist() == ["High", "High", "Moderate"]


def test_assign_prps_labels_every_programme(monkeypatch):
    """``assign_prps`` maps, levels, and labels without discarding programmes.

    Notebooks 17-19 consume one row per (abstract, programme), so the pipeline
    keeps every programme instead of applying ``select_prp_assignments``.
    """
    adjudication_df = pd.DataFrame(
        {
            "Abstract_Index": ["a", "a"],
            "RA2025_ID": [1, 2],
            "RA_Question": ["Question 1", "Question 2"],
            "Final_Affinity": [90, 30],
        }
    )
    mapping_df = pd.DataFrame(
        {"RA2025": [1, 2], "Grouping": ["Planning", "Stability"]}
    )
    monkeypatch.setattr(pd, "read_csv", lambda path: mapping_df)

    result = assign_prps(adjudication_df)

    assert result["PRP_Name"].tolist() == ["Planning", "Stability"]
    assert result["Highest_Question_ID"].tolist() == [1, 2]
    assert result["Highest_Question_Affinity"].tolist() == [90, 30]
    assert result["Affinity_Level"].tolist() == ["High", "Low"]


def test_assign_prps_can_use_six_bands(monkeypatch):
    adjudication_df = pd.DataFrame(
        {
            "Abstract_Index": ["a"] * 6,
            "RA2025_ID": [1, 2, 3, 4, 5, 6],
            "RA_Question": [f"Question {question_id}" for question_id in range(1, 7)],
            "Final_Affinity": [20, 40, 60, 80, 100, 101],
        }
    )
    mapping_df = pd.DataFrame(
        {
            "RA2025": [1, 2, 3, 4, 5, 6],
            "Grouping": ["Low", "Low plus", "Moderate", "Moderate plus", "High", "High plus"],
        }
    )
    monkeypatch.setattr(pd, "read_csv", lambda path: mapping_df)

    result = assign_prps(adjudication_df, six_bands=True)

    assert result["Affinity_Level"].tolist() == [
        "High+",
        "High+",
        "High",
        "Moderate+",
        "Moderate",
        "Low+",
    ]
