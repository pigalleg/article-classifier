import numpy as np
import pandas as pd

from scripts.util.select_ministral_divergent_pairs import (
    DEFAULT_ANCHOR,
    DEFAULT_BIMODAL_PRIORITY,
    select_divergent_pairs,
)


ALL_MODELS = [DEFAULT_ANCHOR] + DEFAULT_BIMODAL_PRIORITY


def _level_of(score):
    if pd.isna(score):
        return np.nan
    score = float(score)
    if score <= 40:
        return "low"
    if score < 70:
        return "moderate"
    return "high"


def _make_pair_rows(abstract_id, question_id, anchor_score, bimodal_scores):
    rows = []
    for model in ALL_MODELS:
        if model == DEFAULT_ANCHOR:
            score = anchor_score
        else:
            score = bimodal_scores.get(model, 60)
        rows.append(
            {
                "abstract_id": abstract_id,
                "question_id": question_id,
                "model": model,
                "score": score,
                "level": _level_of(score),
            }
        )
    return rows


def _build_dataframe(rows):
    return pd.DataFrame(rows)


def test_select_divergent_pairs_defaults_to_one_per_band(tmp_path):
    rows = []
    rows += _make_pair_rows("a1", "q1", 30, {DEFAULT_BIMODAL_PRIORITY[0]: 5})
    rows += _make_pair_rows("a2", "q2", 30, {DEFAULT_BIMODAL_PRIORITY[0]: 15})
    rows += _make_pair_rows("a3", "q3", 50, {DEFAULT_BIMODAL_PRIORITY[0]: 0})
    rows += _make_pair_rows("a4", "q4", 50, {DEFAULT_BIMODAL_PRIORITY[0]: 20})

    df = _build_dataframe(rows)
    out_path = tmp_path / "selected.csv"

    selected, meta = select_divergent_pairs(
        df,
        output_csv=str(out_path),
        abstract_id_col="abstract_id",
        question_id_col="question_id",
        model_col="model",
        score_col="score",
        level_col="level",
    )

    assert out_path.exists()
    assert len(selected) == 2
    assert set(selected["sub_band"]) == {"21-40", "41-55"}
    assert selected["selection_rank"].tolist() == [1, 1]
    assert selected["selection_priority"].tolist() == [1, 1]
    assert meta["final_selected"] == 2


def test_select_divergent_pairs_can_fill_multiple_rows_per_band(tmp_path):
    rows = []
    rows += _make_pair_rows(
        "b1",
        "q1",
        50,
        {
            DEFAULT_BIMODAL_PRIORITY[0]: 0,
            DEFAULT_BIMODAL_PRIORITY[1]: 100,
        },
    )
    rows += _make_pair_rows(
        "b2",
        "q2",
        50,
        {
            DEFAULT_BIMODAL_PRIORITY[0]: np.nan,
            DEFAULT_BIMODAL_PRIORITY[1]: 5,
        },
    )
    rows += _make_pair_rows(
        "b3",
        "q3",
        50,
        {
            DEFAULT_BIMODAL_PRIORITY[0]: 15,
            DEFAULT_BIMODAL_PRIORITY[1]: 20,
        },
    )

    df = _build_dataframe(rows)
    out_path = tmp_path / "selected_multi.csv"

    selected, meta = select_divergent_pairs(
        df,
        output_csv=str(out_path),
        abstract_id_col="abstract_id",
        question_id_col="question_id",
        model_col="model",
        score_col="score",
        level_col="level",
        per_band=3,
    )

    assert out_path.exists()
    assert len(selected) == 3
    assert selected["sub_band"].tolist() == ["41-55", "41-55", "41-55"]
    assert selected[["abstract_id", "question_id"]].drop_duplicates().shape[0] == 3
    assert selected["selection_rank"].tolist() == [1, 2, 3]
    assert selected["selection_priority"].tolist() == [1, 2, 3]
    assert selected["primary_divergent_model"].tolist() == [
        DEFAULT_BIMODAL_PRIORITY[0],
        DEFAULT_BIMODAL_PRIORITY[1],
        DEFAULT_BIMODAL_PRIORITY[2],
    ]
    assert meta["final_selected"] == 3
