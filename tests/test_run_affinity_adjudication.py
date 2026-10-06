import pandas as pd

from scripts.run_postprocessing_affinity_benchmark import (
    _collect_affinity_rows,
    _write_assembled_outputs,
)
from scripts.run_affinity_adjudication import run_prp_adjudication, run_ra_adjudication


class _DummyAdjudicator:
    model = "dummy-adjudicator"

    def adjudicate_batch(self, abstract, targets, target_type="PRP"):
        out = {}
        for t in targets:
            tid = str(t["id"])
            if target_type == "PRP" and tid == "Planning":
                out[tid] = {"score": 70, "reason": "resolved by adjudicator"}
            elif target_type == "RA" and tid.startswith("37"):
                out[tid] = {"score": 80, "reason": "resolved by adjudicator"}
            else:
                out[tid] = 55
        return out


def test_run_prp_adjudication_applies_adjudication_and_mean_fallback():
    merged = pd.DataFrame(
        [
            {
                "Abstract_Index": 1,
                "Document Title": "Doc A",
                "PRP_Name": "Planning",
                "PRP_Description": "Planning desc",
                "LLM_Affinity": 10,
                "LLM_Affinity_Reason": "low",
                "Model_Slug": "m1",
                "Run_ID": "run-x",
            },
            {
                "Abstract_Index": 1,
                "Document Title": "Doc A",
                "PRP_Name": "Planning",
                "PRP_Description": "Planning desc",
                "LLM_Affinity": 90,
                "LLM_Affinity_Reason": "high",
                "Model_Slug": "m2",
                "Run_ID": "run-x",
            },
            {
                "Abstract_Index": 1,
                "Document Title": "Doc A",
                "PRP_Name": "Planning",
                "PRP_Description": "Planning desc",
                "LLM_Affinity": 95,
                "LLM_Affinity_Reason": "high",
                "Model_Slug": "m3",
                "Run_ID": "run-x",
            },
            {
                "Abstract_Index": 1,
                "Document Title": "Doc A",
                "PRP_Name": "IBR",
                "PRP_Description": "IBR desc",
                "LLM_Affinity": 50,
                "LLM_Affinity_Reason": "mid",
                "Model_Slug": "m1",
                "Run_ID": "run-x",
            },
            {
                "Abstract_Index": 1,
                "Document Title": "Doc A",
                "PRP_Name": "IBR",
                "PRP_Description": "IBR desc",
                "LLM_Affinity": 52,
                "LLM_Affinity_Reason": "mid",
                "Model_Slug": "m2",
                "Run_ID": "run-x",
            },
            {
                "Abstract_Index": 1,
                "Document Title": "Doc A",
                "PRP_Name": "IBR",
                "PRP_Description": "IBR desc",
                "LLM_Affinity": 49,
                "LLM_Affinity_Reason": "mid",
                "Model_Slug": "m3",
                "Run_ID": "run-x",
            },
        ]
    )

    pair_out, rows_out = run_prp_adjudication(
        merged_prp=merged,
        abstract_lookup={"1": "abstract text"},
        reasoner=_DummyAdjudicator(),
        min_agreement=0.9,
        min_agreement_mean=0.9,
    )

    planning = pair_out[pair_out["PRP_Name"] == "Planning"].iloc[0]
    assert planning["Final_Affinity"] == 70.0
    assert planning["Final_Method"] == "adjudicated_llm"
    assert planning["Final_Reason"] == "resolved by adjudicator"
    assert planning["Model_Slug"] == "dummy-adjudicator"

    ibr = pair_out[pair_out["PRP_Name"] == "IBR"].iloc[0]
    assert abs(float(ibr["Final_Affinity"]) - (50 + 52 + 49) / 3.0) < 1e-9
    assert ibr["Final_Method"] == "ensemble_mean"
    assert pd.isna(ibr["Model_Slug"])

    assert "Final_Affinity" in rows_out.columns
    assert len(rows_out) == len(merged)


def test_run_ra_adjudication_applies_adjudication_and_mean_fallback():
    merged = pd.DataFrame(
        [
            {
                "Abstract_Index": 1,
                "Document Title": "Doc A",
                "RA2025_ID": 37,
                "RA_Question": "Question 37",
                "Cosine_x100": 67.2,
                "LLM_Affinity": 5,
                "LLM_Affinity_Reason": "low",
                "Model_Slug": "m1",
                "Run_ID": "run-x",
            },
            {
                "Abstract_Index": 1,
                "Document Title": "Doc A",
                "RA2025_ID": 37,
                "RA_Question": "Question 37",
                "Cosine_x100": 67.2,
                "LLM_Affinity": 85,
                "LLM_Affinity_Reason": "high",
                "Model_Slug": "m2",
                "Run_ID": "run-x",
            },
            {
                "Abstract_Index": 1,
                "Document Title": "Doc A",
                "RA2025_ID": 37,
                "RA_Question": "Question 37",
                "Cosine_x100": 67.2,
                "LLM_Affinity": 90,
                "LLM_Affinity_Reason": "high",
                "Model_Slug": "m3",
                "Run_ID": "run-x",
            },
            {
                "Abstract_Index": 1,
                "Document Title": "Doc A",
                "RA2025_ID": 38,
                "RA_Question": "Question 38",
                "Cosine_x100": 56.5,
                "LLM_Affinity": 50,
                "LLM_Affinity_Reason": "mid",
                "Model_Slug": "m1",
                "Run_ID": "run-x",
            },
            {
                "Abstract_Index": 1,
                "Document Title": "Doc A",
                "RA2025_ID": 38,
                "RA_Question": "Question 38",
                "Cosine_x100": 56.5,
                "LLM_Affinity": 51,
                "LLM_Affinity_Reason": "mid",
                "Model_Slug": "m2",
                "Run_ID": "run-x",
            },
            {
                "Abstract_Index": 1,
                "Document Title": "Doc A",
                "RA2025_ID": 38,
                "RA_Question": "Question 38",
                "Cosine_x100": 56.5,
                "LLM_Affinity": 49,
                "LLM_Affinity_Reason": "mid",
                "Model_Slug": "m3",
                "Run_ID": "run-x",
            },
        ]
    )

    pair_out, rows_out = run_ra_adjudication(
        merged_ra=merged,
        abstract_lookup={"1": "abstract text"},
        reasoner=_DummyAdjudicator(),
        min_agreement=0.9,
        min_agreement_mean=0.9,
    )

    ra37 = pair_out[pair_out["RA2025_ID"] == 37].iloc[0]
    assert ra37["Final_Affinity"] == 80.0
    assert ra37["Final_Method"] == "adjudicated_llm"
    assert ra37["Final_Reason"] == "resolved by adjudicator"
    assert ra37["Model_Slug"] == "dummy-adjudicator"

    ra38 = pair_out[pair_out["RA2025_ID"] == 38].iloc[0]
    assert abs(float(ra38["Final_Affinity"]) - (50 + 51 + 49) / 3.0) < 1e-9
    assert ra38["Final_Method"] == "ensemble_mean"
    assert pd.isna(ra38["Model_Slug"])

    assert "Final_Affinity" in rows_out.columns
    assert len(rows_out) == len(merged)


def test_postprocessed_one_abstract_is_adjudicated_with_six_level_agreement(tmp_path):
    run_dir = tmp_path / "one-abstract-run"
    scores_by_model = {
        "model-a": 10,
        "model-b": 25,
        "model-c": 30,
    }

    for model_slug, score in scores_by_model.items():
        model_dir = run_dir / model_slug
        model_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "Abstract_Index": 1,
                    "Document Title": "Single test abstract",
                    "RA2025_ID": 99,
                    "RA_Question": "Question 99",
                    "LLM_Affinity": score,
                    "LLM_Affinity_Reason": f"evidence from {model_slug}",
                }
            ]
        ).to_csv(model_dir / "ra_affinities.csv", index=False)

    _prp_frames, ra_frames = _collect_affinity_rows(run_dir)
    outputs = _write_assembled_outputs(run_dir, _prp_frames, ra_frames)
    assert outputs["ra_rows"] == 3

    merged_ra = pd.read_csv(run_dir / "merged_ra_affinities.csv")
    assert set(merged_ra["Model_Slug"]) == set(scores_by_model)
    assert set(merged_ra["Run_ID"]) == {"one-abstract-run"}

    class CapturingAdjudicator:
        model = "test-adjudicator"

        def __init__(self):
            self.calls = []

        def adjudicate_batch(self, abstract, targets, target_type="PRP"):
            self.calls.append((abstract, targets, target_type))
            return {"99": {"score": 75, "reason": "resolved six-band disagreement"}}

    reasoner = CapturingAdjudicator()
    pair_out, model_rows_out = run_ra_adjudication(
        merged_ra=merged_ra,
        abstract_lookup={"1": "The one abstract being tested."},
        reasoner=reasoner,
        min_agreement=0.5,
        min_agreement_mean=0.8,
    )

    pair = pair_out.iloc[0]
    assert pair["agreement_min"] == 1 / 3
    assert pair["agreement_mean"] == 0.5
    assert pair["Final_Affinity"] == 75
    assert pair["Final_Method"] == "adjudicated_llm"
    assert pair["Final_Reason"] == "resolved six-band disagreement"
    assert len(reasoner.calls) == 1
    abstract, targets, target_type = reasoner.calls[0]
    assert abstract == "The one abstract being tested."
    assert target_type == "RA"
    assert len(targets) == 1
    assert {
        evidence["model_slug"]
        for target in targets
        for evidence in target["model_evidence"]
    } == set(scores_by_model)
    assert len(model_rows_out) == 3
    assert set(model_rows_out["Final_Method"]) == {"adjudicated_llm"}
