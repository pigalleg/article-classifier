import pandas as pd

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
