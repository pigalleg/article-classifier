import pandas as pd

import scripts.run_missing_affinity_repair as repair_module
from scripts.run_missing_affinity_repair import repair_missing_ra_affinities


class _FakeReasoner:
    def __init__(self):
        self.calls = []

    def rate_affinity_batch(self, abstract, targets, target_type):
        self.calls.append((abstract, targets, target_type))
        return {"10": {"score": 72, "reason": "retry score"}}


def test_resolve_profile_accepts_benchmark_directory_slug(monkeypatch):
    profile = type("Profile", (), {"name": "gemma4:31b-cloud"})()
    monkeypatch.setattr(repair_module, "_load_settings", lambda _: {})
    monkeypatch.setattr(repair_module, "_resolve_profiles", lambda _: ([profile], None))

    assert repair_module._resolve_profile("gemma4-31b-cloud") is profile


def test_repair_missing_ra_affinities_updates_only_valid_retry_scores():
    scores = pd.DataFrame(
        [
            {"Abstract_Index": "doi-a", "RA2025_ID": 10, "RA_Question": "Question 10", "LLM_Affinity": None, "LLM_Affinity_Reason": None},
            {"Abstract_Index": "doi-a", "RA2025_ID": 11, "RA_Question": "Question 11", "LLM_Affinity": "missing", "LLM_Affinity_Reason": None},
            {"Abstract_Index": "doi-b", "RA2025_ID": 10, "RA_Question": "Question 10", "LLM_Affinity": 45, "LLM_Affinity_Reason": "original"},
        ]
    )
    reasoner = _FakeReasoner()

    repaired, unresolved = repair_missing_ra_affinities(
        scores=scores,
        abstract_lookup={"doi-a": "Abstract A", "doi-b": "Abstract B"},
        reasoner=reasoner,
        repaired_at_utc="2026-08-07T00:00:00Z",
    )

    assert len(reasoner.calls) == 1
    assert repaired.loc[0, "LLM_Affinity"] == 72.0
    assert repaired.loc[0, "LLM_Affinity_Reason"] == "retry score"
    assert repaired.loc[0, "Was_Repaired"]
    assert repaired.loc[0, "Repair_Method"] == "retry_smaller_batch"
    assert repaired.loc[2, "LLM_Affinity"] == 45.0
    assert not repaired.loc[2, "Was_Repaired"]
    assert unresolved[["Abstract_Index", "RA2025_ID"]].to_dict("records") == [{"Abstract_Index": "doi-a", "RA2025_ID": 11}]