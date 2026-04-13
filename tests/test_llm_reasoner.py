import time


class DummyChoice:
    class Message:
        content = '{"RA2025_ID": "42", "Reason": "Match reason", "Confidence": 0.7}'

    message = Message()


class DummyResponse:
    choices = [DummyChoice()]


class DummyCompletions:
    def create(self, model, messages, temperature):
        return DummyResponse()


class DummyOpenAI:
    def __init__(self, base_url=None, api_key=None, timeout=None):
        self.chat = type("c", (), {"completions": DummyCompletions()})()

def test_llm_reasoner_rate_limit(monkeypatch):
    """Validate that LLMReasoner enforces client-side sleeping when requests are too frequent.

    We patch the OpenAI client to return a deterministic JSON reply and patch
    time.sleep to capture calls without actually sleeping.
    """
    # Prepare a dummy OpenAI client replacement
    # Patch OpenAI in the module before importing LLMReasoner
    import importlib
    mod = importlib.import_module("src.models.llm_reasoner")
    monkeypatch.setattr(mod, "OpenAI", DummyOpenAI)

    # Capture sleep calls
    sleeps = []
    def fake_sleep(sec):
        sleeps.append(sec)

    monkeypatch.setattr("time.sleep", fake_sleep)

    # Now import and instantiate LLMReasoner
    from src.models.llm_reasoner import LLMReasoner

    # Create reasoner with low RPM (3 => 20s min interval)
    reasoner = LLMReasoner(requests_per_minute=3)

    # simulate a recent request so elapsed < min_interval
    reasoner._last_request_time = time.time()

    # Call classify_with_reasoning (should invoke time.sleep once)
    ra_id, reason, conf = reasoner.classify_with_reasoning("abstract text", [{"RA2025_ID": "1", "Question": "Q"}])

    assert ra_id == "42"
    assert "Match reason" in reason
    assert conf == 0.7
    # Check that we attempted to sleep at least once to respect RPM
    assert len(sleeps) >= 1
    # The sleep should be approximately the min_interval (allow small tolerance)
    assert abs(sleeps[0] - reasoner._min_interval) < 1.0


def test_few_shot_prompt_injection_enabled(monkeypatch, tmp_path):
    import importlib

    mod = importlib.import_module("src.models.llm_reasoner")
    monkeypatch.setattr(mod, "OpenAI", DummyOpenAI)

    ra_file = tmp_path / "few_shot_ra.yaml"
    ra_file.write_text(
        """
version: 1
examples:
  - id: ra_case
    target_type: RA
    abstract: Test abstract for RA.
    target_id: 42
    score: 77
    rationale: Useful calibration example.
""".strip()
        + "\n",
        encoding="utf-8",
    )

    prp_file = tmp_path / "few_shot_prp.yaml"
    prp_file.write_text(
        """
version: 1
examples:
  - id: prp_case
    target_type: PRP
    abstract: Test abstract for PRP.
    target_id: Planning
    score: 81
    rationale: Useful PRP calibration example.
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("AFFINITY_ENABLE_FEW_SHOT", "1")
    monkeypatch.setenv("AFFINITY_FEW_SHOT_RA_FILE", str(ra_file))
    monkeypatch.setenv("AFFINITY_FEW_SHOT_PRP_FILE", str(prp_file))
    monkeypatch.setenv("AFFINITY_FEW_SHOT_MAX_EXAMPLES", "2")

    from src.models.llm_reasoner import LLMReasoner

    reasoner = LLMReasoner(requests_per_minute=0)
    prompt = reasoner._build_affinity_prompt(
        abstract="Current abstract",
        targets_text="1. [42] Current target",
        target_type="RA",
        required_ids=["42"],
    )
    assert "Few-shot scoring examples" in prompt
    assert "Test abstract for RA." in prompt
    assert "Target ID: 42" in prompt
    assert "Score: 77.0" in prompt


def test_few_shot_prompt_injection_disabled(monkeypatch, tmp_path):
    import importlib

    mod = importlib.import_module("src.models.llm_reasoner")
    monkeypatch.setattr(mod, "OpenAI", DummyOpenAI)

    ra_file = tmp_path / "few_shot_ra.yaml"
    ra_file.write_text(
        """
version: 1
examples:
  - id: ra_case
    target_type: RA
    abstract: Should not be included.
    target_id: 42
    score: 77
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("AFFINITY_ENABLE_FEW_SHOT", "0")
    monkeypatch.setenv("AFFINITY_FEW_SHOT_RA_FILE", str(ra_file))

    from src.models.llm_reasoner import LLMReasoner

    reasoner = LLMReasoner(requests_per_minute=0)
    prompt = reasoner._build_affinity_prompt(
        abstract="Current abstract",
        targets_text="1. [42] Current target",
        target_type="RA",
        required_ids=["42"],
    )
    assert "Few-shot scoring examples" not in prompt
    assert "Should not be included." not in prompt


def test_few_shot_multiple_labels_per_abstract(monkeypatch, tmp_path):
    import importlib

    mod = importlib.import_module("src.models.llm_reasoner")
    monkeypatch.setattr(mod, "OpenAI", DummyOpenAI)

    ra_file = tmp_path / "few_shot_ra.yaml"
    ra_file.write_text(
        """
version: 1
examples:
  - id: ra_multi
    target_type: RA
    abstract: Multi-label calibration abstract.
    targets:
      - target_id: "42"
        score: 88
        rationale: Primary alignment.
      - target_id: "43"
        score: 61
        rationale: Secondary alignment.
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("AFFINITY_ENABLE_FEW_SHOT", "1")
    monkeypatch.setenv("AFFINITY_FEW_SHOT_RA_FILE", str(ra_file))

    from src.models.llm_reasoner import LLMReasoner

    reasoner = LLMReasoner(requests_per_minute=0)
    prompt = reasoner._build_affinity_prompt(
        abstract="Current abstract",
        targets_text="1. [42] Current target\n2. [43] Other target",
        target_type="RA",
        required_ids=["42", "43"],
    )

    assert "Multi-label calibration abstract." in prompt
    assert "Label 1 Target ID: 42" in prompt
    assert "Label 1 Score: 88.0" in prompt
    assert "Label 2 Target ID: 43" in prompt
    assert "Label 2 Score: 61.0" in prompt
