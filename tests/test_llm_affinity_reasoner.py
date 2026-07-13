import importlib


class BatchReasonMessage:
    def __init__(self, content):
        self.content = content


class BatchReasonChoice:
    def __init__(self, content):
        self.message = BatchReasonMessage(content)


class BatchReasonResponse:
    def __init__(self, content):
        self.choices = [BatchReasonChoice(content)]


class BatchReasonCompletions:
    def __init__(self):
        self.prompts = []

    def create(self, model, messages, temperature):
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        if "affinity_scores" in prompt:
            content = (
                '{"belongs_any_prp": true, "membership_confidence": 91, '
                '"affinity_scores": {"Planning": {"score": 84, "reason": "Planning reason"}}}'
            )
        else:
            content = '{"Planning": {"score": 84, "reason": "Planning reason"}}'

        return BatchReasonResponse(content)


class BatchReasonOpenAI:
    def __init__(self, base_url=None, api_key=None, timeout=None):
        self.chat = type("c", (), {"completions": BatchReasonCompletions()})()


def test_few_shot_prompt_injection_enabled(monkeypatch, tmp_path):
    mod = importlib.import_module("src.models.llm_affinity_reasoner")
    monkeypatch.setattr(mod, "OpenAI", BatchReasonOpenAI)

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

    from src.models.llm_affinity_reasoner import LLMAffinityReasoner

    reasoner = LLMAffinityReasoner(requests_per_minute=0)
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
    mod = importlib.import_module("src.models.llm_affinity_reasoner")
    monkeypatch.setattr(mod, "OpenAI", BatchReasonOpenAI)

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

    from src.models.llm_affinity_reasoner import LLMAffinityReasoner

    reasoner = LLMAffinityReasoner(requests_per_minute=0)
    prompt = reasoner._build_affinity_prompt(
        abstract="Current abstract",
        targets_text="1. [42] Current target",
        target_type="RA",
        required_ids=["42"],
    )
    assert "Few-shot scoring examples" not in prompt
    assert "Should not be included." not in prompt


def test_few_shot_multiple_labels_per_abstract(monkeypatch, tmp_path):
    mod = importlib.import_module("src.models.llm_affinity_reasoner")
    monkeypatch.setattr(mod, "OpenAI", BatchReasonOpenAI)

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

    from src.models.llm_affinity_reasoner import LLMAffinityReasoner

    reasoner = LLMAffinityReasoner(requests_per_minute=0)
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


def test_few_shot_prompt_keeps_long_abstract_and_rationale(monkeypatch, tmp_path):
    mod = importlib.import_module("src.models.llm_affinity_reasoner")
    monkeypatch.setattr(mod, "OpenAI", BatchReasonOpenAI)

    long_abstract = "A" * 1400
    long_rationale = "B" * 500
    ra_file = tmp_path / "few_shot_ra.yaml"
    ra_file.write_text(
        f"""
version: 1
examples:
  - id: ra_long
    target_type: RA
    abstract: {long_abstract}
    target_id: 42
    score: 77
    rationale: {long_rationale}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("AFFINITY_ENABLE_FEW_SHOT", "1")
    monkeypatch.setenv("AFFINITY_FEW_SHOT_RA_FILE", str(ra_file))

    from src.models.llm_affinity_reasoner import LLMAffinityReasoner

    reasoner = LLMAffinityReasoner(requests_per_minute=0)
    prompt = reasoner._build_affinity_prompt(
        abstract="Current abstract",
        targets_text="1. [42] Current target",
        target_type="RA",
        required_ids=["42"],
    )

    assert long_abstract in prompt
    assert long_rationale in prompt


def test_affinity_reasons_enabled_for_ra_and_prp(monkeypatch):
    mod = importlib.import_module("src.models.llm_affinity_reasoner")
    monkeypatch.setattr(mod, "OpenAI", BatchReasonOpenAI)
    monkeypatch.setenv("AFFINITY_ENABLE_AFFINITY_REASONS", "1")

    from src.models.llm_affinity_reasoner import LLMAffinityReasoner

    reasoner = LLMAffinityReasoner(requests_per_minute=0)

    ra_scores = reasoner.rate_affinity_batch(
        "Abstract text",
        [{"id": "Planning", "text": "Planning question"}],
        target_type="RA",
    )
    assert ra_scores["Planning"]["score"] == 84.0
    assert ra_scores["Planning"]["reason"] == "Planning reason"
    assert "'score' and 'reason'" in reasoner.client.chat.completions.prompts[0]

    prp_result = reasoner.rate_prp_affinity_with_scope(
        "Abstract text",
        [{"id": "Planning", "text": "Planning programme"}],
        "General scope anchor",
    )
    assert prp_result["belongs_any_prp"] is True
    assert prp_result["membership_confidence"] == 91.0
    assert prp_result["scores"]["Planning"]["score"] == 84.0
    assert prp_result["scores"]["Planning"]["reason"] == "Planning reason"
    assert "'score' and 'reason'" in reasoner.client.chat.completions.prompts[1]


def test_affinity_reasons_disabled_keeps_score_only(monkeypatch):
    mod = importlib.import_module("src.models.llm_affinity_reasoner")
    monkeypatch.setattr(mod, "OpenAI", BatchReasonOpenAI)
    monkeypatch.setenv("AFFINITY_ENABLE_AFFINITY_REASONS", "0")

    from src.models.llm_affinity_reasoner import LLMAffinityReasoner

    reasoner = LLMAffinityReasoner(requests_per_minute=0)
    scores = reasoner.rate_affinity_batch(
        "Abstract text",
        [{"id": "Planning", "text": "Planning question"}],
        target_type="RA",
    )
    assert scores["Planning"] == 84.0
