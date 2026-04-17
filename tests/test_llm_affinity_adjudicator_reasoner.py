import importlib


class _AdjMessage:
    def __init__(self, content):
        self.content = content


class _AdjChoice:
    def __init__(self, content):
        self.message = _AdjMessage(content)


class _AdjResponse:
    def __init__(self, content):
        self.choices = [_AdjChoice(content)]


class _AdjCompletions:
    def __init__(self):
        self.prompts = []

    def create(self, model, messages, temperature):
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        if "'score' and 'reason'" in prompt:
            content = '{"Planning": {"score": 72, "reason": "resolved from mixed evidence"}}'
        else:
            content = '{"Planning": 72}'
        return _AdjResponse(content)


class _AdjOpenAI:
    def __init__(self, base_url=None, api_key=None, timeout=None):
        self.chat = type("c", (), {"completions": _AdjCompletions()})()


def test_adjudicator_output_shape_with_reasons(monkeypatch):
    mod = importlib.import_module("src.models.llm_affinity_adjudicator_reasoner")
    monkeypatch.setattr(mod, "OpenAI", _AdjOpenAI)
    monkeypatch.setenv("AFFINITY_ENABLE_AFFINITY_REASONS", "1")

    from src.models.llm_affinity_adjudicator_reasoner import LLMAffinityAdjudicatorReasoner

    reasoner = LLMAffinityAdjudicatorReasoner(requests_per_minute=0)
    out = reasoner.adjudicate_batch(
        abstract="Test abstract",
        targets=[
            {
                "id": "Planning",
                "text": "Planning target",
                "model_evidence": [
                    {"model_slug": "m1", "llm_affinity": 85, "llm_affinity_reason": "high confidence"},
                    {"model_slug": "m2", "llm_affinity": 40, "llm_affinity_reason": "moderate confidence"},
                ],
            }
        ],
        target_type="PRP",
    )

    assert out["Planning"]["score"] == 72.0
    assert out["Planning"]["reason"] == "resolved from mixed evidence"


def test_adjudicator_output_shape_without_reasons(monkeypatch):
    mod = importlib.import_module("src.models.llm_affinity_adjudicator_reasoner")
    monkeypatch.setattr(mod, "OpenAI", _AdjOpenAI)
    monkeypatch.setenv("AFFINITY_ENABLE_AFFINITY_REASONS", "0")

    from src.models.llm_affinity_adjudicator_reasoner import LLMAffinityAdjudicatorReasoner

    reasoner = LLMAffinityAdjudicatorReasoner(requests_per_minute=0)
    out = reasoner.adjudicate_batch(
        abstract="Test abstract",
        targets=[
            {
                "id": "Planning",
                "text": "Planning target",
                "model_evidence": [
                    {"model_slug": "m1", "llm_affinity": 85, "llm_affinity_reason": "high confidence"},
                    {"model_slug": "m2", "llm_affinity": 40, "llm_affinity_reason": "moderate confidence"},
                ],
            }
        ],
        target_type="PRP",
    )

    assert out["Planning"] == 72.0


class _AdjCompletionsRA:
    def __init__(self):
        self.prompts = []

    def create(self, model, messages, temperature):
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        content = '{"37": {"score": 65, "reason": "resolved for RA evidence"}}'
        return _AdjResponse(content)


class _AdjOpenAIRA:
    def __init__(self, base_url=None, api_key=None, timeout=None):
        self.chat = type("c", (), {"completions": _AdjCompletionsRA()})()


def test_adjudicator_ra_target_type_uses_research_questions_prompt(monkeypatch):
    mod = importlib.import_module("src.models.llm_affinity_adjudicator_reasoner")
    monkeypatch.setattr(mod, "OpenAI", _AdjOpenAIRA)
    monkeypatch.setenv("AFFINITY_ENABLE_AFFINITY_REASONS", "1")

    from src.models.llm_affinity_adjudicator_reasoner import LLMAffinityAdjudicatorReasoner

    reasoner = LLMAffinityAdjudicatorReasoner(requests_per_minute=0)
    out = reasoner.adjudicate_batch(
        abstract="Test abstract",
        targets=[
            {
                "id": "37",
                "text": "RA question text",
                "model_evidence": [
                    {"model_slug": "m1", "llm_affinity": 85, "llm_affinity_reason": "high confidence"},
                    {"model_slug": "m2", "llm_affinity": 40, "llm_affinity_reason": "moderate confidence"},
                ],
            }
        ],
        target_type="RA",
    )

    assert out["37"]["score"] == 65.0
    assert out["37"]["reason"] == "resolved for RA evidence"
    
    prompt = reasoner.client.chat.completions.prompts[-1]
    assert "Research questions with model evidence" in prompt
