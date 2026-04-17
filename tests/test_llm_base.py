class _DummyCompletions:
    def create(self, model, messages, temperature):
        return None


class _DummyOpenAI:
    def __init__(self, base_url=None, api_key=None, timeout=None):
        self.chat = type("c", (), {"completions": _DummyCompletions()})()


def test_llm_base_normalize_batch_scores_with_reasons(monkeypatch):
    from src.models.llm_base import LLMBase

    worker = LLMBase(
        model="dummy-model",
        requests_per_minute=0,
        api_key="dummy-key",
        openai_client_cls=_DummyOpenAI,
    )

    data = {
        "Planning": {"score": 84, "reason": "Good fit"},
        "Stability": 31,
    }
    targets = [
        {"id": "Planning", "text": "Planning target"},
        {"id": "Stability", "text": "Stability target"},
    ]

    out = worker._normalize_batch_scores(data, targets, enable_reasons=True)

    assert out["Planning"]["score"] == 84.0
    assert out["Planning"]["reason"] == "Good fit"
    assert out["Stability"]["score"] == 31.0
    assert out["Stability"]["reason"] is None


def test_llm_base_chunk_targets():
    from src.models.llm_base import LLMBase

    chunks = LLMBase._chunk_targets([1, 2, 3, 4, 5], 2)
    assert chunks == [[1, 2], [3, 4], [5]]
