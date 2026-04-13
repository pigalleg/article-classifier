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
    """Validate that LLMReasoner enforces client-side sleeping when requests are too frequent."""
    import importlib

    mod = importlib.import_module("src.models.llm_classification_reasoner")
    monkeypatch.setattr(mod, "OpenAI", DummyOpenAI)

    sleeps = []

    def fake_sleep(sec):
        sleeps.append(sec)

    monkeypatch.setattr("time.sleep", fake_sleep)

    from src.models.llm_classification_reasoner import LLMReasoner

    reasoner = LLMReasoner(requests_per_minute=3)
    reasoner._last_request_time = time.time()

    ra_id, reason, conf = reasoner.classify_with_reasoning("abstract text", [{"RA2025_ID": "1", "Question": "Q"}])

    assert ra_id == "42"
    assert "Match reason" in reason
    assert conf == 0.7
    assert len(sleeps) >= 1
    assert abs(sleeps[0] - reasoner._min_interval) < 1.0
