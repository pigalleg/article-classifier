import time

def test_llm_reasoner_rate_limit(monkeypatch):
    """Validate that LLMReasoner enforces client-side sleeping when requests are too frequent.

    We patch the OpenAI client to return a deterministic JSON reply and patch
    time.sleep to capture calls without actually sleeping.
    """
    # Prepare a dummy OpenAI client replacement
    class DummyChoice:
        class Message:
            content = '{"RA2025_ID": "42", "Reason": "Match reason"}'
        message = Message()

    class DummyResponse:
        choices = [DummyChoice()]

    class DummyCompletions:
        def create(self, model, messages, temperature):
            return DummyResponse()

    class DummyOpenAI:
        def __init__(self, timeout=None):
            self.chat = type("c", (), {"completions": DummyCompletions()})()

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
    ra_id, reason = reasoner.classify_with_reasoning("abstract text", [{"RA2025_ID": "1", "Question": "Q"}])

    assert ra_id == "42"
    assert "Match reason" in reason
    # Check that we attempted to sleep at least once to respect RPM
    assert len(sleeps) >= 1
    # The sleep should be approximately the min_interval (allow small tolerance)
    assert abs(sleeps[0] - reasoner._min_interval) < 1.0
