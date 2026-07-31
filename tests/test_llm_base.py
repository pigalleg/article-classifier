class _DummyCompletions:
    def create(self, model, messages, temperature):
        return None


class _DummyOpenAI:
    def __init__(self, base_url=None, api_key=None, timeout=None):
        self.chat = type("c", (), {"completions": _DummyCompletions()})()


class _DummyUsage:
    def __init__(self, prompt_tokens=None, completion_tokens=None, total_tokens=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _DummyMessage:
    def __init__(self, content):
        self.content = content


class _DummyChoice:
    def __init__(self, content):
        self.message = _DummyMessage(content)


class _DummyResponse:
    def __init__(self, content, usage):
        self.choices = [_DummyChoice(content)]
        self.usage = usage


class _DummyCompletionsWithUsage:
    def __init__(self):
        self.calls = 0

    def create(self, model, messages, temperature):
        self.calls += 1
        usage = _DummyUsage(prompt_tokens=12, completion_tokens=8, total_tokens=20)
        return _DummyResponse('{"x": 1}', usage)


class _DummyOpenAIWithUsage:
    def __init__(self, base_url=None, api_key=None, timeout=None):
        self.chat = type("c", (), {"completions": _DummyCompletionsWithUsage()})()


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


def test_llm_base_tracks_token_usage_from_chat_response():
    from src.models.llm_base import LLMBase

    worker = LLMBase(
        model="dummy-model",
        requests_per_minute=0,
        api_key="dummy-key",
        openai_client_cls=_DummyOpenAIWithUsage,
    )

    data = worker._call_json_prompt_with_retries("Return JSON", temperature=0.0)

    assert data == {"x": 1}
    assert worker.get_token_usage_totals() == {
        "input_tokens": 12,
        "output_tokens": 8,
        "total_tokens": 20,
    }
