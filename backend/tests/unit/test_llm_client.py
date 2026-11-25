import groq
import httpx
import pytest

from app.core.config import Settings
from app.core.exceptions import LLMError
from app.generation.llm_client import GroqLLMClient, parse_json_object


class _Resp:
    def __init__(self, text):
        msg = type("M", (), {"content": text})
        self.choices = [type("C", (), {"message": msg})]
        self.usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})


def _rate_limited():
    req = httpx.Request("POST", "https://api.groq.com")
    return groq.RateLimitError("429", response=httpx.Response(429, request=req), body=None)


def make_client(behaviour):
    client = GroqLLMClient(Settings(_env_file=None, groq_api_key="test", groq_model="primary",
                                    groq_fallback_models=["second", "third"], llm_max_retries=1))
    calls = []

    def fake_create(model, *args, **kwargs):
        calls.append(model)
        return behaviour(model)

    client._create = fake_create
    return client, calls


def test_rate_limit_fails_over_to_next_model_without_retrying_same():
    def behaviour(model):
        if model == "primary":
            raise _rate_limited()
        return _Resp("ok")
    client, calls = make_client(behaviour)
    resp = client.complete("sys", [{"role": "user", "content": "hi"}])
    assert resp.model == "second" and resp.text == "ok"
    assert calls == ["primary", "second"]


def test_all_models_exhausted_raises_llm_error():
    client, calls = make_client(lambda m: (_ for _ in ()).throw(_rate_limited()))
    with pytest.raises(LLMError):
        client.complete("sys", [{"role": "user", "content": "hi"}])
    assert calls == ["primary", "second", "third"]


def test_parse_json_object_tolerates_fences_and_prose():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"intent": "support"} hope it helps') == {"intent": "support"}
