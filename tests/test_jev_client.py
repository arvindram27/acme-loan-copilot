"""Jev client contract tests, offline: the transport seam runs on a fixture
response captured from the documented API shape (noul + choice + score)."""

import pytest

from agentmem.providers.jev import JevClient, JevError, JevResult, _parse_answer
from agentmem.settings import Settings

FIXTURE_BODY = {
    "model": "jev-1.14.0",
    "answers": {
        "numeric_provenance": {"type": "noul", "noul": 0.98},
        "tool_used": {
            "type": "choice", "choice": "one", "confidence": 0.91,
            "probabilities": {"zero": 0.02, "one": 0.91, "many": 0.07},
        },
        "quality": {
            "type": "score", "score": 1.7, "confidence": 0.55,
            "legend": {"0": "wrong", "1": "thin", "2": "ok", "3": "good"},
            "probabilities": {"0": 0.1, "1": 0.1, "2": 0.7, "3": 0.1},
        },
    },
    "usage": {"input_tokens": 120, "output_tokens": 12},
}


class FakeTransport:
    """Records requests; replays one canned response."""

    def __init__(self, body=FIXTURE_BODY, status=200):
        self.body = body
        self.status = status
        self.requests = []

    def __call__(self, url, headers, payload):
        import json

        self.requests.append({"url": url, "payload": json.loads(json.dumps(payload)),
                              "auth": headers.get("Authorization", "")})
        return self.status, self.body


def _client(transport=None, **overrides) -> JevClient:
    s = Settings(jev_key="test-key", _env_file=None, **overrides)
    return JevClient(s, transport=transport or FakeTransport())


QUESTIONS = {
    "numeric_provenance": {"type": "noul", "instructions": "x"},
    "tool_used": {"type": "choice", "instructions": "x",
                  "criteria": {"zero": "", "one": "", "many": ""}},
    "quality": {"type": "score", "instructions": "x",
                "criteria": ["wrong", "thin", "ok", "good"]},
}
STATE = {"question": "q", "answer": "a"}


def test_parse_batched_answers():
    jc = _client()
    r = jc.evaluate(STATE, QUESTIONS)
    assert set(r.answers) == {"numeric_provenance", "tool_used", "quality"}
    a = r.answers["numeric_provenance"]
    assert a.type == "noul" and a.value == pytest.approx(0.98)
    assert r.answers["tool_used"].value == "one"
    assert r.answers["quality"].value == pytest.approx(1.7)


def test_one_request_batches_all_questions():
    t = FakeTransport()
    jc = _client(t)
    r = jc.evaluate(STATE, QUESTIONS)
    assert len(t.requests) == 1, "all questions share one state -> one request"
    payload = t.requests[0]["payload"]
    assert set(payload["questions"]) == set(QUESTIONS)
    assert payload["state"] == STATE
    assert t.requests[0]["auth"] == "Bearer test-key"


def test_usage_and_cost_accumulate():
    jc = _client()
    _ = jc.evaluate(STATE, QUESTIONS)
    _ = jc.evaluate(STATE, QUESTIONS)
    assert jc.usage == {"input": 240, "output": 24}
    # 2 * (120/1e6*0.042 + 12/1e6*0.000042)
    assert 0.0000095 < jc.total_cost_usd < 0.000011


def test_cost_reconciles_with_recorded_billing():
    """Anchor pricing to real usage: 566,932 in + 99,603 out tokens billed
    $0.0238 (usage-2026-10-23). Guards against a re-introduced unit slip
    (the constants were once 1000x too high)."""
    r = JevResult(model="jev-1.14.0", answers={},
                  input_tokens=566932, output_tokens=99603, latency_ms=0)
    assert r.cost_usd == pytest.approx(0.0238, abs=1e-4)


def test_missing_answer_is_recorded_not_fatal():
    body = {**FIXTURE_BODY, "answers": {"numeric_provenance": FIXTURE_BODY["answers"]["numeric_provenance"]}}
    jc = _client(FakeTransport(body=body))
    r = jc.evaluate(STATE, QUESTIONS)
    assert "tool_used" in jc.misses and "quality" in jc.misses
    assert set(r.answers) == {"numeric_provenance"}


def test_retries_on_5xx_then_raises():
    class Flaky:
        def __init__(self):
            self.n = 0
        def __call__(self, url, headers, payload):
            self.n += 1
            if self.n <= 2:
                return 503, {"detail": "overloaded"}
            return 200, FIXTURE_BODY

    f = Flaky()
    jc = _client(f)
    r = jc.evaluate(STATE, QUESTIONS)
    assert f.n == 3 and r.model == "jev-1.14.0"

    class Always500(FakeTransport):
        def __call__(self, *a, **kw):
            return 500, {"detail": "broken"}

    with pytest.raises(JevError):
        _client(Always500()).evaluate(STATE, QUESTIONS)


def test_4xx_raises_immediately():
    with pytest.raises(JevError):
        _client(FakeTransport(status=401)).evaluate(STATE, QUESTIONS)


def test_parse_answer_rejects_unknown_type():
    with pytest.raises(JevError):
        _parse_answer("q", {"type": "box", "box": {}})


def test_noul_escalation_band_default():
    s = Settings(_env_file=None)
    assert s.jev_escalate_low < s.jev_escalate_high
    assert 0 <= s.jev_escalate_low and s.jev_escalate_high <= 1
