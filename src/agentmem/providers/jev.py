"""Jev (TypeSafe System One) adapter — the decision-model grader.

Jev does not generate text. You send one `state` plus typed questions
(noul = P(true), choice = pick one of N, score = position on an ordered
rubric) and get typed answers with probabilities back. Batched per request;
the questions run in parallel.

Eval-only: this client is never used by the chat path. The transport is
injected so tests run against recorded fixtures with no network and no key.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable

import httpx

from agentmem.settings import Settings, get_settings

MAX_ATTEMPTS = 3
# 529/524 are Cloudflare's "overloaded"/"timeout" transit signals on the
# TypeSafe edge; both are worth retrying exactly like a 503.
RETRY_STATUS = {429, 500, 502, 503, 504, 524, 529}

# Jev 1.14 pricing, anchored to real billing. The TypeSafe schedule is
# quoted per-billion tokens, which we previously mis-entered as per-million
# (a 1000x overcharge). Reconciliation: 566,932 input + 99,603 output tokens
# billed $0.0238 (usage-2026-10-23). Effective rates:
#   input  $0.042    per 1M tokens   (= $42 per 1B)
#   output $0.000042 per 1M tokens   (= $0.042 per 1B)
PRICE_IN = 0.042
PRICE_OUT = 0.000042


@dataclass(frozen=True)
class JevAnswer:
    """One typed answer. noul -> float in [0,1]; choice -> option name;
    score -> float along the rubric (may fall between levels)."""

    qid: str
    type: str
    value: float | str
    confidence: float | None = None
    probabilities: dict = field(default_factory=dict)


@dataclass(frozen=True)
class JevResult:
    model: str
    answers: dict[str, JevAnswer]
    input_tokens: int
    output_tokens: int
    latency_ms: int

    @property
    def cost_usd(self) -> float:
        return (self.input_tokens / 1e6 * PRICE_IN
                + self.output_tokens / 1e6 * PRICE_OUT)


# post(url, headers, payload) -> (status, parsed_json_body). The seam tests
# and the recorder both implement.
Transport = Callable[[str, dict, dict], tuple[int, dict]]


def _httpx_transport() -> Transport:
    client = httpx.Client(timeout=30.0)

    def post(url: str, headers: dict, payload: dict) -> tuple[int, dict]:
        resp = client.post(url, headers=headers, json=payload)
        return resp.status_code, resp.json()

    return post


class JevError(RuntimeError):
    pass


def _parse_answer(qid: str, body: dict) -> JevAnswer:
    atype = body.get("type")
    if atype == "noul":
        return JevAnswer(qid, "noul", float(body["noul"]))
    if atype == "choice":
        return JevAnswer(
            qid, "choice", body["choice"],
            confidence=body.get("confidence"),
            probabilities=body.get("probabilities") or {},
        )
    if atype == "score":
        return JevAnswer(
            qid, "score", float(body["score"]),
            confidence=body.get("confidence"),
            probabilities=body.get("probabilities") or {},
        )
    raise JevError(f"unknown answer type for {qid}: {atype!r}")


class JevClient:
    def __init__(self, settings: Settings | None = None,
                 transport: Transport | None = None):
        self.settings = settings or get_settings()
        self._post = transport or _httpx_transport()
        self.calls = 0
        self.usage = {"input": 0, "output": 0}
        self.total_cost_usd = 0.0
        self.misses: list[str] = []  # answers the model omitted, by qid

    @property
    def has_key(self) -> bool:
        return bool(self.settings.jev_key)

    def _request(self, payload: dict) -> dict:
        headers = {
            "Authorization": f"Bearer {self.settings.jev_key or 'unset'}",
            "Content-Type": "application/json",
        }
        url = f"{self.settings.jev_base_url}/v1/systemone"
        last = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            status, body = self._post(url, headers, payload)
            if status == 200:
                return body
            last = f"HTTP {status}: {json.dumps(body)[:300]}"
            if status not in RETRY_STATUS:
                break
            time.sleep(2 ** attempt)
        raise JevError(f"jev request failed — {last}")

    def evaluate(self, state, questions: dict[str, dict]) -> JevResult:
        """One batched call. questions: qid -> {"type": ...} dict in the raw
        TypeSafe shape (noul/choice/score). All questions see the same state,
        independently."""
        payload = {
            "model": self.settings.jev_model,
            "state": state,
            "questions": questions,
        }
        started = time.perf_counter()
        body = self._request(payload)
        latency_ms = int((time.perf_counter() - started) * 1000)
        answers: dict[str, JevAnswer] = {}
        got = body.get("answers") or {}
        for qid in questions:
            raw = got.get(qid)
            if raw is None:
                self.misses.append(qid)
                continue
            answers[qid] = _parse_answer(qid, raw)
        usage = body.get("usage") or {}
        in_tok, out_tok = int(usage.get("input_tokens") or 0), \
            int(usage.get("output_tokens") or 0)
        self.calls += 1
        self.usage["input"] += in_tok
        self.usage["output"] += out_tok
        r = JevResult(
            model=body.get("model") or self.settings.jev_model,
            answers=answers,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
        )
        self.total_cost_usd += r.cost_usd
        return r

    def models(self) -> list[str]:
        """GET /v1/models — the smoke call: proves the key and lists IDs."""
        headers = {"Authorization": f"Bearer {self.settings.jev_key or 'unset'}"}
        url = f"{self.settings.jev_base_url}/v1/models"
        status, body = self._get(url, headers)
        if status != 200:
            raise JevError(f"models listing failed — HTTP {status}")
        data = body.get("data") or []
        return [m.get("id") or m.get("name") for m in data if m]

    def _get(self, url: str, headers: dict) -> tuple[int, dict]:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, headers=headers)
            return resp.status_code, resp.json()
