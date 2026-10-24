"""Metrics: k, Wilson intervals, percentiles, and grader-agreement tables.

Cohen's k measures a grader's agreement with labels beyond chance. On a
label set where one class dominates, k can be negative even with high raw
agreement — that is reported, not hidden: the eval prints both, and the
agreement table shows where a grader actually errs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Z95 = 1.96


def wilson(passed: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval — honest accuracy bounds at small n."""
    if n == 0:
        return (0.0, 1.0)
    p = passed / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def kappa(labels: list[bool], predictions: list[bool]) -> dict:
    """Cohen's k for binary agreement. Handles degenerate label sets."""
    n = len(labels)
    if n == 0 or len(predictions) != n:
        return {"kappa": None, "agreement": None, "n": n,
                "note": "empty or mismatched inputs"}
    agree = sum(1 for l, p in zip(labels, predictions) if l == p)
    po = agree / n
    p_yes_l = sum(labels) / n
    p_yes_p = sum(predictions) / n
    pe = p_yes_l * p_yes_p + (1 - p_yes_l) * (1 - p_yes_p)
    if pe == 1.0:
        # only one class in labels: k = 1 if perfect else 0 by convention
        k = 1.0 if po == 1.0 else 0.0
    else:
        k = (po - pe) / (1 - pe)
    confusion = {
        "tp": sum(1 for l, p in zip(labels, predictions) if l and p),
        "fp": sum(1 for l, p in zip(labels, predictions) if not l and p),
        "fn": sum(1 for l, p in zip(labels, predictions) if l and not p),
        "tn": sum(1 for l, p in zip(labels, predictions) if not l and not p),
    }
    return {
        "kappa": round(k, 4),
        "agreement": round(po, 4),
        "n": n,
        "true_yes": sum(labels),
        "pred_yes": sum(predictions),
        **confusion,
    }


def pctl(values: list[float], p: float) -> float:
    """Nearest-rank percentile on a possibly empty list."""
    if not values:
        return 0.0
    vs = sorted(values)
    idx = max(0, min(len(vs) - 1, math.ceil(p / 100 * len(vs)) - 1))
    return vs[idx]


@dataclass
class RunsSummary:
    accuracy: float
    accuracy_ci: tuple[float, float]
    n: int
    op_score: float          # memory-operation correctness
    op_ci: tuple[float, float]
    cache_hit_rate: float    # tool-result cache
    answer_cache_hit_rate: float   # semantic answer cache
    context_tokens_mean: float     # estimated prompt tokens per turn
    summary_used_frac: float       # turns that ran with a working-memory summary
    p50_ms: float
    p95_ms: float
    tokens_mean: float
    cost_usd: float

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy, "accuracy_ci": self.accuracy_ci,
            "n": self.n, "op_score": self.op_score, "op_ci": self.op_ci,
            "cache_hit_rate": self.cache_hit_rate,
            "answer_cache_hit_rate": self.answer_cache_hit_rate,
            "context_tokens_mean": self.context_tokens_mean,
            "summary_used_frac": self.summary_used_frac,
            "p50_ms": self.p50_ms, "p95_ms": self.p95_ms,
            "tokens_mean": self.tokens_mean, "cost_usd": self.cost_usd,
        }


def summarize(results: list[dict]) -> RunsSummary:
    """results: per-item dicts with keys answer_ok(bool), op_ok(bool|None),
    latency_ms, tokens, cost_usd, cache_hit(bool), cache_eligible(bool),
    answer_cache_hit(bool), answer_cache_eligible(bool), context_tokens(int),
    summary_used(bool). op_ok None -> excluded; absent keys -> defaults."""
    n = len(results)
    ok = sum(1 for r in results if r.get("answer_ok"))
    op_rows = [r for r in results if r.get("op_ok") is not None]
    op_passed = sum(1 for r in op_rows if r.get("op_ok"))
    lat = [r.get("latency_ms", 0) for r in results]
    tok = [r.get("tokens", 0) for r in results]
    cost = sum(r.get("cost_usd", 0.0) for r in results)
    cache = sum(1 for r in results if r.get("cache_hit"))
    cacheable = sum(1 for r in results if r.get("cache_eligible"))
    ans_cache = sum(1 for r in results if r.get("answer_cache_hit"))
    ans_cacheable = sum(1 for r in results if r.get("answer_cache_eligible"))
    ctx = [r.get("context_tokens", 0) for r in results]
    summary_used = sum(1 for r in results if r.get("summary_used"))
    return RunsSummary(
        accuracy=ok / n if n else 0.0,
        accuracy_ci=wilson(ok, n),
        n=n,
        op_score=(op_passed / len(op_rows)) if op_rows else float("nan"),
        op_ci=wilson(op_passed, len(op_rows)),
        cache_hit_rate=(cache / cacheable) if cacheable else 0.0,
        answer_cache_hit_rate=(ans_cache / ans_cacheable) if ans_cacheable else 0.0,
        context_tokens_mean=(sum(ctx) / n) if n else 0.0,
        summary_used_frac=(summary_used / n) if n else 0.0,
        p50_ms=pctl(lat, 50),
        p95_ms=pctl(lat, 95),
        tokens_mean=(sum(tok) / n) if n else 0.0,
        cost_usd=cost,
    )


def grader_agreement(grades: list[dict], labels: dict[str, dict]) -> dict:
    """kappa table per (check, grader).

    grades: rows {item_no, check, grader, pass_} from a validated run;
    labels: {check: {item_no: expected_bool}}. A grader absent for a check
    simply contributes no row for it."""
    table: dict[tuple[str, str], list[tuple[int, bool]]] = {}
    for g in grades:
        if g.get("pass_") is None or g.get("grader") not in ("code", "jev", "judge"):
            continue
        item = g["item_no"]
        lab = labels.get(g["check"], {}).get(item)
        if lab is not None:
            table.setdefault((g["check"], g["grader"]), []).append(
                (item, bool(g["pass_"]))
            )
    out = {}
    for (check, grader), rows in table.items():
        l = [labels[check][i] for i, _ in rows]
        p = [pred for _, pred in rows]
        out[f"{check}|{grader}"] = kappa(l, p)
    return out
