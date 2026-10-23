"""Metric tests: kappa degenerate cases, Wilson honesty at small n,
percentiles, and the grader-agreement table shape."""

from agentmem.eval.metrics import (
    grader_agreement, kappa, pctl, summarize, wilson,
)


def test_kappa_perfect_and_chance():
    r = kappa([True, False, True, False], [True, False, True, False])
    assert r["kappa"] == 1.0 and r["agreement"] == 1.0
    r = kappa([True, False], [False, True])   # complete disagreement
    assert r["kappa"] == -1.0


def test_kappa_single_class_labels():
    r = kappa([True, True, True], [True, True, True])
    assert r["kappa"] == 1.0
    r = kappa([True, True, True], [True, True, False])
    # 2/3 agreement against a degenerate label set: k = 0 by convention,
    # and the raw agreement is reported alongside (it is NOT hidden)
    assert r["kappa"] == 0.0 and r["agreement"] > 0.66
    assert r["fn"] == 1


def test_kappa_imbalanced_labels_penalizes_minority_misses():
    # labels 9 yes / 1 no; grader says yes to everything
    labels = [True] * 9 + [False]
    r = kappa(labels, [True] * 10)
    assert r["agreement"] == 0.9
    assert r["kappa"] < r["agreement"], "kappa must punish the lazy majority"


def test_wilson_small_n_interval_wide():
    lo, hi = wilson(4, 5)
    assert lo < 0.8 and hi > 0.8
    assert hi - lo > 0.2          # n=5 => honest uncertainty
    lo1, hi1 = wilson(1, 1)
    assert lo1 < 1.0 < hi1 + 1e-9 # single pass cannot claim 100%
    assert wilson(0, 0) == (0.0, 1.0)


def test_pctl_nearest_rank():
    assert pctl([10, 20, 30, 40, 100], 50) == 30
    assert pctl([], 95) == 0.0
    assert pctl([5], 95) == 5


def _rows(n_ok, n_bad):
    rows = []
    for i in range(n_ok):
        rows.append({"answer_ok": True, "op_ok": True, "latency_ms": 100 + i,
                     "tokens": 10, "cost_usd": 0.001,
                     "cache_hit": i % 2 == 0, "cache_eligible": True})
    for i in range(n_bad):
        rows.append({"answer_ok": False, "op_ok": False, "latency_ms": 900,
                     "tokens": 50, "cost_usd": 0.01,
                     "cache_hit": False, "cache_eligible": True})
    return rows


def test_summarize_aggregates():
    s = summarize(_rows(8, 2))
    d = s.as_dict()
    assert abs(d["accuracy"] - 0.8) < 1e-9
    assert d["accuracy_ci"][0] < 0.8 < d["accuracy_ci"][1]
    assert d["op_score"] == 0.8
    assert d["cache_hit_rate"] == 0.4   # 4 hits across 10 eligible items
    assert d["p50_ms"] >= d["n"] == 10
    lo, hi = 100 + 4, 900
    assert d["p95_ms"] == hi, "2/10 items at 900 -> 95th percentile ran into them"


def test_grader_agreement_shape():
    grades = [
        {"item_no": 1, "check": "required_tool_called", "grader": "code",
         "pass_": True},
        {"item_no": 2, "check": "required_tool_called", "grader": "code",
         "pass_": False},
        {"item_no": 1, "check": "answer_grounded", "grader": "jev",
         "pass_": True},
        {"item_no": 2, "check": "answer_grounded", "grader": "jev",
         "pass_": None},        # choice/score row: excluded
        {"item_no": 1, "check": "answer_grounded", "grader": "judge",
         "pass_": True},        # not in labels for judge -> ignored
    ]
    labels = {
        "required_tool_called": {1: True, 2: False},
        "answer_grounded": {1: True, 2: True},
    }
    out = grader_agreement(grades, labels)
    assert "required_tool_called|code" in out
    assert "answer_grounded|jev" in out
    # judge rows ARE included: the judge graded escalated items only, so its
    # kappa is computed on that subset (reported with its n, never hidden)
    assert "answer_grounded|judge" in out
    assert out["answer_grounded|judge"]["n"] == 1
    assert out["required_tool_called|code"]["kappa"] == 1.0
