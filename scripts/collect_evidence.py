"""Assemble sanitised, metadata-stamped evidence artifacts.

Reads eval_validation.json (grader validation) and the latest mem.eval_run per
backend, then writes docs/evidence/*.json carrying the metadata a reader needs
to trace a number to the code that produced it: commit, model IDs, workload
hash, denominators, CIs, category breakdown, cache state, repeat count.

    python scripts/collect_evidence.py

Everything here is synthetic/demo data: no user PII, no documents, no keys.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import psycopg

from agentmem.eval.workload import WORKLOAD_V1
from agentmem.settings import get_settings


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def main() -> int:
    s = get_settings()
    out = Path("docs/evidence")
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit": _git_commit(),
        "chat_model": s.chat_model,
        "judge_model": s.judge_model,
        "jev_model": s.jev_model,
        "workload": "workload_v1",
        "workload_sha256": _hash(WORKLOAD_V1),
        "prompt_version": "prompts-v1",
        "repeat_count": 1,
        "cache_state": "cold (each backend's demo users are reset before a run)",
        "authority_rule": ("per check: highest kappa >= 0.6 wins; ties to the "
                           "cheaper grader; below 0.6 stays unvalidated"),
        "judge_note": ("the endpoint is direct OpenAI, so the judge is "
                       "same-family (gpt-4o vs gpt-4o-mini); this is NOT a "
                       "cross-family comparison"),
    }

    val_path = Path("eval_validation.json")
    if val_path.exists():
        val = json.loads(val_path.read_text(encoding="utf-8"))
        doc = {"_meta": meta, "validation_items": len(val.get("rows", [])),
               **val}
        (out / "grader-validation.json").write_text(
            json.dumps(doc, indent=2), encoding="utf-8")
        print("wrote docs/evidence/grader-validation.json")

    c = psycopg.connect(s.database_url, autocommit=True)
    runs = []
    for backend in ("memory", "langgraph", "baseline"):
        r = c.execute(
            "SELECT run_id, model, workload, n_items, metrics, jev_cost_usd, "
            " finished_at FROM mem.eval_run WHERE backend = %s "
            "ORDER BY run_id DESC LIMIT 1", (backend,),
        ).fetchone()
        if r:
            runs.append({
                "backend": backend, "run_id": r[0], "model": r[1],
                "workload": r[2], "n_items": r[3], "metrics": r[4],
                "jev_cost_usd": float(r[5] or 0), "finished_at": str(r[6]),
            })
    c.close()
    (out / "workload-run.json").write_text(
        json.dumps({"_meta": meta, "runs": runs}, indent=2), encoding="utf-8")
    print("wrote docs/evidence/workload-run.json")
    for run in runs:
        m = run["metrics"] or {}
        print(f"  {run['backend']:<9} n={run['n_items']} "
              f"acc={m.get('accuracy')} ci={m.get('accuracy_ci')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
