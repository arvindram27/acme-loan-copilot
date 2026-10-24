"""Run-to-code linkage metadata for an evaluation run.

A stored run must answer: which commit, which prompts, which graders, which
dataset version. This builds that record; run_eval writes it into mem.eval_run.
"""

from __future__ import annotations

import hashlib
import json
import subprocess

from agentmem.eval.graders import JEV_QUESTIONS, JUDGE_VALIDATED


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_linkage(settings, workload) -> dict:
    return {
        "commit_hash": _commit(),
        "prompt_version": "prompts-v1",
        "grader_config": {
            "jev_questions": list(JEV_QUESTIONS),
            "judge_validated": sorted(JUDGE_VALIDATED),
            "judge_model": settings.judge_model,
            "jev_model": settings.jev_model,
        },
        "workload_hash": hashlib.sha256(
            json.dumps(workload, sort_keys=True).encode()).hexdigest(),
    }
