"""Regrade the *saved* answers from frozen v1 with scorer v2. No model calls.

This is a **regrade**, not a run and not a holdout: it reads the answers already
committed in `docs/evidence/heldout-frozen.json` and applies the stricter scorer
to them. It answers one question only — do the recorded v1 answers still pass
when h3 also requires the seven-day period?

The original artifact is never modified and the original result is never
replaced; this writes `docs/evidence/heldout-regrade-v2.json` alongside it.

    python scripts/regrade_heldout.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from agentmem.eval.hashing import file_hashes
from agentmem.eval.heldout_scorer import (
    SCORER_V2,
    Answer,
    load_frozen,
    score,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "evidence" / "heldout-frozen.json"
OUT = ROOT / "docs" / "evidence" / "heldout-regrade-v2.json"
OUT_TXT = ROOT / "docs" / "evidence" / "heldout-regrade-v2.txt"


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return ""


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}", file=sys.stderr)
        return 2

    frozen = json.loads(SRC.read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in load_frozen().QUESTIONS}

    rows, lines = [], []
    header = f"{'id':<16} {'v1':<4} {'v2':<4} {'v2 checks'}"
    print(header)
    lines.append(header)

    for saved in frozen["questions"]:
        q = questions[saved["id"]]
        verdict = score(
            q,
            Answer(
                text=saved["answer"],
                retrieved_refs=tuple(saved.get("retrieved_refs") or ()),
                tool_calls=tuple(saved.get("tool_calls") or ()),
            ),
            version=SCORER_V2,
        )
        changed = verdict.passed != saved["pass"]
        rows.append({
            "id": saved["id"],
            "question": saved["question"],
            "answer": saved["answer"],
            "v1_pass": saved["pass"],
            "v2": verdict.to_dict(),
            "changed": changed,
        })
        note = "" if not changed else "  <-- CHANGED"
        line = (f"{saved['id']:<16} {'PASS' if saved['pass'] else 'FAIL':<4} "
                f"{'PASS' if verdict.passed else 'FAIL':<4} "
                f"duration_ok={verdict.checks.get('duration_ok', 'n/a')}{note}")
        print(line)
        lines.append(line)

    v1_n = sum(1 for r in rows if r["v1_pass"])
    v2_n = sum(1 for r in rows if r["v2"]["pass"])
    summary = f"\nv1 {v1_n}/{len(rows)}  ->  v2 {v2_n}/{len(rows)}"
    print(summary)
    lines.append(summary)

    payload = {
        "_meta": {
            "kind": "regrade",
            "label": (
                "Saved answers from frozen v1 re-scored with scorer v2. "
                "NOT a new run, NOT an unseen holdout, and NOT a replacement "
                "for the v1 result."
            ),
            "question_set": frozen["_meta"].get("set"),
            "question_set_frozen_commit": frozen["_meta"].get("frozen_commit"),
            "source_artifact": "docs/evidence/heldout-frozen.json",
            # Both hashes, named explicitly: raw bytes vary with the checkout's
            # line endings; the normalised hash is the stable content identity.
            # The pre-2026-10-24 artifacts used a flat `source_artifact_sha256`
            # that meant the raw-byte hash; it is not reused or reinterpreted.
            "source_artifact_hash": file_hashes(SRC),
            "scorer_version": SCORER_V2,
            "model_calls": 0,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "commit": _git("rev-parse", "--short", "HEAD"),
            "scoring_change": (
                "h3-rag-lock now also requires a seven-day period in the answer; "
                "every other question keeps the frozen v1 rule."
            ),
        },
        "summary": {
            "total": len(rows),
            "v1_passed": v1_n,
            "v2_passed": v2_n,
            "changed": [r["id"] for r in rows if r["changed"]],
        },
        "questions": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"wrote {OUT_TXT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
