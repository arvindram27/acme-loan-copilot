# Evaluation

The evaluation harness grades every answer against constructed labels and reports
grader agreement (Cohen's κ), so the numbers are inspectable, not claimed.

## The workload

`WORKLOAD_V1` (32 turns, two users, seven scenarios) exercises one memory promise per
scenario:

| scenario | memory promise |
|---|---|
| supersession | a newer fact value replaces the older one, everywhere |
| cross-thread | a fact written in thread A is visible in thread B |
| tool | required-tool discipline and provenanced numbers |
| kb grounding | citations `[title §chunk]`, policy numbers only from retrieved text |
| scope | an intruder user never sees another user's facts |
| recall | paraphrase/compaction of earlier turns |
| cache | a repeated question is served with ~0 tokens |

## Grader cascade

- **Code** grades what code can prove deterministically: `required_tool_called`,
  `tool_args_valid`, `number_provenance`, `must_include`, `must_exclude`,
  `citation_present`.
- **Jev** (TypeSafe System One) grades judgment checks against the trace as a structured
  state in one batched call: `answer_grounded`, `forbidden_absent`, `failure_mode`,
  `answer_quality`.
- **Judge** (a configurable model, `MEM_JUDGE_MODEL`; it defaults to the chat model, so
  it is only "cross-family" when you point it at a different family) is the tie-breaker
  for low-confidence Jev answers — and is **gated off**, because with a fixed prompt and a
  fair (all-items) measurement it still never beat Jev on any check. Any cross-family
  claim must name both model IDs in a committed run artifact.

## Authority (pre-declared rule)

For each check, the authoritative grader is the one with the highest κ ≥ 0.6; ties break
to the cheaper grader. Measured on 96 constructed-label items:

| check | grader | κ / agreement |
|---|---|---|
| required_tool_called, tool_args_valid, number_provenance, must_include, must_exclude, citation_present | code | **1.0** |
| forbidden_absent | jev | **1.0** |
| failure_mode | jev | 0.83 exact |
| answer_grounded | jev | 0.79 |
| answer_quality | jev | 0.68 exact / **0.99 ±1** (soft signal) |
| (any) | judge | < Jev — disabled |

`answer_grounded` over-flags a few entity-recall items where the model re-formats a
number (`8,750` vs `8750`) — a documented limitation, still above the 0.6 bar.

## Cost

Jev pricing is anchored to real billing (not the per-billion-vs-per-million unit slip the
first version had): a full 96-item validation run costs **~$0.004**. The chat path stays
OpenAI (one variable at a time); Jev never sits in the runtime.

## Held-out protocol

The 32-item workload was used to find and fix failures, so it cannot by itself
show that a fix generalises. The held-out check follows a freeze-first protocol:

1. **Freeze (F1).** `scripts/heldout_frozen.py` contains the questions, the
   expected answers and the scoring rule — and is committed with **no results**
   (`b92f0b1`). Numeric scoring reuses the shared comparator; there is no
   substring matching.
2. **Run once.** Immediately after F1, once, with the result captured as-is.
3. **Publish (F2).** The artifact keeps `frozen_commit` (F1's hash), the script's
   SHA-256, the models, the timestamp and every answer — **including failures**.
4. **No retuning.** v1 is never edited. If a failure exposes a product bug, the
   bug is fixed separately and any re-evaluation is a **new frozen set (v2)**,
   clearly labelled.

### Question sets and scorers are versioned separately

A question set and a scorer answer different questions, so they version
independently:

| version | what it fixes | where |
|---|---|---|
| `heldout-frozen-v1` | the 5 questions, expected answers, seed message | `scripts/heldout_frozen.py`, `docs/evidence/heldout-frozen.json` |
| `heldout-scorer-v1` | the scoring rule used for the v1 result | frozen inside the v1 script |
| `heldout-scorer-v2` | h3 additionally requires the seven-day period; other questions unchanged | `src/agentmem/eval/heldout_scorer.py` |

Revising the scorer does **not** create a new question set and does **not**
rewrite the v1 result. Two derived artifacts are produced, and they are labelled
differently:

- **Regrade** (`scripts/regrade_heldout.py` → `docs/evidence/heldout-regrade-v2.json`):
  the *saved* v1 answers re-scored under v2. No model calls. It shows only that
  the recorded answers survive a stricter rule.
- **Regression run** (`scripts/heldout_regression.py` →
  `docs/evidence/heldout-regression-v2.json`): the same five **known** questions
  executed once under v2 with the same model configuration. This is a regression
  check, **not** an unseen holdout — the questions are neither new nor hidden.

v1 scored h3 leniently: it asked for `0.125` and a citation, not the period, so
"0.125 points per 30 days" would have passed. v2 adds the duration condition and
is covered by local tests that reject a missing or wrong period even when the
number and citation are present.

Result of v1: 5/5 (`docs/evidence/heldout-frozen.json`), unchanged.

Under scorer v2, reported separately and never merged into v1:

| artifact | what it is | result |
|---|---|---|
| `heldout-frozen.json` | v1 run, scorer v1 (frozen) | 5/5, author-reported |
| `heldout-regrade-v2.json` | saved v1 answers re-scored, no model calls | 5/5 (v1 5/5 → v2 5/5) |
| `heldout-regression-v2.json` | the same 5 known questions, one run, scorer v2 | 5/5 |

The regression run used the same configuration as v1 (`gpt-4o-mini` +
`text-embedding-3-small`, checked before running), took 25 chat calls and 18,011
tokens, and cost an estimated $0.003 (chat only, gpt-4o-mini list price;
embeddings excluded). No failure occurred, so — as with v1 — the
retain-failures rule was again not exercised. The run was executed once; it was
not repeated until it passed.

### Hash scheme (`sha256-lf-v1`)

Hashing the bytes as read from disk is checkout-dependent: this repository uses
`core.autocrlf=true`, and Python's text-mode writes (`Path.write_text`) emit CRLF
on Windows. The same committed content can therefore be LF in git and CRLF on
disk, and a raw SHA-256 differs between them.

The v2 regrade artifact is an instance: its `source_artifact_sha256` is
`db389a25…`, which is the SHA-256 of the **CRLF working-copy bytes** of
`heldout-frozen.json`. The committed content is `247697e0…` (the LF blob). The
answers in both copies are identical, so the two hashes describe the same file;
the hash was simply not portable.

`src/agentmem/eval/hashing.py` defines a versioned content hash:

- **`sha256-lf-v1`** — SHA-256, lowercase hex, over the bytes after replacing
  every `\r\n` with `\n`. A lone `\r` is left untouched, trailing newlines are
  not stripped, bytes are never decoded, and nothing else is normalised. It is a
  byte normalisation, not a canonicalisation: key order and whitespace still
  change the hash.
- The **raw-byte** hash is kept separately as `raw_sha256`. It answers a
  different question ("are these the exact bytes on this machine?") and is
  expected to vary with the checkout.

Historical artifacts are **not** rewritten and their fields keep their original
meaning: in `heldout-regrade-v2.json`, `source_artifact_sha256` is the raw-byte
hash, and in `heldout-regression-v2.json`, `questions_source_sha256` is the same
(raw) hash of `scripts/heldout_frozen.py`. Future artifacts from the regrade and
regression tooling carry a self-describing object instead:

```json
"source_artifact_hash": {
  "scheme": "sha256-lf-v1",
  "raw_sha256": "...",
  "normalized_sha256": "..."
}
```

The old flat field names are retired rather than reinterpreted. To verify a
historical artifact across checkouts, compare `normalized_sha256` against the git
blob's SHA-256:

| file | raw (this CRLF checkout) | normalised (= git blob) |
|---|---|---|
| `docs/evidence/heldout-frozen.json` | `db389a25…` | `247697e0…` |
| `docs/evidence/heldout-regrade-v2.json` | `499af86b…` | `875062d9…` |
| `docs/evidence/heldout-regression-v2.json` | `a3332723…` | `570ccf60…` |
| `scripts/heldout_frozen.py` | `c83d4cfd…` | `c83d4cfd…` |

Offline tests (`tests/test_evidence_hash.py`) pin the scheme: LF and CRLF forms of
identical content hash the same, a content change hashes differently, raw hashes
still distinguish the encodings, and the frozen v1 artifact's normalised hash is
unchanged.

### What the freeze does and does not establish

Be precise about the claim:

- **Established:** the questions and scoring were committed before the *recorded*
  first run, and the artifact records F1's hash and the run timestamp. The order
  of the recorded run is supported by commit history.
- **Not established:** that there were no earlier runs or tuning. Nothing in the
  repo can show that; `worktree_clean_at_run` is self-reported by the script, and
  an informal local run would leave no trace.
- **Not established:** that the model outputs are authentic beyond the committed
  evidence. The 5/5 is **author-reported**; it was not independently reproduced.
- **Independence and coverage are different things.** These questions were
  written by the author, who knew which fixes they were aimed at. An externally
  authored set would improve *independence*; it would not by itself establish
  *coverage* of real loan-officer traffic. Representative sampling and broader
  scenarios are needed for that.
- **n = 5, one recorded run, no repeats.** That cannot support a confidence
  interval or a real-world success rate, and none is claimed.

Frozen set v1 (5 author-written questions, committed before the recorded first
run; result author-reported): **5/5**. The commit order is recorded; it does not
prove no earlier runs or tuning, independence from the author, or coverage of
real traffic.

## Artifacts

`docs/evidence/grader-validation.json` and `docs/evidence/workload-run.json`
are generated by `scripts/collect_evidence.py` and carry the commit hash, model
IDs (chat `gpt-4o-mini`, judge `gpt-4o`, Jev `jev-latest`), the workload hash,
denominators, Wilson CIs and the per-category breakdown. They contain only
synthetic/demo data — no user PII, documents, prompts or keys.

The judge is **same-family** here because the endpoint is direct OpenAI; a
cross-family claim would require a different provider and a fresh artifact that
names both model IDs.

## Running it

```powershell
uv run python scripts/run_eval.py --backend memory,langgraph,baseline   # workload -> mem.eval_*
uv run python scripts/validate_graders.py --json eval_validation.json   # grader kappa table
uv run python scripts/collect_evidence.py                               # metadata-stamped artifacts
```
