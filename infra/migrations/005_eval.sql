-- 005_eval.sql — Unit 6: evaluation harness.
-- A run executes a golden workload against one backend. Items record the
-- question, the expected labels, and the actual outcome/trace; grades record
-- every grader's verdict per check, so grader agreement (kappa vs a
-- validated label set) and escalation behavior are inspectable, not claimed.

CREATE TABLE IF NOT EXISTS mem.eval_run (
    run_id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    backend       text NOT NULL DEFAULT 'memory'
                  CHECK (backend IN ('memory', 'baseline', 'langgraph')),
    model         text,
    workload      text NOT NULL,             -- workload file stem
    n_items       int,
    metrics       jsonb,
    jev_cost_usd  numeric,
    started_at    timestamptz NOT NULL DEFAULT now(),
    finished_at   timestamptz
);

CREATE TABLE IF NOT EXISTS mem.eval_item (
    item_id     bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id      bigint NOT NULL REFERENCES mem.eval_run (run_id),
    item_no     int NOT NULL,
    user_id     text NOT NULL,
    thread_id   text NOT NULL,
    question    text NOT NULL,
    expected    jsonb NOT NULL,   -- labels: must_include, must_exclude,
                                  -- required_tool, cite, category
    outcome     jsonb NOT NULL,   -- TurnOutcome dump (the graded trace)
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS eval_item_run_idx
    ON mem.eval_item (run_id, item_no);

CREATE TABLE IF NOT EXISTS mem.grade (
    grade_id    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    item_id     bigint NOT NULL REFERENCES mem.eval_item (item_id),
    grader      text NOT NULL CHECK (grader IN ('code', 'jev', 'judge')),
    check_name  text NOT NULL,
    pass        boolean,           -- NULL for score-type grades
    score       numeric,           -- e.g. answer_quality 0..3
    probability numeric,           -- jev noul
    confidence  numeric,
    escalated   boolean NOT NULL DEFAULT false,
    rationale   text,
    latency_ms  int,
    cost_usd    numeric,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (item_id, grader, check_name)
);
CREATE INDEX IF NOT EXISTS grade_run_join_idx
    ON mem.grade (item_id);
