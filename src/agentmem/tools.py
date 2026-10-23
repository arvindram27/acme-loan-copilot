"""The Acme tool set: deterministic tools that keep arithmetic in code.

Every fn(args: dict, conn) -> dict returns a JSON-safe dict. Policy
constants here must stay in sync with the knowledge-base corpus (the corpus
is the prose version; these are the numbers the tools act on).
"""

RATE_SHEET = {
    # term_years -> lock_days -> (rate_pct, points)
    15: {30: (5.750, 0.0), 45: (5.750, 0.0), 60: (5.625, 0.25), 90: (5.500, 0.25)},
    30: {30: (6.375, 0.0), 45: (6.375, 0.0), 60: (6.250, 0.25), 90: (6.125, 0.5)},
}

# occupancy -> units bucket -> max LTV (mirrors the LTV corpus doc)
MAX_LTV = {
    "owner_occupied": {1: 95, 2: 85},
    "investment": {1: 80, 2: 75},
}

MAX_DTI_DEFAULT = 45.0


def _num(args: dict, key: str, required: bool = True, default=None):
    v = args.get(key, default)
    if v is None:
        if required:
            raise ValueError(f"missing required argument: {key}")
        return None
    return float(v)


def affordability_calculator(args: dict, conn=None) -> dict:
    income = _num(args, "income_monthly")
    debts = _num(args, "monthly_debts", required=False, default=0.0)
    # Tool-input validation: the planner may pass max_dti_pct: 0 (it did,
    # once) — a poisoned percentage must not produce an authoritative-looking
    # negative payment. Anything outside (0, 100] falls back to the default,
    # and the result says so.
    dti_raw = args.get("max_dti_pct")
    dti_cap = float(dti_raw) if dti_raw is not None else None
    note = None
    if dti_cap is None or not 0 < dti_cap <= 100:
        if dti_cap is not None:
            note = (f"max_dti_pct={dti_cap} out of range (0, 100]; "
                    f"used default {MAX_DTI_DEFAULT}")
        dti_cap = MAX_DTI_DEFAULT
    if income <= 0:
        raise ValueError("income_monthly must be positive")
    max_payment = round(income * (dti_cap / 100) - max(debts, 0), 2)
    result = {
        "max_qualifying_payment": max_payment,
        "dti_cap_pct": dti_cap,
        "assumed_debts_monthly": round(max(debts, 0), 2),
    }
    if note:
        result["constraint_used"] = note
    return result


def rate_lookup(args: dict, conn=None) -> dict:
    term = int(_num(args, "term_years", default=30))
    lock = int(_num(args, "lock_days", default=30))
    sheet = RATE_SHEET.get(term)
    if sheet is None:
        raise ValueError(f"unsupported term: {term} years")
    if lock not in sheet:
        raise ValueError(f"unsupported lock length: {lock} days")
    rate, points = sheet[lock]
    return {
        "term_years": term,
        "lock_days": lock,
        "rate_pct": rate,
        "points": points,
        "note": "rate is indicative; lock is only effective after confirmation email",
    }


def ltv_check(args: dict, conn=None) -> dict:
    price = _num(args, "purchase_price")
    loan = _num(args, "loan_amount")
    occupancy = args.get("occupancy", "owner_occupied")
    units = int(_num(args, "units", default=1))
    if price <= 0 or loan <= 0:
        raise ValueError("purchase_price and loan_amount must be positive")
    occupancy_key = occupancy if occupancy in MAX_LTV else "owner_occupied"
    bucket = 2 if units >= 2 else 1
    max_ltv = MAX_LTV[occupancy_key][bucket]
    ltv = round(loan / price * 100, 2)
    return {
        "ltv_pct": ltv,
        "max_ltv_pct": max_ltv,
        "exceeds": ltv > max_ltv,
        "occupancy_bucket": f"{occupancy_key}/{bucket}-unit",
    }


def policy_full_dump(args: dict, conn) -> dict:
    """The whole corpus text. Deliberately large — exercises the offload."""
    rows = conn.execute(
        """
        SELECT d.title, c.chunk_index, c.text
        FROM mem.chunk c JOIN mem.doc d ON d.doc_id = c.doc_id
        ORDER BY d.doc_id, c.chunk_index
        """
    ).fetchall()
    body = "\n\n".join(f"[{t} §{i}] {x}" for t, i, x in rows)
    return {"documents": len({r[0] for r in rows}),
            "chunks": len(rows), "full_text": body}


TOOLS = {
    "affordability_calculator": {
        "fn": affordability_calculator,
        "description": (
            "Compute the maximum qualifying monthly payment from monthly "
            "income and monthly debts under Acme's DTI limit. Use for any "
            "affordability or payment question — never compute in your head."
        ),
        "params": {"income_monthly": "number", "monthly_debts": "number",
                   "max_dti_pct": "number"},
        "ttl": 0,
    },
    "rate_lookup": {
        "fn": rate_lookup,
        "description": (
            "Look up the current conforming rate-sheet rate for a term "
            "(15 or 30 years) and lock length (30/45/60/90 days), with "
            "points. Use instead of quoting a rate from memory."
        ),
        "params": {"term_years": "15|30", "lock_days": "30|45|60|90"},
        "ttl": 900,
    },
    "ltv_check": {
        "fn": ltv_check,
        "description": (
            "Check a loan amount against Acme's maximum LTV for occupancy "
            "(owner_occupied|investment) and unit count. Use before "
            "committing to a price or loan amount."
        ),
        "params": {"purchase_price": "number", "loan_amount": "number",
                   "occupancy": "owner_occupied|investment", "units": "1|2"},
        "ttl": 0,
    },
    "policy_full_dump": {
        "fn": policy_full_dump,
        "description": (
            "Return the ENTIRE underwriting corpus text (very large). Only "
            "when the user explicitly asks for the complete guide."
        ),
        "params": {},
        "ttl": 600,
    },
}
