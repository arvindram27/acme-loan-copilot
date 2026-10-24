"""Offline tests for the checkout-independent evidence hash (sha256-lf-v1).

No database, no model calls: pure byte handling plus one read of a committed
evidence file.
"""

import hashlib

from agentmem.eval.hashing import (
    HASH_SCHEME,
    file_hashes,
    normalize_line_endings,
    sha256_normalized,
    sha256_raw,
)
from agentmem.settings import PROJECT_ROOT

CONTENT = b'{"a": 1, "b": "two"}\n'
CONTENT_CRLF = b'{"a": 1, "b": "two"}\r\n'


def test_lf_and_crlf_normalize_to_the_same_hash():
    assert sha256_normalized(CONTENT) == sha256_normalized(CONTENT_CRLF)


def test_normalized_hash_equals_raw_hash_of_the_lf_form():
    assert sha256_normalized(CONTENT_CRLF) == sha256_raw(CONTENT)
    assert sha256_normalized(CONTENT_CRLF) == hashlib.sha256(CONTENT).hexdigest()


def test_a_real_content_change_changes_the_normalized_hash():
    changed = CONTENT.replace(b'"two"', b'"three"')
    assert sha256_normalized(changed) != sha256_normalized(CONTENT)


def test_raw_hashes_still_distinguish_the_two_encodings():
    assert sha256_raw(CONTENT) != sha256_raw(CONTENT_CRLF)


def test_file_hashes_reports_both_hashes_and_the_scheme(tmp_path):
    lf = tmp_path / "lf.json"
    crlf = tmp_path / "crlf.json"
    lf.write_bytes(CONTENT)
    crlf.write_bytes(CONTENT_CRLF)

    a, b = file_hashes(lf), file_hashes(crlf)
    assert a["scheme"] == HASH_SCHEME
    assert b["scheme"] == HASH_SCHEME
    assert a["normalized_sha256"] == b["normalized_sha256"]
    assert a["raw_sha256"] != b["raw_sha256"]
    assert a["raw_sha256"] == sha256_raw(CONTENT)


def test_only_crlf_is_normalized_lone_cr_is_preserved():
    lone = b"line\rbreak"
    assert normalize_line_endings(lone) == lone
    assert sha256_normalized(b"a\rb") != sha256_normalized(b"a\nb")


def test_trailing_newline_is_not_stripped():
    assert sha256_normalized(b"x\n") != sha256_normalized(b"x")


def test_helper_is_binary_safe():
    assert len(sha256_normalized(b"\xff\xfe\r\n\x00")) == 64


def test_scheme_is_versioned():
    assert HASH_SCHEME == "sha256-lf-v1"


def test_historical_v1_artifact_normalized_hash_is_unchanged():
    """The v1 evidence file must never change content.

    Its normalised hash equals the committed LF blob's SHA-256, so this passes on
    an LF or a CRLF checkout; it fails if the artifact is edited.
    """
    h = file_hashes(PROJECT_ROOT / "docs" / "evidence" / "heldout-frozen.json")
    assert h["scheme"] == HASH_SCHEME
    assert h["normalized_sha256"] == (
        "247697e080a2b530edf4eca696226d22a8a5787212d2cb464dd83cc49d2c8ed2"
    )
