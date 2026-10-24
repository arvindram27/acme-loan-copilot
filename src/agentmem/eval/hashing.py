"""Checkout-independent content hashes for evidence artifacts.

Why this exists: `core.autocrlf` and Windows text-mode writes mean the same
committed content can sit on disk with LF or CRLF endings. Hashing the bytes as
read (`Path.read_bytes()`) therefore produces a hash that depends on *how the
file was checked out*, not on what it contains. An artifact recorded on a CRLF
checkout then fails to verify against the same file elsewhere, even though every
byte of content matches.

Scheme `sha256-lf-v1` (precise definition):

* normalise by replacing every occurrence of the two-byte sequence ``\\r\\n``
  with ``\\n``. Nothing else is changed: a lone ``\\r`` is left as-is, a trailing
  newline is not stripped, no decoding or Unicode normalisation happens, and the
  function is safe on arbitrary bytes;
* hash the normalised bytes with SHA-256;
* render as lowercase hexadecimal.

This is a **byte** normalisation, not a canonicalisation. Two JSON documents with
the same data but different key order or indentation still hash differently (that
is what `scripts/collect_evidence.py::_hash`, an object-level canonical hash, is
for). Use this helper for files whose only intended cross-machine difference is
line endings.

Raw-byte hashes remain available and separately named (`sha256_raw` /
`file_hashes()["raw_sha256"]`), because they answer a different question — "are
these the exact bytes this machine saw?" — which is still useful for a local
checkout.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: Bumped only if the normalisation or digest changes. Recorded in artifacts.
HASH_SCHEME = "sha256-lf-v1"


def normalize_line_endings(data: bytes) -> bytes:
    """CRLF -> LF only. Lone CR, trailing newlines and bytes are preserved."""
    return data.replace(b"\r\n", b"\n")


def sha256_raw(data: bytes) -> str:
    """SHA-256 of the bytes exactly as given (checkout-dependent)."""
    return hashlib.sha256(data).hexdigest()


def sha256_normalized(data: bytes) -> str:
    """SHA-256 after CRLF -> LF normalisation (checkout-independent)."""
    return hashlib.sha256(normalize_line_endings(data)).hexdigest()


def file_hashes(path: str | Path) -> dict:
    """Both hashes for a file, under a named scheme.

    Returns ``{"scheme", "raw_sha256", "normalized_sha256"}``. Read as bytes on
    purpose: reading as text would apply Python's universal-newline translation
    and silently hide the difference this helper exists to make explicit.
    """
    data = Path(path).read_bytes()
    return {
        "scheme": HASH_SCHEME,
        "raw_sha256": sha256_raw(data),
        "normalized_sha256": sha256_normalized(data),
    }
