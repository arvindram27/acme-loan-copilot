"""Seed the Acme policy corpus into mem.doc / mem.chunk.

Idempotent per (title, version): re-running replaces chunks without
duplicating the doc row. First run embeds ~20 chunks — a few tenths of a cent.

    uv run python scripts/seed_kb.py [--reset]
"""

import json
import sys
from pathlib import Path

import psycopg

from agentmem.db import migrate
from agentmem.providers.embed import Embedder
from agentmem.settings import get_settings
from agentmem.units import knowledge

CORPUS_PATH = Path(__file__).resolve().parents[1] / "infra" / "seed" / "corpus.json"


def main() -> int:
    reset = "--reset" in sys.argv
    s = get_settings()
    migrate(s.database_url)
    embedder = Embedder()
    conn = psycopg.connect(s.database_url, autocommit=True)
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))

    if reset:
        n = conn.execute("DELETE FROM mem.doc").rowcount
        print(f"reset: removed {n} docs")

    for doc in corpus["docs"]:
        out = knowledge.add_doc(
            conn,
            title=doc["title"],
            version=corpus["version"],
            text=doc["text"],
            embedding_fn=embedder.embed,
        )
        print(f"{out['chunks']:>3} chunks | {doc['title']}")

    total = conn.execute("SELECT COUNT(*) FROM mem.chunk").fetchone()[0]
    print(f"total chunks in KB: {total}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
