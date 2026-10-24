"""Test session setup.

`db`-marked tests skip cleanly when Postgres is unreachable, so the suite
errors nowhere on a machine without the database. Markers (`live`, `db`) are
registered in pyproject.toml; `live` is excluded by default there.
"""

import psycopg
import pytest

from agentmem.settings import get_settings


def pytest_collection_modifyitems(config, items):
    try:
        psycopg.connect(get_settings().database_url, connect_timeout=3).close()
        return  # database is reachable: let db tests run
    except Exception:
        pass
    skip = pytest.mark.skip(reason="database not reachable")
    for item in items:
        if item.get_closest_marker("db"):
            item.add_marker(skip)
