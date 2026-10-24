"""Robustness: bounded provider calls and concurrent fact corrections."""

import random
import threading

import psycopg
import pytest

from agentmem.db import migrate
from agentmem.providers.chat import MAX_RETRIES as CHAT_RETRIES
from agentmem.providers.chat import ChatClient
from agentmem.providers.embed import Embedder
from agentmem.settings import Settings, get_settings
from agentmem.units import entity


def test_chat_client_sets_timeout_and_bounded_retries():
    c = ChatClient(settings=Settings(api_key="dummy", _env_file=None))
    assert c._client.timeout is not None
    assert c._client.max_retries == CHAT_RETRIES


def test_embed_client_sets_timeout_and_bounded_retries():
    e = Embedder(settings=Settings(api_key="dummy", _env_file=None))
    assert e._client.timeout is not None
    assert e._client.max_retries == 2


@pytest.fixture()
def dsn():
    migrate(get_settings().database_url)
    return get_settings().database_url


def test_concurrent_corrections_leave_one_active_value(dsn):
    u = f"cc-{random.randint(10**8, 10**9 - 1)}"
    seed = psycopg.connect(dsn, autocommit=True)
    entity.write_fact(seed, user_id=u, thread_id="t", kind="applicant",
                      name=u, attr="income_monthly", value=1000)
    c1 = psycopg.connect(dsn, autocommit=True)
    c2 = psycopg.connect(dsn, autocommit=True)
    errors: list[Exception] = []

    def correct(conn, value):
        try:
            entity.write_fact(conn, user_id=u, thread_id="t",
                              kind="applicant", name=u,
                              attr="income_monthly", value=value)
        except Exception as exc:  # noqa: BLE001 - recorded, asserted below
            errors.append(exc)

    t1 = threading.Thread(target=correct, args=(c1, 2000))
    t2 = threading.Thread(target=correct, args=(c2, 3000))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    active = seed.execute(
        "SELECT value FROM mem.fact WHERE user_id = %s "
        "AND attr = 'income_monthly' AND status = 'active'", (u,),
    ).fetchall()
    try:
        assert not errors, f"writers should serialize, got {errors}"
        assert len(active) == 1, f"exactly one current value, got {active}"
        assert active[0][0] in (2000, 3000)
    finally:
        entity.purge_user(seed, user_id=u)  # fact_audit stays (immutable)
        seed.close()
        c1.close()
        c2.close()


pytestmark = pytest.mark.db
