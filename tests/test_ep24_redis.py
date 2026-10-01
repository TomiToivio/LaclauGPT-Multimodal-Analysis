"""Redis coordination-layer tests (issue #64).

Redis is coordination/cache only, so the important properties are not "it stores
things" but:

- the module imports and the pipeline runs when Redis is not installed at all;
- an unconfigured or unreachable Redis degrades to "no coordination" rather than
  raising;
- keys are namespaced so EP24 cannot collide with the other projects;
- the single-writer lock releases only a lock this process still holds.

No live Redis is required: the tests use a small fake that records calls, which
is what makes these runnable on CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ep24_redis as er  # noqa: E402


class FakeRedis:
    """Minimal stand-in for the client surface this module uses."""

    def __init__(self, *, fail: bool = False) -> None:
        self.store: dict[str, str] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.calls: list[tuple] = []
        self.fail = fail

    def _maybe_fail(self):
        if self.fail:
            raise RuntimeError("redis down")

    def set(self, name, value, nx=False, ex=None):  # noqa: A002 - mirrors redis-py
        self._maybe_fail()
        self.calls.append(("set", name))
        if nx and name in self.store:
            return False
        self.store[name] = value
        return True

    def get(self, name):
        self._maybe_fail()
        return self.store.get(name)

    def delete(self, name):
        self._maybe_fail()
        self.store.pop(name, None)
        return 1

    def hset(self, name, mapping=None):
        self._maybe_fail()
        self.hashes.setdefault(name, {}).update({k: str(v) for k, v in (mapping or {}).items()})
        return len(mapping or {})

    def hgetall(self, name):
        self._maybe_fail()
        return dict(self.hashes.get(name, {}))

    def hincrby(self, name, key, amount=1):
        self._maybe_fail()
        bucket = self.hashes.setdefault(name, {})
        bucket[key] = str(int(bucket.get(key, 0)) + amount)
        return int(bucket[key])

    def expire(self, name, ttl):
        self.calls.append(("expire", name))
        return True


# --- import safety and graceful degradation --------------------------------

def test_module_imports_without_the_redis_package():
    """The whole point: a machine without the extra must still import."""
    assert hasattr(er, "connect")
    assert er.NAMESPACE == "laclaugpt:ep24"


def test_connect_returns_none_when_no_url_is_configured(monkeypatch):
    monkeypatch.delenv(er.REDIS_URL_ENV, raising=False)
    assert er.connect() is None


def test_no_operation_raises_when_redis_is_disabled():
    """With client=None every call is a safe no-op, not an exception."""
    assert er.set_progress(None, country="Finland", step=1, counts={"complete": 1}) is None
    assert er.get_progress(None, country="Finland", step=1) == {}
    assert er.heartbeat(None, country="Finland", step=1) is False
    assert er.cached_lookup(None, country="Finland", kind="entity", value="x") is None
    er.store_lookup(None, country="Finland", kind="entity", value="x", result="y")
    assert er.bump_retry(None, country="Finland", step=1, record_id="r1") == 0
    assert er.get_retries(None, country="Finland", step=1) == {}


def test_a_failing_redis_degrades_instead_of_raising():
    """A Redis outage must make the run slower, not stop it."""
    bad = FakeRedis(fail=True)
    assert er.get_progress(bad, country="Finland", step=1) == {}
    assert er.heartbeat(bad, country="Finland", step=1) is False
    assert er.cached_lookup(bad, country="Finland", kind="entity", value="x") is None
    assert er.bump_retry(bad, country="Finland", step=1, record_id="r1") == 0
    er.set_progress(bad, country="Finland", step=1, counts={"complete": 1})  # no raise


# --- namespacing -----------------------------------------------------------

def test_keys_are_namespaced_by_project_country_and_step():
    key = er.progress_key("Finland", 3)
    assert key.startswith("laclaugpt:ep24:")
    assert "finland" in key and "step-3" in key


def test_country_and_step_do_not_collide():
    assert er.progress_key("Finland", 1) != er.progress_key("Poland", 1)
    assert er.progress_key("Finland", 1) != er.progress_key("Finland", 2)


def test_key_normalizes_case_and_spaces_so_one_country_is_one_namespace():
    assert er.key("  Finland ", "Step") == er.key("finland", "step")


def test_key_ignores_empty_parts():
    assert "::" not in er.key("Finland", "", None, "progress")


def test_lookup_cache_keys_include_the_kind_and_value():
    a = er.cache_key("Finland", "entity", "Party A")
    b = er.cache_key("Finland", "theme", "Party A")
    assert a != b


# --- single-writer lock ----------------------------------------------------

def test_lock_yields_true_when_redis_is_disabled():
    """No Redis means no coordination, so the durable claim in Mongo is the guard."""
    with er.single_writer(None, country="Finland", step=1) as acquired:
        assert acquired is True


def test_lock_is_exclusive_between_two_holders():
    client = FakeRedis()
    with er.single_writer(client, country="Finland", step=1) as first:
        assert first is True
        with er.single_writer(client, country="Finland", step=1) as second:
            assert second is False, "a second writer must not acquire the same lock"


def test_lock_is_released_for_the_next_holder():
    client = FakeRedis()
    with er.single_writer(client, country="Finland", step=1) as first:
        assert first is True
    with er.single_writer(client, country="Finland", step=1) as second:
        assert second is True


def test_lock_does_not_delete_a_lock_someone_else_now_holds():
    """A lock that expired and was re-taken must not be deleted on our exit.

    Simulated by swapping the stored token while we hold the lock: the release
    path must notice it no longer owns the key.
    """
    client = FakeRedis()
    with er.single_writer(client, country="Finland", step=1) as acquired:
        assert acquired is True
        client.store[er.lock_key("Finland", 1)] = "someone-else-token"
    assert client.store.get(er.lock_key("Finland", 1)) == "someone-else-token"


def test_lock_releases_even_when_the_body_raises():
    client = FakeRedis()
    try:
        with er.single_writer(client, country="Finland", step=1) as acquired:
            assert acquired is True
            raise ValueError("body failed")
    except ValueError:
        pass
    assert er.lock_key("Finland", 1) not in client.store


# --- progress / retries ----------------------------------------------------

def test_progress_round_trips():
    client = FakeRedis()
    er.set_progress(client, country="Finland", step=1, counts={"complete": 7, "error": 2})
    assert er.get_progress(client, country="Finland", step=1) == {"complete": 7, "error": 2}


def test_retry_counter_increments_per_record():
    client = FakeRedis()
    assert er.bump_retry(client, country="Finland", step=1, record_id="r1") == 1
    assert er.bump_retry(client, country="Finland", step=1, record_id="r1") == 2
    assert er.bump_retry(client, country="Finland", step=1, record_id="r2") == 1
    assert er.get_retries(client, country="Finland", step=1) == {"r1": 2, "r2": 1}


def test_lookup_cache_round_trips():
    client = FakeRedis()
    er.store_lookup(client, country="Finland", kind="entity", value="Party A", result="CANON-1")
    assert er.cached_lookup(client, country="Finland", kind="entity", value="Party A") == "CANON-1"
    assert er.cached_lookup(client, country="Finland", kind="entity", value="Party B") is None
