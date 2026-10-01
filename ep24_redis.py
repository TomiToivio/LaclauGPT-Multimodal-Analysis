"""EP24 Redis coordination/cache layer (#64).

Redis is **coordination and cache only** -- step queues, locks, heartbeats, retry
counters, fast progress counters, and lookup caches. MongoDB stays canonical and
SQLite stays the local backup. Nothing durable lives only here, so losing Redis
loses speed and an in-flight claim, not data.

Two consequences of that role shape this module:

- The connection is imported lazily (`redis` is an optional extra, like `pymongo`),
  so the pipeline and its tests import on a machine that has never installed it.
- Every operation is best-effort: a Redis outage must degrade the pipeline to
  "slower" rather than "stopped", because the durable state is elsewhere. Calls
  return a sensible default instead of raising, and log once rather than per call.

Keys are namespaced by project + country + step so EP24 cannot collide with the
other LaclauGPT projects sharing the same Redis deployment.
"""
from __future__ import annotations

import contextlib
import logging
import os
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

#: Namespace prefix. Every key this module writes starts with it.
NAMESPACE = "laclaugpt:ep24"

#: The environment variable the URI is read from. Never a literal, never committed.
REDIS_URL_ENV = "LACLAUGPT_REDIS_URL"


def redis_url() -> str:
    """Read the Redis URL from the environment.

    Deliberately not defaulted: a missing URL should surface as "coordination is
    disabled", not as a connection to somewhere unintended.
    """
    return os.getenv(REDIS_URL_ENV, "").strip()


def connect(*, url: str | None = None, decode_responses: bool = True):
    """Return a Redis client, or None when Redis is not configured.

    Returning None (rather than raising) is deliberate: Redis is optional
    coordination, so an unconfigured or unreachable Redis must not stop a run.
    """
    resolved = (url or redis_url()).strip()
    if not resolved:
        logger.debug("no %s configured; Redis coordination disabled", REDIS_URL_ENV)
        return None
    try:
        import redis  # noqa: PLC0415 - optional extra, imported on demand
    except ImportError:
        logger.warning("redis not installed; coordination disabled (install the 'redis' extra)")
        return None
    try:
        client = redis.Redis.from_url(resolved, decode_responses=decode_responses)
        client.ping()
    except Exception as exc:  # noqa: BLE001 - coordination must never be fatal
        logger.warning("Redis unavailable (%s); continuing without coordination", type(exc).__name__)
        return None
    return client


def key(*parts: Any) -> str:
    """Build a namespaced key: ``laclaugpt:ep24:<country>:<step>:<suffix>``."""
    cleaned = [str(part).strip().lower().replace(" ", "-") for part in parts if str(part).strip()]
    return ":".join([NAMESPACE, *cleaned])


def progress_key(country: str, step: int) -> str:
    return key(country, f"step-{step}", "progress")


def lock_key(country: str, step: int) -> str:
    return key(country, f"step-{step}", "lock")


def heartbeat_key(country: str, step: int) -> str:
    return key(country, f"step-{step}", "heartbeat")


def retry_key(country: str, step: int) -> str:
    return key(country, f"step-{step}", "retries")


def cache_key(country: str, kind: str, value: str) -> str:
    return key(country, "cache", kind, value)


@contextlib.contextmanager
def single_writer(client, *, country: str, step: int, ttl_seconds: int = 3600):
    """A best-effort single-writer lock for one country/step.

    Yields True when this process owns the lock, False when another worker holds
    it. When Redis is unavailable (client is None) it yields True: the durable
    guard is the claim state in MongoDB, and refusing to run because the *cache*
    is down would be the wrong trade.
    """
    if client is None:
        yield True
        return

    name = lock_key(country, step)
    token = uuid.uuid4().hex
    try:
        acquired = bool(client.set(name, token, nx=True, ex=ttl_seconds))
    except Exception as exc:  # noqa: BLE001
        logger.warning("lock acquisition failed (%s); proceeding without it", type(exc).__name__)
        yield True
        return

    if not acquired:
        yield False
        return

    try:
        yield True
    finally:
        # Release only if we still hold it: a lock that expired and was taken by
        # someone else must not be deleted by this process on the way out.
        try:
            if client.get(name) == token:
                client.delete(name)
        except Exception:  # noqa: BLE001
            pass


def set_progress(client, *, country: str, step: int, counts: dict[str, int]) -> None:
    """Publish fast progress counters (the durable numbers live in MongoDB)."""
    if client is None:
        return
    try:
        client.hset(progress_key(country, step), mapping={k: int(v) for k, v in counts.items()})
        client.expire(progress_key(country, step), 86400)
    except Exception as exc:  # noqa: BLE001
        logger.debug("progress publish failed: %s", type(exc).__name__)


def get_progress(client, *, country: str, step: int) -> dict[str, int]:
    if client is None:
        return {}
    try:
        raw = client.hgetall(progress_key(country, step)) or {}
        return {k: int(v) for k, v in raw.items()}
    except Exception:  # noqa: BLE001
        return {}


def heartbeat(client, *, country: str, step: int, ttl_seconds: int = 120) -> bool:
    """Record that this worker is alive. Returns False when Redis is unavailable."""
    if client is None:
        return False
    try:
        client.set(heartbeat_key(country, step), f"{os.getpid()}@{time.time():.0f}", ex=ttl_seconds)
        return True
    except Exception:  # noqa: BLE001
        return False


def cached_lookup(client, *, country: str, kind: str, value: str) -> str | None:
    """Read a cached normalization/lookup result, or None."""
    if client is None:
        return None
    try:
        return client.get(cache_key(country, kind, value))
    except Exception:  # noqa: BLE001
        return None


def store_lookup(client, *, country: str, kind: str, value: str, result: str, ttl_seconds: int = 86400) -> None:
    if client is None:
        return
    try:
        client.set(cache_key(country, kind, value), result, ex=ttl_seconds)
    except Exception:  # noqa: BLE001
        pass


def bump_retry(client, *, country: str, step: int, record_id: str) -> int:
    """Increment a record's retry counter. Returns the new value, or 0 if disabled."""
    if client is None:
        return 0
    try:
        return int(client.hincrby(retry_key(country, step), record_id, 1))
    except Exception:  # noqa: BLE001
        return 0


def get_retries(client, *, country: str, step: int) -> dict[str, int]:
    if client is None:
        return {}
    try:
        raw = client.hgetall(retry_key(country, step)) or {}
        return {k: int(v) for k, v in raw.items()}
    except Exception:  # noqa: BLE001
        return {}
