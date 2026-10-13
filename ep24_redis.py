"""Optional Redis coordination for EP24. MongoDB remains the durable source of truth."""
from __future__ import annotations

import os
from contextlib import contextmanager


class RedisCoordinator:
    def __init__(self, country: str, step: int):
        self.country = country
        self.step = step
        self.client = None
        url = os.getenv("LACLAUGPT_REDIS_URL")
        if url:
            try:
                import redis
            except ImportError as exc:
                raise RuntimeError("LACLAUGPT_REDIS_URL is set but redis package is not installed") from exc
            self.client = redis.Redis.from_url(url, decode_responses=True)

    @property
    def prefix(self) -> str:
        return f"laclaugpt:ep2024_reprocess:{self.country}:step_{self.step:02d}"

    def mark(self, record_id: str, status: str) -> None:
        if self.client:
            self.client.hset(f"{self.prefix}:status", record_id, status)

    @contextmanager
    def lock(self, record_id: str, timeout: int = 3600):
        if not self.client:
            yield True
            return
        lock = self.client.lock(f"{self.prefix}:lock:{record_id}", timeout=timeout, blocking_timeout=1)
        acquired = lock.acquire(blocking=True)
        try:
            yield acquired
        finally:
            if acquired:
                try:
                    lock.release()
                except Exception:
                    pass
