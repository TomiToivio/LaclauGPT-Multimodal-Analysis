"""Durable EP24 pipeline state: stage eligibility, claims, and completion (#64).

MongoDB is the canonical store; this module owns the *state machine* that decides
which records a stage may process and records what happened. It is deliberately
storage-agnostic -- it talks to a small protocol that both the Mongo-backed store
and an in-memory store implement -- so the restartability rules can be tested on
CI without a live MongoDB, which the repository's CI cannot provide.

The rules that matter for issue #64:

- a record is eligible for stage N only when stage N-1 is complete for it;
- a claimed record is not processed by anyone else (stale claims can be reclaimed);
- a record is marked complete only after its outputs are durable;
- a successful record is never reprocessed unless explicitly forced;
- a later stage may run while an earlier stage is only part of the way through.

States are the ones the issue names: pending, claimed, complete, skipped, error,
retry.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Protocol


class StageState(str, Enum):
    """Per-record, per-stage processing state."""

    PENDING = "pending"
    CLAIMED = "claimed"
    COMPLETE = "complete"
    SKIPPED = "skipped"
    ERROR = "error"
    RETRY = "retry"


#: States from which a record may be claimed for processing.
CLAIMABLE = frozenset(
    {StageState.PENDING.value, StageState.RETRY.value, StageState.ERROR.value}
)

#: States that mean "do not process again" unless --force is given.
TERMINAL = frozenset({StageState.COMPLETE.value, StageState.SKIPPED.value})


@dataclass
class Record:
    """One record's state for one stage."""

    record_id: str
    stage: int
    state: str = StageState.PENDING.value
    attempts: int = 0
    claimed_at: float | None = None
    claimed_by: str | None = None
    completed_at: float | None = None
    error: str = ""
    provenance: dict = field(default_factory=dict)


class StateStore(Protocol):
    """The storage operations StageStateService needs. Mongo and memory implement it."""

    def load(self, stage: int) -> dict[str, Record]: ...
    def save(self, record: Record) -> None: ...


class MemoryStateStore:
    """In-memory store. For tests and for a dry run without MongoDB."""

    def __init__(self) -> None:
        self._records: dict[tuple[int, str], Record] = {}

    def load(self, stage: int) -> dict[str, Record]:
        return {
            record_id: record
            for (s, record_id), record in self._records.items()
            if s == stage
        }

    def save(self, record: Record) -> None:
        self._records[(record.stage, record.record_id)] = record


@dataclass
class StageStateService:
    """Decide eligibility, claim work, and record outcomes for one stage."""

    stage: int
    store: StateStore
    now: Callable[[], float] = time.time
    claim_ttl_seconds: float = 3600.0
    owner: str = "ep24"

    # -- eligibility --------------------------------------------------------

    def eligible(
        self,
        record_ids: Iterable[str],
        *,
        previous: "StageStateService | None" = None,
        force: bool = False,
    ) -> list[str]:
        """Return the records this stage may process, in the order given.

        A record needs its own stage to be claimable. Stage 1 has no predecessor;
        every later stage also requires the previous stage to be COMPLETE for that
        record, which is what lets Step 2 run over exactly the subset Step 1
        finished.
        """
        states = self.store.load(self.stage)
        prev_states = previous.store.load(previous.stage) if previous is not None else None

        out: list[str] = []
        for record_id in record_ids:
            record = states.get(record_id)
            state = record.state if record is not None else StageState.PENDING.value

            if force:
                out.append(record_id)
                continue
            if state in TERMINAL:
                continue
            if state not in CLAIMABLE and state != StageState.CLAIMED.value:
                continue
            if state == StageState.CLAIMED.value and not self._is_stale(record):
                continue

            if prev_states is not None:
                prev = prev_states.get(record_id)
                if prev is None or prev.state != StageState.COMPLETE.value:
                    continue
            out.append(record_id)
        return out

    # -- claims -------------------------------------------------------------

    def claim(self, record_ids: Iterable[str]) -> list[str]:
        """Mark records claimed by this owner and return the ones actually claimed.

        A record that is already terminal (complete/skipped) is never claimed --
        claiming is what makes work happen, so a completed row must not be
        re-claimed just because a caller passed it in. Use ``eligible()`` to
        select candidates; this method is the enforcement point.
        """
        states = self.store.load(self.stage)
        claimed: list[str] = []
        for record_id in record_ids:
            record = states.get(record_id) or Record(record_id=record_id, stage=self.stage)
            if record.state in TERMINAL:
                continue
            if record.state == StageState.CLAIMED.value and not self._is_stale(record):
                continue
            record.state = StageState.CLAIMED.value
            record.claimed_at = self.now()
            record.claimed_by = self.owner
            self.store.save(record)
            claimed.append(record_id)
        return claimed

    def reclaim_stale(self, *, record_ids: Iterable[str] | None = None) -> list[str]:
        """Return records whose claim is older than the TTL, so a dead worker's
        batch is picked up again instead of being lost."""
        states = self.store.load(self.stage)
        candidates = list(record_ids) if record_ids is not None else list(states)
        stale: list[str] = []
        for record_id in candidates:
            record = states.get(record_id)
            if record is not None and self._is_stale(record):
                stale.append(record_id)
        return stale

    def _is_stale(self, record: Record | None) -> bool:
        if record is None or record.state != StageState.CLAIMED.value:
            return False
        if record.claimed_at is None:
            return True
        return (self.now() - record.claimed_at) > self.claim_ttl_seconds

    # -- outcomes -----------------------------------------------------------

    def complete(self, record_id: str, *, provenance: dict | None = None) -> Record:
        """Mark a record complete. Call only after the outputs are durable."""
        record = self._get(record_id)
        record.state = StageState.COMPLETE.value
        record.completed_at = self.now()
        record.error = ""
        record.claimed_at = None
        record.claimed_by = None
        if provenance:
            record.provenance.update(provenance)
        self.store.save(record)
        return record

    def fail(self, record_id: str, error: str) -> Record:
        """Record a failure. `attempts` drives the retry counter."""
        record = self._get(record_id)
        record.state = StageState.ERROR.value
        record.attempts += 1
        record.error = str(error)
        record.claimed_at = None
        record.claimed_by = None
        self.store.save(record)
        return record

    def retry(self, record_id: str) -> Record:
        """Move an errored record back to retry so it becomes claimable again."""
        record = self._get(record_id)
        record.state = StageState.RETRY.value
        self.store.save(record)
        return record

    def skip(self, record_id: str, reason: str) -> Record:
        """A record this stage deliberately does not process (e.g. no media)."""
        record = self._get(record_id)
        record.state = StageState.SKIPPED.value
        record.error = str(reason)
        self.store.save(record)
        return record

    def _get(self, record_id: str) -> Record:
        return self.store.load(self.stage).get(record_id) or Record(
            record_id=record_id, stage=self.stage
        )

    # -- reporting ----------------------------------------------------------

    def counts(self) -> dict[str, int]:
        """State counts, for the progress reporting the issue asks for."""
        states = self.store.load(self.stage)
        counts = {s.value: 0 for s in StageState}
        for record in states.values():
            counts[record.state] = counts.get(record.state, 0) + 1
        return counts
