"""EP24 active stage scope: Step 0 import, then the normal Steps 1–9 (#380).

Step 8 is evidence-grounded Social Network Analysis and Step 9 is the
CPU-only deterministic RDF export. The older paused-step opt-in is obsolete.
"""
from __future__ import annotations

ACTIVE_STEPS: tuple[int, ...] = tuple(range(10))
PAUSED_STEPS: tuple[int, ...] = ()
# Retained for callers importing the former compatibility symbol only.
INCLUDE_PAUSED_ENV = "EP24_INCLUDE_PAUSED_STEPS"


def paused_included() -> bool:
    """Compatibility helper: all numbered stages are now enabled."""
    return True


def is_paused(step: int) -> bool:
    return False


def default_last_step() -> int:
    return max(ACTIVE_STEPS)


def allows(step: int) -> bool:
    return step in ACTIVE_STEPS


def guard(step: int) -> None:
    if not allows(step):
        raise ValueError(f"Invalid EP24 stage {step}; expected 0..9")


def scope_label(step: int) -> str:
    return "active" if allows(step) else "invalid"
