"""Scope of the active EP24 numbered pipeline (#380).

All nine analysis/export stages are active by default. Stage 0 is bootstrap.
Older paused-stage environment switches remain harmless compatibility inputs.
"""
from __future__ import annotations

ACTIVE_STEPS: tuple[int, ...] = tuple(range(10))
PAUSED_STEPS: tuple[int, ...] = ()
INCLUDE_PAUSED_ENV = "EP24_INCLUDE_PAUSED_STEPS"


def paused_included() -> bool:
    """All formerly paused stages are now part of normal operation."""
    return True


def is_paused(step: int) -> bool:
    return step in PAUSED_STEPS


def default_last_step() -> int:
    return max(ACTIVE_STEPS)


def allows(step: int) -> bool:
    return step in ACTIVE_STEPS


def guard(step: int) -> None:
    if not allows(step):
        raise ValueError(f"Unknown EP24 step {step}; expected 0..9")


def scope_label(step: int) -> str:
    return "active" if allows(step) else "unknown"
