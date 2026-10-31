"""Which EP24 steps are part of the current run (#346).

Steps 8 (SNA) and 9 (RDF) are **paused**, not failed and not deleted. That distinction has to live in
code, because a dashboard or an audit that cannot tell "paused" from "produced nothing" will keep
reporting a deliberate scope decision as a defect — and someone will eventually "fix" it.

Pausing is a default, not a prohibition: an explicit opt-in still allows a run to include them, which is
what keeps the roadmap alive without scheduling it by accident.
"""
from __future__ import annotations

import os

#: Steps that a normal run submits, in order.
ACTIVE_STEPS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)

#: Steps deliberately excluded from the current run. Code, tests and outputs stay.
PAUSED_STEPS: tuple[int, ...] = (8, 9)

#: Opt-in switch. Absent or "0" means paused steps stay out.
INCLUDE_PAUSED_ENV = "EP24_INCLUDE_PAUSED_STEPS"


def paused_included() -> bool:
    return os.environ.get(INCLUDE_PAUSED_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def is_paused(step: int) -> bool:
    return step in PAUSED_STEPS


def default_last_step() -> int:
    """The last step a normal run may submit."""
    return max(ACTIVE_STEPS)


def allows(step: int) -> bool:
    """May this step be submitted under the current configuration?"""
    if not is_paused(step):
        return True
    return paused_included()


def guard(step: int) -> None:
    """Refuse a paused step unless the operator opted in, with a message that says why."""
    if allows(step):
        return
    raise ValueError(
        f"Step {step} is paused (#346): EP24 runs stop at Step {default_last_step()}. "
        f"Set {INCLUDE_PAUSED_ENV}=1 only for a deliberate experiment."
    )


def scope_label(step: int) -> str:
    """How the UI and audits must describe the step."""
    return "paused" if is_paused(step) else "active"
