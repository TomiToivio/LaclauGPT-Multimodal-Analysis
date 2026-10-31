"""Which EP24 stages a normal run covers, and which are paused on purpose (#346).

Scope reduction, not deletion: Steps 8 (SNA) and 9 (RDF) keep their code, tests and stored outputs, but a
normal FI/PL/PT run ends at Step 7. The distinction has to survive into the artifacts, or a paused stage
looks exactly like one that failed — and the audit would report a defect that does not exist.

The launcher exports ``LACLAUGPT_PAUSED_STAGES=8,9``; the exporter records that in the manifest; the audit
reads it and stops calling those steps anomalous. Absence of the key means "this export predates the
scope statement" and is treated as unknown, never as "nothing is paused".
"""
from __future__ import annotations

import os

__all__ = [
    "NORMAL_LAST_STEP",
    "PAUSED_STEPS",
    "is_paused",
    "normal_steps",
    "paused_stages",
]

#: The terminal stage of a normal EP24 run. Step 7 (DNA) is where the active pipeline stops.
NORMAL_LAST_STEP = 7

#: Paused stages: preserved, tested, opt-in only.
PAUSED_STEPS: tuple[int, ...] = (8, 9)

ENV_VAR = "LACLAUGPT_PAUSED_STAGES"


def paused_stages(environ: dict[str, str] | None = None) -> tuple[int, ...]:
    """Paused stages from the environment, defaulting to SNA and RDF.

    An explicit empty value (``LACLAUGPT_PAUSED_STAGES=""``) means "nothing is paused" — that is how a
    deliberate reactivation is expressed, rather than by editing this module.
    """
    raw = (os.environ if environ is None else environ).get(ENV_VAR)
    if raw is None:
        return PAUSED_STEPS
    return tuple(int(part) for part in str(raw).replace(" ", "").split(",") if part.strip())


def is_paused(step: int, environ: dict[str, str] | None = None) -> bool:
    return int(step) in paused_stages(environ)


def normal_steps(environ: dict[str, str] | None = None) -> tuple[int, ...]:
    """Stages a normal run submits: 0-7 minus whatever is paused explicitly."""
    paused = set(paused_stages(environ))
    return tuple(step for step in range(0, NORMAL_LAST_STEP + 1) if step not in paused)
