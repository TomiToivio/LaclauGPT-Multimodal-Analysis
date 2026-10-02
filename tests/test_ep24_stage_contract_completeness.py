#!/usr/bin/env python3
"""The stage contract must not silently under-declare a stage (#132, #129).

The defect this file pins
-------------------------
`ep24_stage_contract.py` declared stage 3 (video) as:

    _s(3, "video", "ep24_video.py", (), notes="... Appends no dataframe columns by design.")

Both halves were wrong on green `main`:

1. **The module is not an entry point.** `ep24_video.py` is the shared
   skip/trim rules library — `analysis_start_seconds`, `build_trim_command`,
   `split_plan`, `derived_rows` — with no `main` and no `__main__`. The real stage
   entry is `step_3_roihu_video.py`, which wraps
   `experiments/vllm_video_test.py::main`. Anything resolving stage 3 through the
   contract got a library it could not execute.
2. **"Appends no dataframe columns" was false.** The stage appends 46 columns
   (`vllm_video_test.OUTPUT_COLUMNS`). The contract said zero.

Why nothing caught it
---------------------
`tests/test_ep24_stage_contract.py` checks the direction *declared -> written*
and skips stages whose `appends` is empty:

    for stage in contract.STAGE_CONTRACT:
        if not stage.appends:
            continue

So an empty declaration was exempt from the only check that could notice, and no
test looked in the other direction (*written -> declared*) at all. A stage could
shrink or grow silently as long as it declared nothing.

These tests close both directions and remove the skip.

Public, synthetic fixtures only.

Run:  python -m pytest tests/test_ep24_stage_contract_completeness.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ep24_stage_contract as contract  # noqa: E402


def test_stage_3_declares_the_columns_the_video_stage_really_writes() -> None:
    """The exact #132 defect: stage 3 declared zero columns and writes 46."""
    stage = contract.stage(3)
    assert stage.appends, (
        "stage 3 declares no columns; it appends vllm_video_* fields at runtime, "
        "so an empty declaration understates the schema"
    )

    import experiments.vllm_video_test as video  # noqa: E402

    real = set(video.OUTPUT_COLUMNS)
    declared = set(stage.appends)
    missing = real - declared
    assert not missing, (
        "columns the video stage writes but the contract does not declare: "
        f"{sorted(missing)}"
    )


def test_stage_3_points_at_an_executable_entry_point() -> None:
    """A contracted module must be runnable, not a support library.

    The contract exists so orchestration can locate each stage's entry point.
    Pointing it at `ep24_video.py` (a rules library with no `main`) meant the
    contract named a module that cannot be executed as that stage.
    """
    stage = contract.stage(3)
    module_path = ROOT / stage.module
    assert module_path.is_file(), f"contracted module missing: {stage.module}"
    source = module_path.read_text(encoding="utf-8")
    assert "__main__" in source, (
        f"stage 3 module {stage.module!r} has no __main__ block; it is not an "
        "executable stage entry point"
    )


def test_the_wrapper_is_what_the_contract_credits() -> None:
    """Explicitly pin the corrected value so a revert reads clearly."""
    assert contract.stage(3).module == "step_3_roihu_video.py"


@pytest.mark.parametrize("number", [s.number for s in contract.STAGE_CONTRACT])
def test_every_stage_with_a_module_file_can_be_resolved(number: int) -> None:
    """Sanity: every contracted module path exists (or is documented as absent)."""
    stage = contract.stage(number)
    path = ROOT / stage.module
    assert path.is_file(), f"stage {number} credits a missing module: {stage.module}"


def test_no_stage_is_exempt_from_the_append_check_by_declaring_nothing() -> None:
    """The blindspot: `if not stage.appends: continue` exempted stage 3.

    Two stages legitimately append nothing (7 discourse_network, 9 rdf) because
    they write elsewhere. That must be an explicit, justified state — not a
    silent default that also covers a stage which really does append columns.

    This test records which stages are allowed to be empty, so a NEW empty stage
    has to be added here deliberately rather than slipping through the skip.
    """
    #: Stages whose emptiness is intended and documented in their `notes`.
    empty_by_design = {7, 9}
    empty = {s.number for s in contract.STAGE_CONTRACT if not s.appends}
    unexpected = empty - empty_by_design
    assert not unexpected, (
        f"stage(s) {sorted(unexpected)} declare no appended columns; if that is "
        "intended, add them to empty_by_design with a reason, otherwise the "
        "contract understates their schema"
    )
    for number in empty_by_design & empty:
        stage = contract.stage(number)
        assert stage.notes, f"stage {number} is empty by design but carries no note"


def test_stage_3_is_not_treated_as_empty_by_design() -> None:
    """The specific regression: stage 3 must never rejoin the empty set."""
    assert contract.stage(3).number not in {7, 9}
    assert contract.stage(3).appends


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
