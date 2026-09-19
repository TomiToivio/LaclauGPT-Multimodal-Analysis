# Phase branch workflow

This repository participates in the shared LaclauGPT phase-branch model.

## Persistent branches

| Branch | Purpose |
| --- | --- |
| `phase-0` | Current stable operational baseline. **Active now.** |
| `phase-1` | Phase-1 development and restoration work. May advance without changing Phase 0. |
| `phase-2` | Phase-2 theoretical/methodological/technical work. |
| `phase-3` | Phase-3 preliminary-results/integration work. |
| `phase-4` | Phase-4 final-results/finalization work. |
| `main` | Mirror/integration branch for the **currently active phase**. It is not a catch-all branch for future-phase work. |

As of 2026-09-19, the active phase is **Phase 0**, so `main` and `phase-0` must describe the same current stable project state.

## Issue workflow

For every issue:

1. Determine the intended phase from the issue title/body, labels, milestone, linked roadmap, or explicit user instruction.
2. Start from that persistent phase branch.
3. Create a short-lived implementation branch from it when practical.
4. Open/merge the PR back into the same persistent phase branch.
5. Do not target `main` for Phase-1/2/3/4 work while Phase 0 is active.
6. For Phase-0 work, merge to `phase-0`, validate it, then synchronize `main` with `phase-0`.
7. If the issue is unphased, default to the current active phase. Currently: Phase 0.

Example:

```text
Phase-1 issue
    |
    v
phase-1
    |
    +--> issue-123-some-feature
              |
              +--> PR -> phase-1

main / phase-0 remain stable
```

## Main branch invariant

`main` means "the current released/development phase", not "everything newest anywhere".

Current invariant:

```text
main == current Phase-0 baseline
phase-0 == current Phase-0 baseline
phase-1..phase-4 may contain future work independently
```

When the project moves to Phase 1, the human maintainer explicitly promotes/synchronizes `phase-1` into `main`. The same rule applies for later phases.

Do not automatically promote a future phase because it has newer commits.

## Cross-repository work

For changes spanning multiple LaclauGPT repositories, use the same phase in each participating repository unless the task explicitly defines a cross-phase dependency. A Phase-1 feature should therefore use the `phase-1` branch of each affected repository.

Repository: `TomiToivio/LaclauGPT-Multimodal-Analysis`.

## Stability and backports

If a fix developed in a later phase is also required in the current stable phase, backport/cherry-pick the minimal fix into `phase-0` separately and validate it there. Do not merge the entire later phase into the current phase just to obtain one fix.

If Phase-0 receives a fix after future branches have diverged, forward-port it to affected future phase branches when relevant, without erasing their phase-specific work.

## Agent rule

Agents must read `AGENTS.md` and this file before issue-driven repository changes. Branch selection is part of correctness: code that solves the issue on the wrong phase branch is not considered complete.
