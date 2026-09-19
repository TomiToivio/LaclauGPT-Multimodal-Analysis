# Phase branch workflow

This repository uses persistent `phase-0` through `phase-4` branches.

As of 2026-09-19, **Phase 0 is active**, so `main` and `phase-0` must represent the same current stable baseline. Future-phase branches may advance independently.

## Rules

1. Determine an issue's phase from its title/body, labels, milestone, linked roadmap, or explicit instruction.
2. Start work from the matching `phase-N` branch.
3. Prefer an issue branch from that phase branch and PR back to the same `phase-N`.
4. Do not target `main` with Phase-1/2/3/4 work while Phase 0 is active.
5. Phase-0 work lands in `phase-0`, is validated there, then `main` is synchronized.
6. Unphased issues default to the active phase, currently Phase 0.
7. For cross-repository work, use the same phase branch in every affected LaclauGPT repo unless explicitly documented otherwise.
8. Backport minimal fixes between phases when required; never merge an entire future phase into the stable phase just to obtain one fix.

`main` means the current active phase, not the globally newest code.

Current invariant:

```text
main == phase-0 stable baseline
phase-1..phase-4 = isolated future work
```

When the project advances to a later phase, promotion into `main` requires explicit human approval.

Repository: `TomiToivio/LaclauGPT-Multimodal-Analysis`.

Agents must read `AGENTS.md` and this file before issue-driven changes. Working on the wrong phase branch is an incorrect implementation.
