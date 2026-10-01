# Branch workflow

This repository no longer uses the old Phase 0 synchronization model.

## Current branch contract

- `main` is the active **EP24 Phase 2** implementation for CSC Roihu.
- `legacy` is the immutable historical human-coded EP24 / CSC Puhti implementation.
- Older `phase-0` through `phase-4` branches may remain for historical context, but they do not control current development.

## Rules

1. Current EP24 development starts from `main` unless the human author explicitly requests another branch.
2. Never synchronize `main` back to Phase 0.
3. Never modify, rebase, merge into, or rewrite `legacy`.
4. Keep the legacy branch visible and linked from documentation because publications may rely on it.
5. Phase 2 features from `LaclauGPT-Data-Analysis` are integrated around the legacy-compatible EP24 stages rather than replacing them.
6. Any change that would remove a historical stage, field, prompt, schema, or compatibility output requires explicit human permission.
7. Private codebooks, settings, source data, researcher notes, restricted prompts, and other sensitive material belong in `TomiToivio/LaclauGPT-Private` or private CSC storage.

Repository: `TomiToivio/LaclauGPT-Multimodal-Analysis`.

Agents must read `AGENTS.md` before making changes.
