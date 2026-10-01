# Historical migration record: Puhti → Roihu

This document records the earlier migration inventory. It is retained for provenance only.

The repository has since advanced beyond the original migration plan:

- `legacy` is now the immutable historical EP24 / CSC Puhti implementation.
- `main` is now the active **EP24 Phase 2 LaclauGPT** implementation for CSC Roihu.
- Active stage files on `main` use the `roihu_*.py` naming convention.
- The old Phase 0 branch-synchronization policy is obsolete for this repository.

## Historical compatibility principle

The original human-coded five-stage EP24 pipeline remains the compatibility spine:

1. preprocess
2. frame
3. summary
4. postprocess
5. populism

Phase 2 functionality from `LaclauGPT-Data-Analysis` is added around or alongside that spine. Legacy stages and contracts are not removed without explicit human permission.

For exact historical source code used by earlier papers and publications, use the [`legacy` branch](https://github.com/TomiToivio/LaclauGPT-Multimodal-Analysis/tree/legacy).

## Private research material

Real EP24 data, codebooks, settings, mappings, researcher notes, restricted prompts, credentials and other sensitive material belong in `TomiToivio/LaclauGPT-Private` and/or private CSC storage, not in this public repository.

For the current architecture, see `docs/ROIHU_MIGRATION.md` and `AGENTS.md`.
