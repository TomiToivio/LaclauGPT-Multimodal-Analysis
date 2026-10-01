# CSC Roihu / Phase 2 EP24 architecture

## Branch contract

- `legacy` is the frozen historical human-coded EP24 implementation originally run on CSC Puhti.
- `main` is the active **new EP24 analysis** using the **Phase 2 LaclauGPT pipeline** on CSC Roihu.

The legacy branch must remain visible because papers and publications may depend on that exact code.

## Compatibility spine

The historical five-stage logical sequence is retained on `main` with Roihu filenames:

1. `roihu_preprocess.py`
2. `roihu_frame.py`
3. `roihu_summary.py`
4. `roihu_postprocess.py`
5. `roihu_populism.py`

Phase 2 functionality may be inserted before, after, or alongside these stages. Existing legacy steps must not be removed, collapsed, or made unavailable without explicit human permission.

The active Roihu implementation may add newer multimodal models, provenance, RDF, DNA, SNA, richer structured outputs, and other compatible improvements from `LaclauGPT-Data-Analysis`.

## Public/private contract

Public repository:

`TomiToivio/LaclauGPT-Multimodal-Analysis`

Private repository:

`TomiToivio/LaclauGPT-Private`

Set a private runtime root, for example:

```bash
export LACLAUGPT_MULTIMODAL_PRIVATE_ROOT=/path/to/LaclauGPT-Private/analysis/ep24-multimodal
```

Private codebooks, settings, source data, researcher notes, restricted prompts, mappings, credentials and other sensitive material remain in `LaclauGPT-Private` and/or private CSC storage. They are not copied into this public repository.

## Roihu runtime

The public batch template targets CSC Roihu. Roihu GPU nodes use ARM/aarch64 CPUs, so environments must be created for Roihu rather than copied from the historical Puhti environment.

Site allocation IDs and absolute private paths are intentionally not committed.

## Models

The model must be configurable. Current defaults may use `gemma4:12b`, but the active Phase 2 implementation should test newer/better suitable local multimodal models when available within Roihu constraints.

No cloud fallback should be introduced implicitly for private research workloads.

## Submit

```bash
export CSC_ACCOUNT='<project>'
export LACLAUGPT_MULTIMODAL_PRIVATE_ROOT='/private/path'
sbatch --account="$CSC_ACCOUNT" scripts/roihu/multimodal_roihu.sbatch
```

The compatibility stages remain:

`preprocess frame summary postprocess populism`

## Compatibility requirements

Main must preserve the ability to reproduce legacy-compatible products needed by existing EP24 research, including historical field meanings, stage contracts and compatibility outputs.

Phase 2 additions should be additive. If an internal implementation changes, provide a compatibility adapter rather than deleting the old contract.

The frozen `legacy` branch remains the authority for exact historical reproduction.

## Phase 2 extensions

Relevant additions from `LaclauGPT-Data-Analysis` include, where appropriate:

- improved Roihu-local multimodal inference;
- richer provenance and resumability;
- stronger evidence validation;
- RDF graph outputs;
- Discourse Network Analysis;
- Social Network Analysis;
- improved structured schemas;
- other compatible Phase 2 modules.

See `AGENTS.md` for mandatory constraints.
