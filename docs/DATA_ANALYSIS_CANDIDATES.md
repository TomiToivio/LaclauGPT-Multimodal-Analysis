# Phase 2 integration from LaclauGPT-Data-Analysis

This repository is the active **EP24 Phase 2** analysis implementation for CSC Roihu.

The original human-coded EP24 workflow remains the compatibility spine. New functionality from [LaclauGPT-Data-Analysis](https://github.com/TomiToivio/LaclauGPT-Data-Analysis) should be integrated **around, before, after, or alongside** the legacy-compatible stages without removing their behavior or outputs.

| Phase 2 feature | EP24 integration direction | Compatibility rule |
|---|---|---|
| Job-local Ollama | Use Roihu-local inference and fail-fast model checks | Keep legacy-compatible stage inputs/outputs |
| Configurable newer models | Prefer the best suitable local models available on Roihu, with Gemma4 as fallback/baseline where appropriate | Model changes must be explicit and provenance-recorded |
| Explicit private root | Resolve sensitive runtime material from `LaclauGPT-Private` / CSC storage | Never commit real private material |
| Provenance and resumability | Add fingerprints, model/config provenance, stage status and safe resumption | Do not make legacy outputs unavailable |
| Deterministic media handling | Improve frame/audio handling where validated | Preserve a compatibility path for legacy semantics |
| Structured validation | Validate media evidence, schemas and stage products | Add checks rather than silently changing research meaning |
| RDF graph layer | Produce graph-compatible semantic outputs around the core pipeline | Additive output |
| Discourse Network Analysis (DNA) | Add when source data and codebooks support it | Additive analytical layer |
| Social Network Analysis (SNA) | Add when actor/interaction data supports it | Additive analytical layer |
| Richer structured outputs | Add Phase 2 schemas and provenance tables | Retain legacy-compatible fields/files needed by existing research |

## Integration rule

For each Phase 2 addition:

1. identify the legacy-compatible stage(s) it surrounds or extends;
2. preserve the historical inputs, outputs and interpretation contract;
3. add adapters when new internal representations differ;
4. validate both the new output and the legacy-compatible output;
5. document model/config provenance;
6. keep private codebooks, settings, mappings, prompts, researcher notes and data in `TomiToivio/LaclauGPT-Private`.

The `legacy` branch remains the frozen source for exact historical reproduction.
