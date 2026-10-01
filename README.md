# LaclauGPT Multimodal Analysis

<!-- project-logos:start -->
<p align="center">
  <a href="https://www.co3socialcontract.eu/"><img src="https://raw.githubusercontent.com/TomiToivio/LaclauGPT/main/assets/co3-logo.svg" width="180" alt="CO3 project logo"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.pledgeproject.eu/"><img src="https://www.pledgeproject.eu/wp-content/uploads/2024/04/Pledge-Logo.png" height="88" alt="PLEDGE project logo"></a>
</p>
<p align="center">
  <small>CO3 and PLEDGE: European Union funding</small><br>
  <a href="https://european-union.europa.eu/principles-countries-history/symbols/european-flag_en"><img src="https://www.pledgeproject.eu/wp-content/uploads/2024/04/co-funded-by-european-union.png" height="54" alt="European Union funding acknowledgement for CO3 and PLEDGE"></a>
</p>
<p align="center">
  <a href="https://www.endure-project.org/"><img src="https://www.endure-project.org/_inhaltselemente/logo-kurz.png?width=500" height="54" alt="ENDURE project logo"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.aka.fi/"><img src="https://www.aka.fi/globalassets/aka_fi_vaaka_sininen.svg" height="44" alt="Research Council of Finland (Suomen Akatemia) logo"></a>
</p>
<p align="center"><small>ENDURE: University of Helsinki research funded by the Research Council of Finland</small></p>
<!-- project-logos:end -->

> **Repository status:** `main` is the active **new EP24 analysis** for CSC Roihu, built with the **Phase 2 LaclauGPT pipeline** while preserving full compatibility with the original human-coded EP24 workflow. The historical implementation remains permanently visible on the [`legacy` branch](https://github.com/TomiToivio/LaclauGPT-Multimodal-Analysis/tree/legacy) because papers and publications rely on that exact version.

This repository is the EP24 multimodal analysis implementation of the current LaclauGPT research pipeline. It combines the original five-stage human-written EP24 workflow with compatible Phase 2 improvements from [LaclauGPT-Data-Analysis](https://github.com/TomiToivio/LaclauGPT-Data-Analysis), including newer model support and, where applicable, richer Phase 2 analysis layers such as RDF, Discourse Network Analysis (DNA), and Social Network Analysis (SNA).

The compatibility rule is strict: **new functionality is added around or alongside the legacy stages. Existing legacy steps, fields, prompts, schemas, and outputs must not be removed or replaced without explicit human permission.**

## Legacy research record

The original EP24 implementation used CSC Puhti. That exact historical code is preserved on the [`legacy` branch](https://github.com/TomiToivio/LaclauGPT-Multimodal-Analysis/tree/legacy). Keep it visible and citable. Do not modernize or rewrite that branch.

The `main` branch is different: it is the active Roihu/Phase 2 implementation. References to Puhti in `main` should normally describe the historical legacy environment only.

## Research context

LaclauGPT is a political science multimodal data collection and analysis pipeline named as a tribute to [Ernesto Laclau](https://en.wikipedia.org/wiki/Ernesto_Laclau).

LaclauGPT is developed by [Tomi Toivio](mailto:tomi.toivio@helsinki.fi) for research at the University of Helsinki, including CO3, ENDURE and PLEDGE.

The EP24 data covers multimodal social-media material related to the 2024 European Parliament elections. Collection included TikTok and Instagram material from multiple European countries. Real research data cannot be published here. Public code may contain safe examples or dummy fixtures only.

## Current Roihu pipeline

The compatibility spine remains the historical five-stage sequence, now named for CSC Roihu:

1. `roihu_preprocess.py` - extract video frames, OCR text, and audio transcripts.
2. `roihu_frame.py` - multimodal frame analysis.
3. `roihu_summary.py` - summary analysis from metadata, transcript, and multimodal evidence.
4. `roihu_postprocess.py` - structured post-processing of summary output.
5. `roihu_populism.py` - Laclau/Palonen analysis.

Phase 2 functionality from LaclauGPT-Data-Analysis may be attached before, after, or alongside these stages, provided legacy-compatible inputs and outputs remain available.

### Video analysis rules

The EP24 clips were split from continuous GrapheneOS screen recordings, so **the first 1.0 second of every clip is the scroll transition from the previous feed item** and is excluded from every analysis path (transcription, frames, OCR, VLM, embeddings, summaries). The rule is defined once in `video_config.py`. Scroll-detection failures are reported with `SCROLL` / `SCROLL_SECONDS` and feed a bounded re-splitting step.

See [docs/EP24_VIDEO_HANDLING.md](docs/EP24_VIDEO_HANDLING.md).

## Public/private boundary

This repository is public open source. **Do not store real codebooks, private settings, researcher notes, restricted prompts, credentials, source data, or other sensitive research material here.**

Those belong in [TomiToivio/LaclauGPT-Private](https://github.com/TomiToivio/LaclauGPT-Private) and/or private CSC project storage. Public code should consume private configuration through explicit paths, environment variables, or interfaces without copying private content into this repository.

See [AGENTS.md](AGENTS.md) for mandatory development rules and [docs/ROIHU_MIGRATION.md](docs/ROIHU_MIGRATION.md) for the current architecture.


## Memory, codebooks and distributed research storage

The Roihu enrichment layer is documented in [docs/MEMORY_CODEBOOKS.md](docs/MEMORY_CODEBOOKS.md). Operational codebooks, researcher notes, database credentials and real research material remain in [TomiToivio/LaclauGPT-Private](https://github.com/TomiToivio/LaclauGPT-Private) and/or private CSC storage.

For EP2024 reprocessing, the normal distributed data plane is:

- **MongoDB** for durable dataframe mirrors/results, codebooks, memory, RAG, researcher notes, embeddings, graph/RDF/DNA/SNA material, provenance and backup-oriented state;
- **Redis** for transient coordination, cache, messaging, locks and run/worker status;
- **CSC Allas** for source videos, downloaded on demand from their stored URL/object identifier after `allas_conf` setup;
- **Pandas CSV/DataFrames** as the mandatory human-readable input/output and backward-compatible legacy contract;
- **SQLite/DuckDB** only as optional local/job-local helpers or compatibility artifacts.

PostgreSQL is not part of the EP2024 reprocess architecture.

Country-specific MongoDB collections use `laclaugpt_ep2024_reprocess_<country_name>_<collection_name>`, for example `laclaugpt_ep2024_reprocess_finland_memory`, `..._rag`, `..._research_notes`, `..._rdf` and `..._dataframe`.

Every major analysis step must preserve machine-readable structured output **and** add a human-readable Markdown-formatted summary field to the CSV. Existing legacy columns are retained unchanged and new fields are additive.

See [docs/MONGODB_STORAGE.md](docs/MONGODB_STORAGE.md) for the canonical storage contract.
