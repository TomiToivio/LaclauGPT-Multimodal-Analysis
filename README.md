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

## Canonical EP24 reprocessing data

Active reprocessing uses the researcher-feed 15-column schema and an additive dataframe through every stage. Scraper-era field names are compatibility-only. See [docs/EP24_CANONICAL_SCHEMA.md](docs/EP24_CANONICAL_SCHEMA.md).

## Restartable MongoDB orchestration

The numbered Roihu steps now have an additive restartable orchestration layer: bootstrap merges researcher entity/theme fields **before Step 1**, MongoDB tracks per-record stage state, Redis provides optional coordination/cache, and CSV + SQLite cumulative checkpoints are written after every durable batch. Countries run Finland -> Poland -> Portugal -> remaining countries. See [docs/EP24_RESTARTABLE_PIPELINE.md](docs/EP24_RESTARTABLE_PIPELINE.md) and [config/ep24_pipeline_columns.json](config/ep24_pipeline_columns.json).

Typical operation:

```bash
python scripts/ep24/bootstrap_ep24_mongodb.py
bash scripts/roihu/run_step_1.sh
bash scripts/roihu/run_step_2.sh
python scripts/ep24/ep24_status.py
```

## Current Roihu pipeline

The active Roihu interface is deliberately numbered because each stage is submitted as a separate Slurm batch job. The canonical demo limit is **100 rows/videos per language** via `LACLAUGPT_MAX_ROWS=100`; set it to `0` for the full corpus where supported.

For the native-video Step 3 path, after Roihu reconnects the tested one-command launcher is:

```bash
source /scratch/project_2009497/LaclauGPT-Multimodal-Analysis/scripts/roihu/activate_vllm_video.sh && roihu_vllm_submit
```

EP24 video handling has a mandatory collection-quality rule: all media analysis excludes the first 1.0 second of every split clip, and whole-video VLM analysis reports additional feed-scroll failures. See [docs/EP24_VIDEO_SCROLL_ARTIFACTS.md](docs/EP24_VIDEO_SCROLL_ARTIFACTS.md).

1. `step_1_roihu_preprocess.py` - ASR/Whisper transcript + translation, OCR, and exactly one keyframe extracted at original source t=1.0s.
2. `step_2_roihu_frame.py` - deep multimodal social-semiotic analysis of exactly that one t=1.0s frame. It concentrates on fine visual detail, rendered text, platform UI, symbols, composition and scene inventory.
3. `step_3_roihu_video.py` - native whole-video Qwen3-VL/vLLM analysis from t=1.0s onward. It supplies temporal narrative, ordered events, scene changes and failed feed-scroll detection.
4. `step_4_roihu_summary.py` - evidence-preserving fusion of the complementary evidence streams: deep one-frame analysis + native-video narrative + Whisper transcript/translation + OCR/source metadata.
5. `step_5_roihu_postprocess.py` - legacy-compatible structured entities/topics/sentiment-target post-processing.
6. `step_6_roihu_discourse_analysis.py` - Laclau/Palonen discourse analysis; canonical new name for the historical `roihu_populism.py`.
7. `step_7_roihu_discourse_network_analysis.py` - Phase 2 DNA statement extraction: actor + concept/proposition + stance/agreement + evidence + uncertainty.
8. `step_8_roihu_social_network_analysis.py` - Phase 2 SNA relation extraction with evidence-supported actor-to-actor edges.
9. `step_9_roihu_rdf.py` - deterministic RDF export after analytical stages. This is CPU-only; it does not need Ollama or a GPU.

Each stage has its own matching batch file under `scripts/roihu/step_N_*.sbatch`. Submit one stage at a time, inspect its CSV/log output, then submit the next. See [docs/ROIHU_NUMBERED_PIPELINE.md](docs/ROIHU_NUMBERED_PIPELINE.md).

The historical `roihu_preprocess.py`, `roihu_frame.py`, `roihu_summary.py`, `roihu_postprocess.py`, and `roihu_populism.py` files remain available as compatibility implementations and must not be deleted merely because numbered entry points exist.

### Video analysis rules

The EP24 clips were split from continuous GrapheneOS screen recordings, so **the first 1.0 second of every clip is the scroll transition from the previous feed item** and is excluded from media analysis. The canonical implementation is `ep24_video.py`, which also normalizes later `SCROLL` / `SCROLL_SECONDS` detections and provides deterministic, provenance-preserving re-split planning.

See [docs/EP24_VIDEO_HANDLING.md](docs/EP24_VIDEO_HANDLING.md) and [docs/EP24_VIDEO_SCROLL_ARTIFACTS.md](docs/EP24_VIDEO_SCROLL_ARTIFACTS.md).

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
