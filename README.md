# LaclauGPT 

> **Legacy repository:** This repository is preserved for academic research documentation. Active development continues in [LaclauGPT](https://github.com/TomiToivio/LaclauGPT).

LaclauGPT is a political science multimodal data collection and analysis pipeline. It is called LaclauGPT as a tribute to [Ernesto Laclau](https://en.wikipedia.org/wiki/Ernesto_Laclau).

LaclauGPT is developed by [Tomi Toivio](mailto:tomi.toivio@helsinki.fi) for three [Helsinki Hub on Emotions, Populism and Polarisation](https://www.helsinki.fi/en/researchgroups/emotions-populism-and-polarisation) research projects funded by the European Union and the Research Council of Finland:
* [CO3](https://www.co3socialcontract.eu/) researches the social contract. 
* [ENDURE](https://www.endure-project.org/) researches the world after the pandemic. 
* [PLEDGE](https://www.pledgeproject.eu/) researches grievance politics.

The pipeline was used to collect and analyze multimodal social media data related to the 2024 European parliament elections. Data was collected from TikTok and Instagram. Data collection started in 1st of May 2024 and continued until the election day in 9th of June 2024. Collection was based on usernames of official election candidates as well as hashtags and search queries related to the elections. Election data was collected for Bulgaria, Croatia, Finland, France, Germany, Hungary, Portugal, Spain and Sweden. Collected and analyzed data cannot be released yet due to GDPR. This open source version uses dummy data. 

## LaclauGPT Multimodal Data Analysis

These data analysis scripts are published for research documentation. You probably cannot use these without some modification.

These are used with [Ollama](https://ollama.com/) running on [CSC Puhti](https://docs.csc.fi/computing/systems-puhti/) supercomputer.

The scripts are submitted as [batch jobs](https://docs.csc.fi/computing/running/creating-job-scripts-puhti/) in a sequence:

1. puhti_preprocess.py - This extracts video frames with OpenCV, processes the with EasyOCR and extracts a Whisper transcript of the audio.

2. puhti_frame.py - This uses Llama to create a multimodal analysis of 1-6 extracted frames.

3. puhti_summary.py - This creates a Llama summary analysis based on the metadata, Whisper transcript and Llama multimodal analysis results.

4. puhti_postprocess.py - Create structured version of the summary output.

5. puhti_populism.py - Analyze the results using the theories of Laclau and Palonen. 

Code for the [TikTok Scraper](https://github.com/TomiToivio/LaclauGPT-TikTok-Scraper) used to collect EP2024 data is also available.


## AI26 Phase 0 legacy reference

This repository is a **legacy reference implementation** for AI26 Phase 0. Active Phase 0 development belongs in [LaclauGPT-Data-Analysis](https://github.com/TomiToivio/LaclauGPT-Data-Analysis); this repository should remain useful as historical, behavioral, prompt-ordering, and regression reference code rather than being refactored into the current architecture.

### Legacy stages worth porting

1. `puhti_preprocess.py`
   - Extracts up to six keyframes at roughly 30-second intervals.
   - Runs EasyOCR over extracted frames.
   - Runs Whisper transcription and preserves a legacy `whisperResult` compatibility field.
   - Uses SQLite as a cache so expensive preprocessing is not repeated.
   - **Port the behavior:** deterministic media preprocessing, bounded keyframe sampling, transcript/OCR capture, stable cache keys, and compatibility handling.
   - **Do not copy blindly:** hard-coded TikTok paths, language lists, Google translation behavior, and the exact CSV schema are EP2024-specific.

2. `puhti_frame.py`
   - Sends each extracted frame independently to an Ollama vision model.
   - Uses a political-science frame-analysis prompt covering framing, visual elements, activity, color, objects, subjects, and screen-recording indicators.
   - Stores per-frame results in SQLite and writes six ordered frame-analysis fields.
   - **Port the behavior:** one-frame-at-a-time analysis, deterministic low-temperature inference, explicit analytical dimensions, ordered temporal labels, and resumable caching.
   - **Do not copy blindly:** `llama3.2-vision:11b`, six fixed output columns, TikTok-only identifiers, and EP2024 election wording should be configuration rather than architecture.

3. `puhti_summary.py`
   - Combines metadata, transcript, frame descriptions, and OCR into one video-level prompt.
   - Produces a structured political analysis covering narrative, political classification, difficult language, topics, entities, sentiment, populism, social contract, and grievance politics.
   - Caches video-level summaries in SQLite.
   - **Port the behavior:** explicit fusion of modalities into a single analysis context, preservation of provenance by modality, deterministic generation, and stage-level caching.
   - **Do not copy blindly:** EP2024/TikTok metadata field names, fixed prompt categories that exceed Phase 0 scope, or the old vision model choice.

4. `puhti_postprocess.py`
   - Converts free-form summary analysis into structured lists of topics, entities, and sentiment targets using a Pydantic schema.
   - Emits both the canonical CSV and the legacy `ep24_<language>.csv` compatibility artifact used by the next stage.
   - **Port the behavior:** schema-validated structured extraction, normalization/deduplication, and a clean boundary between generative analysis and machine-readable outputs.
   - **Do not copy blindly:** CSV-to-CSV compatibility artifacts should be replaced by the AI26 data contract where possible.

5. `puhti_populism.py`
   - Re-analyzes the video-level summary through Laclau and Palonen.
   - Uses structured output for `populism_analysis`, `populism_us`, and `populism_frontier`.
   - Persists analysis by a stable `video_file` key.
   - **Port the behavior:** a separate discourse-analysis stage downstream of the general summary, explicit theoretical prompts, schema validation, and structured Us/Frontier elements with affects.
   - **Do not copy blindly:** the very long prompt, EP2024 assumptions, current model name, legacy caret/newline serialization, or the old filename contract.

### Stage ordering to preserve conceptually

`preprocess -> frame analysis -> multimodal summary -> structured postprocessing -> Laclau/Palonen discourse analysis`

AI26 can simplify or reorganize storage and interfaces, but this ordering is a useful regression reference because each stage consumes explicit artifacts from the previous one.

### Legacy environment and data assumptions

- Designed for CSC Puhti batch/HPC execution and a locally reachable Ollama service.
- Assumes repository-local runtime directories such as `logs/`, `database/`, `csv/`, `whisper/`, and `Keyframes/`.
- Assumes TikTok EP2024 CSV fields such as `authorUniqueId`, `videoId`, `scrapedCountry`, `videoCreated`, engagement counters, description, nickname, and signature.
- Assumes media under an Allas-derived TikTok path.
- Uses per-stage SQLite caches rather than one shared job/state store.
- Uses fixed language/country lists from the EP2024 collection.
- Uses OpenCV, EasyOCR, Whisper, Ollama, pandas, Pydantic, and in preprocessing a Google translation helper.
- Several scripts execute their full language loops at import/run time, so they should be treated as batch scripts, not reusable library modules.
- Model names, context sizes, and prompt wording are historical implementation details, not Phase 0 requirements.

### Phase 0 rule

Use this repository for **legacy archaeology**: copy semantics and proven stage behavior only where they still fit the AI26 Phase 0 contract. Prefer hand-porting into LaclauGPT-Data-Analysis over importing these scripts wholesale. Keep this repository stable as a reference unless a later issue explicitly requests implementation work here.
