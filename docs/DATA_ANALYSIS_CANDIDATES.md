# Candidate improvements from LaclauGPT-Data-Analysis

This file is an evaluation list for the Roihu migration. It is **not** permission
to copy or merge the current LaclauGPT-Data-Analysis pipeline wholesale.

The baseline in this repository remains the historical five-stage EP24
multimodal pipeline, adapted as conservatively as possible.

| Candidate | Current multimodal behavior | Newer Data-Analysis behavior inspected | Potential value here | Compatibility risk | Issue #4 action |
|---|---|---|---|---|---|
| Job-local Ollama | Historical scripts assume Ollama is reachable | Roihu launchers start a job-local server and fail fast | Necessary for reproducible Roihu jobs | low if kept outside analysis code | adopted in public sbatch wrapper |
| Explicit private root | Relative paths assume CWD | EP24/Hungary26 launchers require private runtime roots | Keeps GDPR/research material out of public repo | low | adopted via private working directory, avoiding path rewrite |
| ARM64 Roihu environment | Puhti environment assumptions implicit | Current Roihu launchers require Roihu-built ARM64 venvs | Prevents copying incompatible Puhti x86 environments | low | documented/adopted in wrapper |
| Configurable Gemma4 | Four hard-coded historical models | Current pipelines use configurable Gemma4 models | Allows controlled model comparison | medium because outputs change | adopted only as model-selection variable |
| No cloud fallback | Not explicit in historical scripts | Current LLM routing makes cloud use explicit | Reproducibility/privacy | low | baseline remains local Ollama only |
| Deterministic ffmpeg frames | OpenCV extraction in historical preprocess | Newer reprocessing uses ffmpeg/ffprobe | More reproducible frame extraction | medium/high: changes evidence extraction | **not ported** |
| faster-whisper | Historical `openai-whisper` large model | Some newer workflows use local faster-whisper | ARM/runtime efficiency | medium/high: changes transcript implementation | **not ported** |
| Cache fingerprints/provenance | Per-stage SQLite existence checks | Newer pipeline fingerprints source/model/prompt/config | Better reproducibility and invalidation | medium: changes cache semantics | **not ported** |
| Structured media validation | Frame files are consumed directly | Newer pipeline validates media/checksums and actual image evidence | Stronger QA | medium | **not ported** |
| Richer stage schemas | Historical CSV + SQLite handoff | Newer EP24 runner has richer explicit stage/provenance tables | Better inspectability | high: output/schema compatibility | **not ported** |
| Phase 2 DNA/SNA | Not present | Current Data-Analysis can add DNA/SNA | Future research functionality | very high: methodological expansion | **out of scope** |

## Porting rule

Any candidate marked **not ported** requires a separate narrow task after a real
Roihu + Gemma4 baseline has been run and reviewed.

For each future port:

1. preserve a working baseline commit;
2. state the exact behavior being changed;
3. state why the change is needed;
4. compare output compatibility;
5. validate only that change;
6. do not bundle methodology changes with infrastructure changes.
