# Speech-to-text (ASR) evaluation for the EP24 preprocess stage

Issue: #5 — *EP24 Phase 2 multimodal analysis pipeline for CSC Roihu*, acceptance
criterion *"Transcription and OCR options have been reevaluated against modern
local alternatives."*

This document covers the **transcription** half only. OCR is a separate step.

Status: **evaluation complete on a GPU workstation; Roihu confirmation pending.**

---

## 1. What the legacy pipeline does

`puhti_preprocess.py` (frozen on `legacy`) loads and calls:

```python
model = whisper.load_model('large', download_root='./whisper/')
...
result = model.transcribe(video_filename, temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
whisper_transcript = result['text']
whisper_language = result['language']
```

Three outputs are persisted — `whisper_transcript`, `whisper_language`, and
`whisper_translated` (English translation, via `deep_translator`, truncated to
3000 characters). These three fields are part of the EP24 output contract.

Two observations about the legacy call, both relevant to the choice below:

1. **The checkpoint is `large`, not `large-v3`.** `whisper.load_model('large')`
   resolves to the `large-v3` weights in current `openai-whisper`, so the
   baseline is already effectively v3 — but it is pinned by *alias*, which means
   a future `openai-whisper` release that re-points `large` would silently
   change EP24 transcripts. Pinning the exact checkpoint is a reproducibility
   improvement, not a method change.
2. **`temperature=[0.0, …, 1.0]` is a fallback ladder, not a fixed temperature.**
   Whisper tries `0.0` first and only escalates on a failed decode. Any
   replacement must honour the same ladder or the fallback behaviour changes.

---

## 2. Candidate engines

All are **local** and require no cloud API, which is mandatory: EP24 material is
GDPR-restricted and the pipeline must not send audio to a third party.
Cloud services (Deepgram, AssemblyAI, Soniox, Azure, Google) were therefore
**excluded on data-protection grounds**, regardless of accuracy.

| Engine | Backend | Licence | Notes |
|---|---|---|---|
| `openai-whisper` | PyTorch | MIT | Legacy default. Reference implementation. |
| `faster-whisper` | CTranslate2 | MIT | Same weights, different runtime. |
| `whisper.cpp` | ggml (C++) | MIT | Strong on CPU/Apple; not needed where a GPU exists. |
| `transformers` Whisper | PyTorch | Apache-2.0 | Heavier; no advantage here. |

**Key point that determines the decision:** `faster-whisper` decodes *the same
published Whisper weights*. CTranslate2 changes the inference engine, not the
model. At a given checkpoint the accuracy difference is a numerical artefact of
quantisation, not a different model. So the engine choice is a
**throughput/memory** decision, and the accuracy decision is **which
checkpoint**.

---

## 3. Measurement

### Method

- **Data:** Google FLEURS `test` split, public academic read-speech corpus.
  2 utterances per language across the **10 EP24 languages** (en, fi, pl, de,
  fr, es, sv, pt, hu, hr). 20 utterances total.
- **No private EP24 material was used**, and no restricted audio left the machine.
- **Settings mirror the legacy call:** the same temperature fallback ladder, no
  conditioning on previous text.
- **Metric:** word error rate (WER) after lowercasing, Unicode NFKC
  normalisation and punctuation stripping — the standard Whisper evaluation
  recipe.
- **Caveat, stated plainly:** FLEURS is *read* speech from a studio. TikTok audio
  is noisy, music-backed and code-switched, so these numbers are **optimistic
  for every engine equally**. They are valid for *ranking engines and
  checkpoints against each other*; they are not a prediction of EP24 WER.

### Result

<!-- BENCHMARK_RESULTS -->

---

## 4. Recommendation

<!-- RECOMMENDATION -->

---

## 5. What is implemented in this branch

- `asr_backend.py` — configurable engine/checkpoint selection.
- `puhti_preprocess.py` — calls the backend; **legacy behaviour is the default**.
- `tests/test_asr_backend.py` — contract tests pinning the default, the
  temperature ladder, loud failure on a bad engine name, and the three legacy
  output fields.

Environment variables:

```bash
LACLAUGPT_ASR_ENGINE   whisper (default) | faster-whisper
LACLAUGPT_ASR_MODEL    large (default) | large-v3 | large-v3-turbo | ...
LACLAUGPT_ASR_DEVICE   cuda (default) | cpu          # faster-whisper only
LACLAUGPT_ASR_COMPUTE_TYPE int8_float16 (default)    # faster-whisper only
LACLAUGPT_ASR_DOWNLOAD_ROOT ./whisper/ (default)     # historical engine only
```

With **no variables set**, the pipeline runs exactly the historical call:
`openai-whisper`, `large`. Switching is explicit and reviewable.

---

## 6. Why this is not a silent method change

Transcription feeds `puhti_summary.py`, `puhti_postprocess.py` and
`puhti_populism.py`. Changing the engine or checkpoint changes transcript text,
which changes downstream analytical output. `AGENTS.md` requires that such
changes be explicit rather than inferred from a "cleaner" implementation.

So this branch does not change the default. It makes the choice *visible and
reproducible*, documents the trade-off with measurements, and leaves the
decision of which checkpoint EP24 should use to the researcher — who is the only
person who can weigh transcript fidelity against Roihu GPU-hours on real data.

---

## 7. Not verified here

- **Roihu (ARM64 / GH200) performance.** Measured on x86-64 + V100. Different
  hardware; CTranslate2 wheel availability on Roihu's ARM environment must be
  checked. `faster-whisper` is a Python package with CTranslate2 wheels; Roihu
  uses ARM, so a source build may be required. **This is the main open risk** and
  it is a `pip install` away from being resolved on the actual platform.
- **EP24 accuracy on real TikTok audio.** Requires restricted data.
- **Diarisation.** Neither engine provides it natively; EP24 transcripts are
  single-speaker TikTok clips, so it is not needed for the current contract.
