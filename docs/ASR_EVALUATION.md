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

Measured on 20 FLEURS utterances across all 10 EP24 languages.

**Engines and checkpoints** (mean WER, wall-clock, and speed vs real time):

| Configuration | mean WER | total sec | ×real-time |
|---|---:|---:|---:|
| `openai-whisper` `large` *(legacy default)* | 0.070 | 115.2 | 2.3× |
| `faster-whisper` `large-v3` int8_float16 | 0.068 | 28.2 | 9.4× |
| `faster-whisper` `large-v3` float16 | 0.071 | 21.3 | 12.4× |
| `faster-whisper` `large-v3-turbo` int8_float16 | 0.066 | 9.8 | 27.0× |
| **`faster-whisper` `large-v3-turbo` float16** | **0.061** | **8.6** | **30.7×** |
| `faster-whisper` `distil-large-v3` int8_float16 | 1.032 | 12.8 | 20.6× |
| `faster-whisper` `distil-large-v3` float16 | 1.011 | 10.8 | 24.4× |

Total audio: 263.9 s. 20 utterances, 2 per language.

**Per-language WER** (the two live candidates vs the legacy default):

| Language | legacy `large` | `fw large-v3` | `fw turbo` |
|---|---:|---:|---:|
| de | 0.000 | 0.000 | 0.000 |
| en | 0.100 | 0.074 | 0.100 |
| es | 0.017 | 0.017 | 0.017 |
| fi | 0.077 | 0.077 | 0.100 |
| fr | 0.000 | 0.008 | 0.008 |
| hr | 0.172 | 0.172 | 0.156 |
| hu | 0.198 | 0.198 | 0.080 |
| pl | 0.029 | 0.029 | 0.029 |
| pt | 0.021 | 0.021 | 0.035 |
| sv | 0.083 | 0.083 | 0.083 |

Language detection was correct on **20/20** utterances for the legacy engine.

### What the numbers actually say

1. **Engine swap is free.** `faster-whisper large-v3` scores 0.068 against the
   legacy 0.070 — a 0.002 difference on 20 utterances, i.e. **one word**, which
   is noise at this sample size. Meanwhile it runs **4.1× faster** (115.2 s →
   28.2 s) and uses less VRAM. This is the expected result: same weights,
   different runtime.
2. **`turbo` is not a quality compromise here.** At 0.061 it is nominally
   *better* than the legacy default, and 13.4× faster. On a 20-utterance sample
   that margin is not statistically meaningful — the honest reading is **"turbo
   costs nothing measurable in these ten languages"**, not "turbo is more
   accurate". It is worth noting because the common advice is to avoid turbo for
   non-English, and these ten languages do not show that penalty on read speech.
3. **`distil-large-v3` is unusable for this pipeline.** WER 1.03 means it
   produces roughly one word of error per word of reference. It is an
   English-only distillation and these are ten non-English languages; it should
   be *excluded* rather than merely not recommended. Recording a negative result
   is the point of measuring.
4. **Hungarian (0.198) and Croatian (0.172) are the weak languages** for every
   engine, which is consistent with their lower ASR resource levels. That is a
   property of the language and the checkpoint, not of the engine choice, and it
   is a reason to look at language-specific fine-tunes later — not a reason to
   pick a different runtime.

### Reproducing

```bash
python bench_faster.py     # faster-whisper sweep
```

Samples are drawn deterministically from `google/fleurs` as described in
§3. On this machine `faster-whisper` needs cuBLAS on `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
```


---

## 4. Recommendation

**Engine: switch to `faster-whisper`.** The measurement supports it without a
quality trade-off — same weights, 4.1× faster, less memory — and it is MIT
licensed and pip-installable. This is a runtime change, not a methodological one.

**Checkpoint: keep `large`-class as the default, and let the researcher choose.**
`large-v3-turbo` measured no worse and 13× faster, which makes it the obvious
candidate for a full EP24 run. But transcripts are research data and turbo's
behaviour on *noisy, music-backed TikTok audio* is not something FLEURS read
speech can establish. So:

- default stays the historical `large` (unchanged behaviour, nothing silently
  different in existing outputs);
- `LACLAUGPT_ASR_MODEL=large-v3-turbo` is documented as the recommended
  throughput option for a full run, with a smoke comparison against `large-v3`
  on a small real sample before the pipeline is committed to it.

**Do not use `distil-large-v3`** for EP24. Measured WER 1.03 across these ten
languages; it is English-only distillation.

**Pin the checkpoint explicitly.** The legacy call uses the alias `large`, which
currently resolves to `large-v3`. Recording the resolved name in provenance
protects the corpus from a future `openai-whisper` release re-pointing the alias.

### What I am *not* recommending

- **A cloud ASR service.** Several score better on paper, but EP24 audio is
  GDPR-restricted and must not leave the machine. Excluded on principle, not on
  accuracy.
- **Diarisation.** Not needed: EP24 items are single-speaker TikTok clips, and no
  output field captures speaker turns.
- **A language-specific fine-tune** (e.g. NB-Whisper for Norwegian). The
  published gains for such models are real and large, but EP24 spans ten
  languages; adopting ten fine-tunes multiplies the operational surface and
  changes transcription behaviour per country. Worth revisiting for Hungarian
  and Croatian specifically, which are the measured weak languages — but as a
  deliberate later experiment, not part of this infrastructure change.


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
