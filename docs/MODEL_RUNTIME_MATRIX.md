# Model and runtime selection for the Roihu pipeline

Issue: #144 — *Research optimal Ollama/vLLM model and runtime for each Roihu analysis step.*

Scope: **documentation only.** This record specifies how to choose a model and
runtime per step and the exact procedure that produces the evidence. It does not
change any production model, prompt, runtime or configuration. Choosing a new
default is a separate issue that must cite measured results from the procedure
below.

Status: **procedure and candidates fixed; empirical Roihu results not yet
recorded.** The apples-to-apples Ollama-vs-vLLM run and the blinded multilingual
quality scores need CSC Roihu GPU time and the private EP24 gold sample, so they
are outstanding. This document is the "concrete benchmark procedure and
documented evidence" that #144 accepts in place of an unavailable run.

---

## 1. Evidence boundary

Read every number in this document with its label:

- **Measured (repository):** what the current code actually does — model
  defaults, runtimes, and the fields the Step-3 harness records. Verified by
  reading the files named below.
- **Measured (Roihu):** a real Roihu result. There is currently **one** such
  datapoint: Step 3 has completed a real EP24 video with `Qwen/Qwen3-VL-8B-Instruct`
  under vLLM on GH200 (reported in the #144 discussion). Nothing else here has a
  recorded Roihu measurement.
- **Estimated:** candidate rankings and feasibility from model/runtime
  documentation and architecture. These are hypotheses the procedure below is
  designed to falsify.

No throughput or quality figure below is presented as a Roihu measurement when
it is not one.

---

## 2. What the pipeline does today (measured)

| Step | Runtime actually used | Model default (env override) |
|---|---|---|
| 1 ASR | specialist backend, no LLM | `canary` → `nvidia/canary-1b-v2` (`LACLAUGPT_ASR_ENGINE`) |
| 1 OCR | specialist backend, no LLM | `paddleocr` → `PP-OCRv5` (`LACLAUGPT_OCR_ENGINE`) |
| 2 keyframe VLM | **Ollama** | `qwen3.8:27b` (`LACLAUGPT_MULTIMODAL_MODEL`) |
| 3 whole video | **vLLM** | `Qwen/Qwen3-VL-8B-Instruct` (`LACLAUGPT_VLLM_TEST_MODEL`) |
| 4 summary/fusion | **Ollama** | `qwen3.8:27b` |
| 5 structured postprocess | **Ollama** | `qwen3.8:27b` |
| 6 discourse analysis | **Ollama** | `qwen3.8:27b` |
| 7 DNA extraction | **Ollama** | `qwen3.8:27b` |
| 8 SNA extraction | **Ollama** | `qwen3.8:27b` |
| 9 RDF export | none — deterministic | `model=none` |

Source: `asr_backend.py`, `ocr_backend.py`, `roihu_frame.py`, `roihu_summary.py`,
`roihu_postprocess.py`, `roihu_populism.py`,
`step_7_roihu_discourse_network_analysis.py`,
`step_8_roihu_social_network_analysis.py`, `experiments/vllm_video_test.py`,
`roihu_csv_rdf.py`.

Two facts shape the whole decision:

1. **Step 3 already proves vLLM works on Roihu.** Its harness records
   `vllm_version`, `vllm_torch_version`, `vllm_cuda_version`, `vllm_gpu_name`,
   `vllm_hostname`, `vllm_peak_gpu_memory_mb`, `vllm_video_inference_seconds`,
   `vllm_video_runtime_seconds`, `vllm_structured_output_status` and the prompt
   SHA-256. That is the empirical baseline to reuse, not a synthetic comparison.
2. **Steps 2 and 4–8 share one Ollama model.** `LACLAUGPT_MULTIMODAL_MODEL`
   defaults to `qwen3.8:27b` for all six (issue #156). Whether that one model is
   right for six different tasks (image reasoning, text fusion, JSON extraction,
   theory) is exactly what §7 decides. Historically the shared default was
   `gemma4:12b`; the matrix below reflects the current code.

---

## 3. Roihu GH200 resource model

CSC documents each Roihu GPU node as 4 × NVIDIA GH200, **96 GiB HBM per GPU**,
~120 GiB CPU memory per GPU, on ARM. 384 GiB per node is **not** one flat address
space: it needs model parallelism to use across GPUs.

Consequence — **design the default around one GH200 per worker.** An ~80 GiB
model is not automatically a good 96 GiB deployment once KV cache, activations,
the vision tower, CUDA graphs and multimodal buffers are counted. Use 2–4 GPUs
only when a measured quality gain justifies the allocation and queue cost, and
prefer **four independent one-GPU workers** for corpus throughput over one
4-GPU model-parallel job.

---

## 4. Candidate matrix (estimated)

Fast/balanced/heavy are tiers to test, not a claim that larger wins. The
production choice is the **smallest tier statistically indistinguishable from the
heavier reference** on the frozen sample.

| Step | Task | Fastest viable | Balanced | Max accuracy | Runtime | 1 GH200 |
|---|---|---|---|---|---|---|
| 1 | ASR | Parakeet-TDT-0.6B-v3 | **Canary-1B-v2** | Canary-1B-v2 vs Qwen3-ASR-1.7B (supported langs only) | specialist native | yes |
| 1 | OCR | PP-OCRv5 mobile | **PP-OCRv5 multilingual** | VLM cross-check on hard frames only | PaddleOCR native | yes |
| 2 | keyframe VLM | Qwen3-VL-4B / 8B | **Qwen3-VL-30B-A3B** | larger VL, 2-GPU experiment | **vLLM**, Ollama for dev | yes (≤30B) |
| 3 | whole video | **Qwen3-VL-8B** (proven) | Qwen3-VL-30B-A3B | 2-GPU VL | **vLLM** | yes |
| 4 | fusion/summary | text 4B/8B | **same family as 2, language-only** | 35B-A3B-class text | vLLM batch | yes |
| 5 | entities/topics JSON | **4B** | 8B | 30B-A3B | vLLM structured | yes |
| 6 | discourse analysis | 8B (not assumed enough) | **30B-A3B-class** | 2-GPU / 120B-class text | vLLM batch | balanced yes |
| 7 | DNA JSON | **4B** | 8B | 30B-A3B | vLLM structured | yes |
| 8 | SNA JSON | **4B** | 8B | 30B-A3B | vLLM structured | yes |
| 9 | RDF | none | **none** | none | deterministic Python | no GPU |

Families worth testing per the issue: Qwen3-VL dense/MoE, a Gemma4 comparator
(current default), a GPT-OSS text-reasoning comparator for step 6, and
Mistral/Ministral text. Confirm availability at run time; do not pick by
popularity.

### Step-specific notes

- **Step 1 ASR.** Canary-1B-v2 is the only single backend documented to cover all
  ten EP24 languages (plus translation to English). Qwen3-ASR is faster but its
  documented list omits **Croatian and Bulgarian**, so it cannot be the sole EP24
  backend without a fallback. Parakeet/Whisper stay benchmarks, not defaults.
- **Step 1 OCR.** PP-OCRv5 documents 106 languages including Finnish and
  Bulgarian. The repo already records that EasyOCR cannot cover Finnish in the
  current setup. Keep a general VLM as a hard-frame cross-check, never as the
  primary OCR.
- **Step 4.** Consumes already-extracted text; it does **not** need the vision
  tower. Use language-only mode where the runtime supports it (`--language-model-only`
  on vLLM).
- **Step 9.** Confirmed deterministic (`model=none` in `roihu_csv_rdf.py`). No
  LLM, no GPU. Keep it that way.

---

## 5. Ollama vs vLLM — the question and the method

> Would vLLM be faster and/or more efficient than Ollama for LaclauGPT batch
> processing on CSC Roihu?

**Working hypothesis (to test, not a decision):** for corpus-scale batch work,
vLLM wins on aggregate throughput because of continuous batching, PagedAttention
KV-cache management, chunked prefill, async scheduling, native structured
outputs and native multimodal/video handling; Ollama stays attractive for
interactive work, single rows, small quantized models and operational
simplicity. This is not a universal claim, and there is **no** apples-to-apples
Roihu number yet.

**Method (mandatory to be meaningful).** Compare the **same** model weights and
precision on both runtimes. Comparing Ollama Q4 against vLLM BF16 measures
quantization, not the runtime.

Freeze and record for every run: exact source rows/videos, prompt text and hash,
context, output token cap, sampling settings, image/video frame budget, context
limit, model revision, and code SHA. The Step-3 harness already emits the prompt
hash and version fields needed for this.

Record per run:

- model load time, cold-start latency, warm single-request latency, TTFT,
  output tokens/s, end-to-end rows or videos per minute
- throughput at concurrency 1, 2, 4, 8, then higher only while it still improves
- GPU utilisation, peak HBM, CPU utilisation
- multimodal preprocessing and video-decode time
- OOM/failure rate, JSON-valid rate, wall clock, GPU-seconds per item
- one-GPU vs multi-GPU behaviour, and operational complexity under Slurm

Runtime pairs that answer the question with the least work:

1. **Step 2 image:** Qwen3-VL-8B (or 9B) under both Ollama and vLLM.
2. **Step 3 video:** the existing Qwen3-VL-8B vLLM path is the baseline. Compare
   Ollama only if it accepts an equivalent native video representation;
   otherwise mark the comparison **non-equivalent** rather than feeding stills.
3. **Step 5 structured text:** a 4B/8B model under both runtimes.
4. **Step 6 long text:** a 30B-A3B-class model under both, at equal precision.

vLLM features to exercise: continuous batching, PagedAttention, async scheduling,
chunked prefill, tensor/data/pipeline parallelism, FlashAttention/FlashInfer
backends, structured outputs, multimodal processor caching, native video
handling, and hardware video decode (NVDEC) where supported.

---

## 6. Reproducing the run on Roihu

The harness pattern already exists for #128 and is reused here.

```bash
# Private manifest, frozen and identical for every candidate:
export LACLAUGPT_BENCH_MANIFEST=/scratch/project_2009497/LaclauGPT-Private/.../gold_manifest.csv
export LACLAUGPT_BENCH_OUTPUT=/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess/benchmarks/issue144

# Specialist backends (Step 1) — same harness as #128:
sbatch scripts/roihu/issue128_backend_benchmark.sbatch   # ASR/OCR sweep

# Model-backed steps (2–8) run per step, per country, on the frozen sample.
# The numbered CLI resolves country + limit and chains the checkpoints:
python step_3_roihu_video.py --country finland --limit 25
```

Rules that keep results comparable:

- run on one GH200 (`--gres=gpu:gh200:1`) before trying multi-GPU;
- pin the model revision and precision in the report, never "latest";
- reuse the Step-3 vLLM harness as the baseline so its recorded fields
  (GPU name, peak HBM, inference seconds, structured-output status, prompt hash)
  carry into the comparison;
- keep prompts, sampling and the frame/pixel budget fixed while varying only the
  model or the runtime.

---

## 7. Quality evaluation design

Speed alone does not select a model. Build a frozen gold sample in **private
storage** (not this repository): 30–50 items across the EP24 countries
(FI, PL, PT, DE, FR, ES, HU, HR, BG, SE) and both TikTok and Instagram, including
deliberate hard cases — name variants, caption/transcript disagreement, OCR
noise, code-switching, irony/memes, visually dense content, ambiguous Laclau
frontiers, DNA statements with and without explicit evidence, and SNA
co-occurrence *without* a real tie.

The negative controls matter most: political content that does **not** justify a
populist/frontier reading (step 6), no explicit actor-concept statement (step 7),
co-occurrence with no evidence-supported edge (step 8). These expose
hallucination far better than easy positives.

Metrics per tier:

- **Generic:** wall-clock, GPU-seconds per item, max VRAM, tokens/s, failure/OOM
  rate, schema-validity rate, reproducibility.
- **Step 5/7/8:** JSON validity and exact schema adherence, entity/actor/concept
  precision & recall, stance accuracy, evidence-quote validity, unsupported
  statement and invented-edge rate, multilingual robustness, throughput.
- **Step 6:** blinded human review on theoretical correctness, evidence
  grounding, unsupported inference, missed structure, clarity, cross-language
  consistency. Schema validity alone is not evidence of quality.

---

## 8. Strategy (recommended, pending measurement)

- **Preprocessing** — specialist models, not an LLM: Canary-1B-v2 (ASR),
  PP-OCRv5 (OCR).
- **Perception** — step 2 VLM; step 3 keeps the proven Qwen3-VL-8B vLLM baseline
  and must beat it on video quality before any swap.
- **Text synthesis / theory** — step 4 language-only; step 6 spends the most
  capacity and is the strongest case for a larger model plus human review.
- **Structured extraction** — steps 5/7/8 target the smallest model that passes
  the schema and grounding checks.
- **Deterministic** — step 9 stays model-free.

One model for every stage is unlikely to be optimal: specialised, smaller models
are usually faster, cheaper and more reproducible.

**Precision.** Compare BF16/FP8 first, then 8-bit, then 4-bit only where capacity
or throughput requires it. Do not prefer Q4 merely because it fits. For step 6
explicitly check whether aggressive quantization changes theoretical
distinctions, evidence grounding or multilingual handling.

## 9. One GPU vs many

One GH200 is the default and covers the balanced tiers. Two GPUs are justified
only for a genuinely larger model with multimodal/KV headroom, a high-concurrency
stress test, or proving a quality gain earns the allocation. Do not use 4-GPU
model parallelism as a default merely because the node has four GPUs.

---

## 10. What remains blocked

- Apples-to-apples Ollama-vs-vLLM Roihu numbers (§5) — needs Roihu GPU time.
- Blinded multilingual quality scores (§7) — needs the private gold sample and a
  Roihu run.

Both require CSC credentials held by the researcher, so they cannot be produced
from the repository alone. When they are recorded, they should be added here (or
to a dedicated results file) as **measured** rows, and any model-default change
should be proposed as a separate issue citing them.

## 11. Acceptance-criteria status for this document

- [x] Steps 1–9 audited against the code.
- [x] Fast / balanced / max-accuracy candidates for every model-backed step.
- [x] Step 1 specialist ASR/OCR treated separately.
- [x] Step 9 classified deterministic, no model.
- [x] Concrete benchmark procedure documented in place of a blocked run.
- [x] Step 3 reused as the empirical baseline.
- [x] Single-GPU vs multi-GPU Roihu strategy.
- [x] Multilingual EP24 evaluation design (not assumed from English).
- [x] Structured-output reliability design for steps 5/7/8.
- [x] Theoretical/human evaluation design for step 6.
- [x] Measured facts separated from estimates.
- [x] No implementation or configuration change performed.
- [ ] Measured Ollama-vs-vLLM Roihu numbers — blocked on CSC access.
- [ ] Blinded multilingual quality scores — blocked on the private sample + run.