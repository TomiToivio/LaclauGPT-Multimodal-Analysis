# Laskin vLLM video experiment (issue #32)

> **Security status (2026-10-02): retired for execution.** The historical Volta-compatible stack used legacy Transformers/PyTorch versions that now have published security advisories. The installable Laskin requirements manifest has been removed, and both Laskin launcher scripts fail closed. Keep this document as a reproducibility record only; use CSC Roihu for active native-video analysis.

Measured feasibility result for running the same native-video vLLM path on
**Laskin** as issue #24 runs on **CSC Roihu**. Everything below was measured on
`laskin01`; nothing is assumed.

## Answer

**vLLM native whole-video inference works on Laskin — but not with the reference
model.** The machine's V100s can run it; the reference `Qwen/Qwen3-VL-8B-Instruct`
cannot be loaded there at all.

## 1. Laskin environment (measured)

```text
host            laskin01 (this machine)
GPU             3x Tesla V100-PCIE-32GB, compute capability 7.0 (Volta, sm_70)
driver          570.211.01, CUDA 12.8
python          3.10.12 (system), 3.11.14 (agent env)
media           ffmpeg + ffprobe present
object store    rclone remote `s3allas:` works (Allas reachable)
ollama          running; several models resident — the GPU is shared
```

Diagnostic script: `scripts/laskin/check_vllm_environment.sh` (read-only).

Two pre-existing traps on this machine, both measured:

- the installed **vLLM 0.15.1** under `~/.local` (py3.10) is **broken**:
  `ImportError: cannot import name 'infer_schema' from 'torch.library'`;
- the **py3.11** torch `2.14.1+cu130` **cannot initialise CUDA** against the 12.8
  driver ("The NVIDIA driver on your system is too old").

Neither is used by this experiment: it runs in an isolated venv precisely so
those two cannot shadow the working stack.

## 2. Why the stack is pinned (the architectural constraint)

Laskin is **Volta (sm_70)**. Current vLLM requires **compute capability ≥ 7.5**;
its prebuilt wheels are built on CUDA ≥ 12.8, whose arch list begins at sm_75.
Upstream dropped practical sm_70 support around vLLM 0.21.

The last era whose wheels still carry sm_70 kernels is **vLLM 0.8.x** — and that
release predates Qwen3-VL. Verified directly from the installed registry:

```text
vllm/model_executor/models/registry.py  (vLLM 0.8.5.post1)
  "Qwen2_5_VLForConditionalGeneration"   <- newest VL architecture present
  "Qwen2VLForConditionalGeneration"
  "QwenVLForConditionalGeneration"
  (no Qwen3VL entry)
```

So the reference model is blocked for **two independent reasons**: no vLLM
release both (a) contains Qwen3-VL and (b) runs on compute capability 7.0. This
is not a flag or VRAM problem.

Known escape hatch, **not** attempted here: vLLM's source retains an sm_70 CMake
path, so a source build on a CUDA 12.6 toolchain can re-enable Volta. That is a
container/source-build project, not an isolated-venv install.

## 3. Historical working combination (proven, now retired)

The following stack was measured successfully before its security retirement. It is intentionally **not shipped as an installable requirements file anymore**:

```text
vllm           0.8.5.post1
torch          2.6.0+cu124   (pulled by vllm; arch_list contains sm_70)
transformers   4.51.3        (pinned: newer transformers breaks vllm 0.8.5)
tokenizers     0.21.4
qwen-vl-utils  0.0.14
av             17.1.0
```

Historical verification: `torch.cuda.is_available()` → True, 3 devices, a real GPU matmul succeeded, and the vLLM engine loaded a model and generated text. Do not reconstruct this environment for new work: the legacy dependency versions are now security-retired.

## 4. Native video inference results (measured)

Same trimmed clip for both (`ffmpeg -ss 1.0`, 12.0 s → 11.0 s, ffprobe-verified),
native video tensor input, one video per request:

| model | native video | inference | peak GPU | output |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen2.5-VL-3B-Instruct` | ✅ | 21.9 s | 14.54 GB | "The video is a red screen." |
| `Qwen/Qwen2.5-VL-7B-Instruct` | ✅ | 21.0 s | 23.32 GB | "The video shows a solid red background with no discernible objects, characters, or actions taking place." |
| `Qwen/Qwen3-VL-8B-Instruct` (reference) | ❌ | — | — | architecture not present in any Volta-capable vLLM |

The 7B result is the useful Laskin arm: closest sibling to the 8B reference that
Volta can load, fits one 32 GB V100 with ~8.7 GB headroom, and gives materially
better output than 3B.

## 5. The API difference the harness now handles

The modern Qwen3-VL call passes per-video metadata inside `mm_processor_kwargs`.
On vLLM 0.8.5 that raises:

```text
TypeError: unhashable type: 'dict'
```

because vLLM 0.8.5 hashes the processor config. The harness therefore resolves
the video API from the installed vLLM version (`--video-api auto|modern|legacy`; `modern`/`legacy` alias the internal `mm_processor_kwargs`/`direct` shapes):

- **modern** (vLLM ≥ 0.9): `process_vision_info(..., image_patch_size=16,
  return_video_metadata=True)` + `mm_processor_kwargs` with `video_metadata`.
- **legacy** (vLLM ≤ 0.8.x): `process_vision_info(messages)` and the video
  tensors passed directly, no `mm_processor_kwargs`.

Both are recorded per output row (`vllm_video_api`, `vllm_model_version`) so a
Laskin result can never be mistaken for a Roihu one. This keeps **one** harness
for both hosts, as §2 of the issue requires, rather than a fork.

## 6. Media access from Laskin

`rclone lsd s3allas:` succeeds and lists the EP24 buckets (including
`EP2024_Mobile` and `HEPP24`), so **Allas is directly reachable from Laskin** —
no staged mirror is required. Remote names only were inspected; no credentials
are printed or committed. `--fetch-backend local` remains available for an
already-staged mirror, and `none` to fail loudly.

## 7. Running it

Do **not** run the legacy Laskin vLLM environment for new analysis. Both `scripts/laskin/install_vllm_video_test.sh` and `scripts/laskin/vllm_video_test.sh` intentionally terminate with a security-retirement message.

Use the active CSC Roihu workflow instead:

```bash
sbatch scripts/roihu/vllm_video_test.sbatch
```

See `docs/VLLM_VIDEO_TEST.md` for the current setup and execution instructions. A future Laskin path requires a newly validated, fully patched stack that supports Volta without reintroducing the retired dependencies.

## 8. Roihu vs Laskin

| | Roihu (issue #24) | Laskin (this) |
| --- | --- | --- |
| GPU | GH200, 96 GB, Hopper | V100, 32 GB, Volta (sm_70) |
| vLLM | CSC `python-vllm` module | historical 0.8.5.post1 result; execution retired |
| reference Qwen3-VL-8B | intended | **not loadable** (architecture + sm_70) |
| workable VLM | Qwen3-VL family | Qwen2.5-VL-3B/7B (measured) |
| video API | modern | legacy (auto-resolved) |
| Allas | via `allas_conf` | via `s3allas:` rclone remote |
| 1.0 s trim | mandatory | identical contract, same ffmpeg rule |

No winner is declared: the two hosts cannot run the same model, so a per-video
timing comparison is not meaningful. What is comparable — and recorded — is the
video API, the trim contract, the output schema, and the per-row provenance.
