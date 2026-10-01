"""Benchmark faster-whisper checkpoints on the same FLEURS samples.

Same normalisation and temperature ladder as the openai-whisper run so the two
tables are directly comparable.
"""
import json
import re
import time
import unicodedata
from pathlib import Path

meta = json.load(open("audio/refs.json"))
TEMP = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def wer(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    if not r:
        return None
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(
                d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] != h[j - 1])
            )
    return d[len(r)][len(h)] / len(r)


from faster_whisper import WhisperModel  # noqa: E402

results = {}
for ckpt in ["large-v3", "large-v3-turbo", "distil-large-v3"]:
    for ct in ["int8_float16", "float16"]:
        try:
            t0 = time.time()
            m = WhisperModel(ckpt, device="cuda", compute_type=ct)
            load = time.time() - t0
        except Exception as e:
            print(f"=== faster-whisper {ckpt} {ct}: LOAD FAILED {type(e).__name__} {str(e)[:90]}", flush=True)
            continue
        print(f"=== faster-whisper {ckpt} {ct} (load {load:.1f}s) ===", flush=True)
        rows = []
        for p, r in sorted(meta.items()):
            t0 = time.time()
            segs, info = m.transcribe(p, temperature=tuple(TEMP))
            hyp = " ".join(s.text for s in segs).strip()
            dt = time.time() - t0
            rows.append(
                {
                    "file": p, "lang": r["lang"], "ref": r["text"], "hyp": hyp,
                    "detected": getattr(info, "language", None),
                    "sec": round(dt, 2), "dur": r["dur"], "wer": wer(r["text"], hyp),
                }
            )
            print(f"   {Path(p).name:16} {r['lang']} WER={rows[-1]['wer']:.3f} {dt:.1f}s", flush=True)
        results[f"faster-whisper:{ckpt}:{ct}"] = rows
        del m
        json.dump(results, open("results_faster.json", "w"), indent=1)
        print(f"   -> saved after {ckpt}/{ct}", flush=True)

print("done")
