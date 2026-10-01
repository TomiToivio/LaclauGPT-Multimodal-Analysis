"""Fetch the FLEURS evaluation samples used by the ASR benchmark.

Public academic data (google/fleurs), deterministic selection: 2 utterances per
language from the `test` split, for the 10 EP24 languages.

No private EP24 material is downloaded or used.

Usage::

    python scripts/asr_bench/fetch_fleurs_samples.py --out /tmp/asr-bench

Requires ``datasets``, ``soundfile`` and ``librosa``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# EP24 languages -> FLEURS config name.
# FLEURS uses es_419 / pt_br for Spanish and Portuguese.
LANGUAGES = {
    "en_us": "en_us",
    "fi_fi": "fi_fi",
    "pl_pl": "pl_pl",
    "de_de": "de_de",
    "fr_fr": "fr_fr",
    "es_es": "es_419",
    "sv_se": "sv_se",
    "pt_pt": "pt_br",
    "hu_hu": "hu_hu",
    "hr_hr": "hr_hr",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="audio", help="output directory for wav files")
    ap.add_argument("--per-language", type=int, default=2)
    args = ap.parse_args()

    import soundfile as sf
    from datasets import load_dataset

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta: dict[str, dict] = {}

    for code, config in LANGUAGES.items():
        split = f"test[:{args.per_language}]"
        ds = load_dataset("google/fleurs", config, split=split)
        for i, row in enumerate(ds):
            path = out / f"{code}_{i + 1}.wav"
            audio = row["audio"]
            sf.write(path, audio["array"], audio["sampling_rate"])
            text = row.get("transcription") or row.get("raw_transcription") or ""
            meta[f"{code}_{i + 1}.wav"] = {
                "lang": code,
                "text": text,
                "dur": round(len(audio["array"]) / audio["sampling_rate"], 2),
            }
        print(f"  {code} (via {config}): {args.per_language} utterances")

    (out / "refs.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"total: {len(meta)} utterances -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
