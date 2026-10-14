"""Bounded, provenance-labelled prompt evidence for EP24 Steps 6-8.

Direct documentary evidence has priority. External memory/RAG never constitutes
evidence for a new political claim, statement, or actor relation.
This module does not modify cumulative source rows.
"""
from __future__ import annotations

import logging
import os

LOG = logging.getLogger(__name__)

# Field order is deliberate: source speech and visible words outrank prior models.
SOURCE_FIELDS = (
    ("asr_transcript", 9000),
    ("videoDescription", 4500),
    ("whisper_transcript", 5000),
    ("asr_translation", 5500),
    ("asr_translated", 5500),
    ("whisper_translated", 3500),
    ("ocr_1", 3000),
    ("frame_analysis_1", 2500),
    ("vllm_video_analysis", 3500),
    ("summary_analysis", 4000),
)
META_FIELDS = (("video_id", 180), ("country", 90), ("platform", 100),
               ("url", 400), ("created_at", 120), ("date", 120))
RESEARCHER_FIELDS = (("entities", 1800), ("themes", 1400),
                     ("researcher_note", 1500))
DNA_FIELDS = (("dna_statements_json", 4500), ("laclau_structured_json", 1200))
SNA_FIELDS = (("dna_statements_json", 3800), ("ep24_entity_resolution_json", 1200),
              ("entities", 800), ("researcher_note", 600))


def evidence_packet(row, *, stage: int, max_chars: int | None = None) -> tuple[str, bool]:
    if stage not in (6, 7, 8):
        raise ValueError("stage must be 6, 7 or 8")
    defaults = {6: 28000, 7: 20000, 8: 12000}
    budget = max_chars if max_chars is not None else int(
        os.getenv(f"LACLAUGPT_STEP{stage}_MAX_CONTEXT_CHARS", str(defaults[stage]))
    )
    if budget < 128:
        raise ValueError("context budget must be >=128")
    sections = ["EP24 current-document evidence; prior model output is descriptive context, not direct speech."]
    fields = list(SOURCE_FIELDS) + list(META_FIELDS)
    if stage in (6, 7):
        fields += list(RESEARCHER_FIELDS)
        fields += [("codebook_context_json", 1200)]
    if stage == 7:
        fields += list(DNA_FIELDS[1:])
    if stage == 8:
        fields = [("asr_transcript", 5000), ("asr_translation", 1500),
                  ("ocr_1", 1600)] + list(SNA_FIELDS) + list(META_FIELDS)
    truncated = False
    dropped_fields = []
    source_names = {"asr_transcript", "videoDescription", "whisper_transcript",
                    "asr_translation", "asr_translated", "whisper_translated", "ocr_1"}
    for name, cap in fields:
        value = row.get(name, "")
        value = "" if value is None else str(value).strip()
        if not value or value.lower() in ("nan", "none", "[]", "{}"):
            continue
        if name == "codebook_context_json":
            role = "researcher normalization, not source quote"
        elif name in source_names:
            role = "original source/derived transcript; verify original modality"
        elif name in dict(META_FIELDS):
            role = "record metadata, not quotation evidence"
        elif name in {"frame_analysis_1", "vllm_video_analysis", "summary_analysis",
                      "laclau_structured_json", "dna_statements_json"}:
            role = "prior model analysis, context only, not source quotation"
        else:
            role = "researcher annotation or entity context, verify provenance"
        heading = f"\n[{name} | {role}]\n"
        remaining = budget - len("\n".join(sections)) - len(heading) - 44
        take = min(len(value), cap, max(0, remaining))
        if take <= 0:
            truncated = True
            dropped_fields.append(name)
            continue
        shortened = take < len(value)
        sections.append(heading + value[:take] + (" [TRUNCATED]" if shortened else ""))
        truncated |= shortened
    packet = "\n".join(sections)
    if dropped_fields:
        LOG.warning("step=%s evidence_fields_dropped=%s", stage, dropped_fields)
    LOG.info("step=%s model_visible_chars=%d budget=%d truncated=%s estimated_tokens=%d",
             stage, len(packet), budget, truncated, (len(packet)+3)//4)
    return packet, truncated
