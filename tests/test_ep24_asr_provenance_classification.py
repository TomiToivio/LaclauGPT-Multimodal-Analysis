"""The ASR/preprocess fields are model-derived, not factual source metadata (#129).

`ep24_pipeline.metadata_context` renders each row into three provenance
sections before it reaches the frame-analysis model: factual source metadata,
human researcher annotation, and derived model/enrichment output. The split
exists so the model does not treat model output as ground truth.

#130 renamed the producer's `whisper_*` fields to `asr_*` and added
`preprocess_*`, but `MODEL_PREFIXES` was not updated. A machine-generated
transcript was therefore presented to the model as factual
researcher-recorded source context -- a provenance inversion, and the opposite
of what the stage prompt's own observation/inference separation requires.

These tests pin the classification so a future rename cannot silently invert it
again. Synthetic data only; no model, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ep24_pipeline import MODEL_PREFIXES, metadata_context  # noqa: E402

#: Fields Step 1 produces that are machine output, not human-recorded facts.
MODEL_DERIVED_FIELDS = (
    "asr_transcript", "asr_language", "asr_translated", "asr_backend", "asr_model",
    "asr_runtime_ms", "ocr_1", "ocr_backend", "ocr_model", "ocr_runtime_ms",
    "frame_file", "frame_timestamp_seconds", "preprocess_status", "preprocess_note",
    "preprocess_completed_at",
)

#: Fields a human researcher supplied; these must stay in the annotation bucket.
RESEARCHER_FIELDS = (
    "political_preference", "new_entity", "new_theme",
    "researcher_new_persons", "researcher_new_themes", "researcher_note",
)


def _section_of(ctx: str, column: str) -> str:
    """Return the header of the provenance section `column` was rendered into."""
    lines = ctx.splitlines()
    line = next((cand for cand in lines if cand.startswith(f"- {column}:")), None)
    assert line is not None, f"{column} was not rendered at all"
    for cand in reversed(lines[:lines.index(line)]):
        if cand.endswith(":") and not cand.startswith("- "):
            return cand
    return "?"


def test_every_model_derived_field_matches_a_model_prefix():
    unmatched = [c for c in MODEL_DERIVED_FIELDS
                 if not c.startswith(MODEL_PREFIXES)]
    assert not unmatched, (
        "these fields are machine output but would be rendered as factual source "
        f"metadata: {unmatched}"
    )


def test_researcher_fields_are_not_treated_as_model_output():
    wrong = [c for c in RESEARCHER_FIELDS if c.startswith(MODEL_PREFIXES)]
    assert not wrong, f"human annotation must not be marked derived: {wrong}"


def test_asr_output_renders_in_the_derived_section():
    row = pd.Series({
        "country": "Finland",
        "author_username": "auth",
        "allas_filename": "a.mp4",
        "political_preference": "right",
        "researcher_note": "human note",
        "asr_transcript": "MACHINE TRANSCRIPT",
        "asr_translated": "MACHINE TRANSLATION",
        "asr_backend": "parakeet",
        "preprocess_status": "ok",
        "ocr_1": "ON-SCREEN TEXT",
    })
    ctx = metadata_context(row)
    for column in ("asr_transcript", "asr_translated", "asr_backend",
                   "preprocess_status", "ocr_1"):
        assert "derived" in _section_of(ctx, column), (
            f"{column} must render as derived model output"
        )


def test_machine_text_never_appears_in_the_factual_section():
    """The whole point of the split: derived text must not reach the model as fact."""
    row = pd.Series({
        "country": "Finland",
        "allas_filename": "a.mp4",
        "asr_transcript": "MACHINE_TRANSCRIPT_MARKER",
        "asr_backend": "parakeet",
        "preprocess_status": "ok",
    })
    ctx = metadata_context(row)
    factual = ctx.split("RESEARCHER ANNOTATION")[0]
    assert "MACHINE_TRANSCRIPT_MARKER" not in factual
    assert "parakeet" not in factual


def test_researcher_fields_still_render_as_annotation():
    row = pd.Series({
        "country": "Finland",
        "allas_filename": "a.mp4",
        "researcher_note": "HUMAN_NOTE_MARKER",
        "political_preference": "left",
    })
    ctx = metadata_context(row)
    assert "RESEARCHER" in _section_of(ctx, "researcher_note")
    assert "HUMAN_NOTE_MARKER" not in ctx.split("RESEARCHER ANNOTATION")[0]


def test_legacy_whisper_prefix_remains_classified():
    """Frozen legacy artifacts still read back through the old names."""
    assert "whisper_" in MODEL_PREFIXES
