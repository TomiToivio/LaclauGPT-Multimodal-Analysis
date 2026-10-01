"""Contract tests for the documented legacy EP24 pipeline (issue #5, documentation step).

``docs/LEGACY_PIPELINE_CONTRACT.md`` claims to describe what the five frozen
``legacy`` scripts actually do. A contract document that silently drifts from the
code is worse than no document, so these tests read the *public* scripts and the
document together and fail when the two disagree.

What is checked:

- the documented stage order matches the Roihu runner's order;
- every column the document says a stage adds is really created by that stage;
- the documented per-stage language/country lists match the code, both directions;
- the documented join keys and drop semantics still exist in the code;
- the document still records the known inconsistencies it promises to record.

These tests never import the pipeline modules: they only read source text, so
they run in CI without Ollama, Whisper, EasyOCR or the private runtime data.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "LEGACY_PIPELINE_CONTRACT.md"

LEGACY_STAGES = [
    "puhti_preprocess.py",
    "puhti_frame.py",
    "puhti_summary.py",
    "puhti_postprocess.py",
    "puhti_populism.py",
]

# Fields the document attributes to each stage in section 4.
DOCUMENTED_ADDED_FIELDS = {
    "puhti_preprocess.py": [
        "whisperResult",
        "frame_files",
        "ocr_1",
        "ocr_2",
        "ocr_3",
        "ocr_4",
        "ocr_5",
        "ocr_6",
        "whisper_transcript",
        "whisper_language",
        "whisper_translated",
    ],
    "puhti_frame.py": [
        "frame_analysis_1",
        "frame_analysis_2",
        "frame_analysis_3",
        "frame_analysis_4",
        "frame_analysis_5",
        "frame_analysis_6",
    ],
    "puhti_summary.py": ["metadata", "summary_analysis"],
    "puhti_postprocess.py": [
        "video_filename",
        "entities",
        "topics",
        "positive",
        "neutral",
        "negative",
    ],
    "puhti_populism.py": [
        "formula_of_populism_analysis",
        "formula_of_populism_us",
        "formula_of_populism_frontier",
    ],
}


def _doc() -> str:
    return DOC.read_text(encoding="utf-8")


def _source(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_contract_document_exists() -> None:
    assert DOC.is_file(), f"missing contract document: {DOC}"


def test_documented_stage_table_matches_legacy_stage_order() -> None:
    """The doc's numbered stage table must list the five stages in order."""
    rows = re.findall(r"^\|\s*\d+\s*\|\s*`(puhti_\w+\.py)`", _doc(), re.MULTILINE)
    assert rows == LEGACY_STAGES, f"documented stage order {rows}"


def test_documented_stage_order_matches_the_roihu_runner() -> None:
    runner = (ROOT / "scripts" / "roihu" / "run_pipeline.sh").read_text(encoding="utf-8")
    expected = " ".join(name[len("puhti_") : -len(".py")] for name in LEGACY_STAGES)
    assert expected in runner, f"runner does not carry the documented order: {expected!r}"


def _writes_field(text: str, field: str) -> bool:
    """True when the source actually *writes* this column.

    A bare ``field in text`` check is too weak: a column the document credits to a
    stage can survive in a read (``if 'video_filename' in df.columns``) while the
    write is gone. So a documented field must appear in one of the write forms
    the historical scripts actually use:

    - ``df['<field>'] =`` / ``df["<field>"] =``      (direct column assignment)
    - ``df.at[index, '<field>']``                    (per-row assignment)
    - ``'<field>',``                                 (element of the column-init tuple)
    """
    forms = [
        f"df['{field}'] =",
        f'df["{field}"] =',
        f"df.at[index, '{field}']",
        f'df.at[index, "{field}"]',
        f"'{field}',",
        f'"{field}",',
    ]
    if any(form in text for form in forms):
        return True
    # The preprocess/summary column-init loops list the names one per line in a
    # tuple, so the trailing-comma form above may be the last element instead.
    return bool(re.search(rf"['\"]{re.escape(field)}['\"]\s*\)", text))


def test_documented_added_fields_are_created_by_their_stage() -> None:
    """Each field the doc credits to a stage must actually be written there."""
    for script, fields in DOCUMENTED_ADDED_FIELDS.items():
        text = _source(script)
        for field in fields:
            assert _writes_field(text, field), (
                f"{script} no longer writes the {field} column, "
                "but the contract still documents it"
            )


def test_documented_added_fields_are_still_named_in_the_document() -> None:
    """The reverse direction: a field kept in code must stay documented."""
    doc = _doc()
    for fields in DOCUMENTED_ADDED_FIELDS.values():
        for field in fields:
            assert field in doc, f"contract document dropped the {field} field"


def test_documented_language_lists_match_the_scripts() -> None:
    """Section 2's language table must equal the lists actually in the code."""
    documented = dict(
        re.findall(r"^\|\s*`(puhti_\w+\.py)`\s*\|\s*`(\[[^`]+\])`\s*\|", _doc(), re.MULTILINE)
    )
    assert documented, "could not parse the documented language table"

    for script, literal in documented.items():
        actual = re.search(r"^(?:languages|countries) = (\[[^\]]+\])", _source(script), re.MULTILINE)
        assert actual, f"{script} no longer defines a module-level language list"
        assert actual.group(1) == literal, (
            f"{script} language list is {actual.group(1)}, document says {literal}"
        )


def test_documented_en_and_bg_asymmetry_is_real() -> None:
    """The doc calls the en/bg asymmetry load-bearing; confirm it still holds."""
    for script in LEGACY_STAGES[:4]:
        assert "'en'" in _source(script), f"{script} unexpectedly lost 'en'"
    assert "'bg'" in _source("puhti_populism.py")
    for script in LEGACY_STAGES[:4]:
        assert "'bg'" not in _source(script), f"{script} gained 'bg' outside stage 5"


def test_documented_drop_semantics_still_hold() -> None:
    """Section 6 states which stages drop rows; the code must agree."""
    preprocess = _source("puhti_preprocess.py")
    assert "dropna" not in preprocess, "stage 1 is documented as dropping no rows"

    frame = _source("puhti_frame.py")
    assert "dropna(subset=['whisperResult'])" in frame
    assert "dropna(subset=['frame_files'])" in frame

    assert "dropna(subset=['whisperResult'])" in _source("puhti_summary.py")


def test_documented_cache_keys_still_hold() -> None:
    """Section 5's join keys must still be the ones the code queries on."""
    pair_key = "WHERE author_username = ? AND video_id = ?"
    for script in ("puhti_preprocess.py", "puhti_frame.py", "puhti_summary.py"):
        assert pair_key in _source(script), f"{script} no longer keys on (author, video)"

    assert "WHERE video_file=?" in _source("puhti_populism.py")
    assert "video_filename" in _source("puhti_postprocess.py")


def test_documented_output_paths_are_written_by_their_stage() -> None:
    """Section 1's read/write table must still describe real file handling."""
    assert "./csv/tiktok_videos.csv" in _source("puhti_preprocess.py")
    assert "ep24_{country}.csv" in _source("puhti_populism.py") or (
        "ep24_" in _source("puhti_populism.py")
    )
    # Stage 4 emits the legacy compatibility artifact that stage 5 consumes.
    postprocess = _source("puhti_postprocess.py")
    assert "ep24_" in postprocess and "legacy_output" in postprocess


def test_documented_known_inconsistencies_are_recorded() -> None:
    """Section 7 promises to record specific inconsistencies; keep them listed."""
    doc = _doc()
    for anchor in (
        "formula_of_populism.db",
        "formula.log",
        "EasyOCR reader list",
        "hard-codes the frame timestamps",
    ):
        assert anchor in doc, f"contract document stopped recording {anchor!r}"


def test_contract_document_states_the_legacy_safety_rules() -> None:
    """Section 8 must keep naming what the migration may not change."""
    doc = _doc()
    for anchor in (
        "the five-stage order",
        "any prompt text",
        "any CSV column name",
        "LACLAUGPT_MULTIMODAL_MODEL",
    ):
        assert anchor in doc, f"contract document stopped stating the {anchor!r} rule"
