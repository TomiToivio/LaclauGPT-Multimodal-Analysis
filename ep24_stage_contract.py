"""Machine-readable stage contract for the numbered EP24 Roihu pipeline (issue #64).

Every numbered stage is additive: it receives the cumulative dataframe, preserves
every incoming column, and appends its own fields. The pipeline relies on that
discipline being upheld by hand in nine separate scripts, which means the one
failure mode that matters -- a stage silently dropping a column an earlier stage
produced -- is invisible until a downstream stage reads a missing field.

This module makes the column contract explicit and testable without making the
individual analysis scripts complicated:

``STAGE_CONTRACT`` names, for each stage, the file that implements it and the
columns that stage *appends*. It is deliberately a contract about *additions*:
the shared rule (every prior column survives) is enforced once by
``ep24_pipeline.assert_source_metadata_preserved`` and asserted for every
boundary by ``tests/test_ep24_stage_contract.py``. The ``appends`` tuples record
what each stage writes today, so a refactor that stops writing one of them fails
a test instead of silently shrinking the cumulative dataframe.

Nothing here filters or projects a dataframe. A country CSV that carries extra
columns still flows through untouched -- the contract tests the pipeline, it does
not reshape the data.
"""
from __future__ import annotations

from dataclasses import dataclass

from ep24_schema import EP24_REPROCESS_COLUMNS

# The canonical source columns every stage must still be carrying at the end.
# Re-exported here so a stage-contract test needs only one import.
SOURCE_COLUMNS: tuple[str, ...] = EP24_REPROCESS_COLUMNS

# The private country CSVs are canonicalized before analysis. Bootstrap therefore
# adds only stable pipeline identity; human `entities` and `themes` already exist.
BOOTSTRAP_ADDED_COLUMNS: tuple[str, ...] = (
    "_storage_id",
)

# Issue #21 removes the four legacy annotation columns before Step 1.
BOOTSTRAP_PRESERVED_COLUMNS: tuple[str, ...] = ()


@dataclass(frozen=True)
class Stage:
    """One numbered pipeline stage and the columns it is contracted to append."""

    number: int
    name: str
    module: str
    entry_point: str
    sbatch: str
    appends: tuple[str, ...] = ()
    gpu: bool = True
    notes: str = ""


def _s(number, name, module, appends=(), gpu=True, notes=""):
    return Stage(
        number=number,
        name=name,
        module=module,
        entry_point=f"step_{number}_roihu_{name}.py",
        sbatch=f"scripts/roihu/step_{number}_roihu_{name}.sbatch",
        appends=tuple(appends),
        gpu=gpu,
        notes=notes,
    )


STAGE_CONTRACT: tuple[Stage, ...] = (
    _s(
        1, "preprocess", "roihu_preprocess.py",
        ("frame_file", "frame_timestamp_seconds", "ocr_1", "ocr_backend", "ocr_model",
         "ocr_runtime_ms", "asr_transcript", "asr_language", "asr_translated",
         "asr_backend", "asr_model", "asr_runtime_ms", "video_duration_seconds",
         "preprocess_status", "preprocess_note", "preprocess_completed_at"),
        notes="Allas media is staged here; exactly one frame/OCR is taken at original t=1.0s and full-video ASR is backend-neutral.",
    ),
    _s(
        2, "frame", "roihu_frame.py",
        ("frame_analysis_1", "frame_analysis_timestamp_seconds",
         "frame_analysis_status", "frame_analysis_model",
         "frame_analysis_context_sha256"),
        notes="Exactly one Step 1 keyframe at original t=1.0s; all accumulated row fields are preserved and passed as cumulative prompt context.",
    ),
    _s(
        3, "video", "step_3_roihu_video.py",
        ("vllm_video_model", "vllm_video_version", "vllm_video_status", "vllm_video_analysis",
         "vllm_video_error", "vllm_video_source_row_index", "vllm_video_source_id",
         "vllm_video_allas_source", "vllm_video_remote_path", "vllm_video_local_path",
         "vllm_video_analysis_path", "vllm_video_remote_path_logged", "vllm_video_bytes",
         "vllm_video_sha256", "vllm_video_source_duration_seconds",
         "vllm_video_analysis_duration_seconds", "vllm_video_trim_command",
         "vllm_video_trim_exit_status", "vllm_video_prompt_version", "vllm_video_prompt_sha256",
         "vllm_video_markdown_analysis", "vllm_video_raw_output", "vllm_video_structured_json",
         "vllm_video_structured_output_status", "vllm_video_structured_output_error",
         "vllm_video_runtime_seconds", "vllm_video_selected_index", "vllm_video_prompt",
         "vllm_video_context_sha256", "vllm_video_persistence_status",
         "SCROLL", "SCROLL_SECONDS", "needs_resplit", "video_initial_skip_seconds",
         "vllm_video_api", "vllm_video_analyzed_duration_seconds",
         "vllm_video_inference_seconds", "vllm_video_prompt_hash",
         "vllm_version", "vllm_torch_version", "vllm_cuda_version", "vllm_gpu_name",
         "vllm_hostname", "vllm_peak_gpu_memory_mb", "vllm_structured_status",
         "vllm_structured_output"),
        notes="Cumulative whole-video VLM stage. The entry point is "
              "step_3_roihu_video.py, which wraps experiments/vllm_video_test.py; "
              "ep24_video.py is the shared skip/trim rules library, not an "
              "executable stage. Column list mirrors vllm_video_test.OUTPUT_COLUMNS.",
    ),
    _s(4, "summary", "roihu_summary.py",
       ("metadata", "summary_analysis", "summary_summary_md")),
    _s(
        5,
        "postprocess",
        "roihu_postprocess.py",
        (
            "video_filename",
            "postprocess_entities",
            "postprocess_themes",
            "positive",
            "neutral",
            "negative",
            "postprocess_summary_md",
            "ep24_entity_resolution_json",
            "ep24_entity_ids",
            "ep24_entity_canonical_names",
            "ep24_entity_unresolved_json",
            "ep24_codebook_fingerprint",
            "ep24_codebook_context_json",
            "ep24_memory_entity_ids",
            "ep24_memory_topic_ids",
            "ep24_memory_theme_ids",
            "ep24_theme_resolution_json",
            "ep24_theme_canonical_names",
            "ep24_memory_sentiment_target_ids_json",
            "ep24_memory_unresolved_json",
            "ep24_seed_entities_json",
            "ep24_seed_themes_json",
            "ep24_sentiment_targets_json",
            "ep24_human_seed_context",
        ),
        notes="Structured machine entity/theme/sentiment extraction is stored separately from immutable human entities/themes, followed immediately by conservative codebook/memory normalization before Step 6.",
    ),
    _s(
        6,
        "discourse_analysis",
        "roihu_populism.py",
        (
            "formula_of_populism_analysis",
            "formula_of_populism_us",
            "formula_of_populism_frontier",
            "laclau_summary_md",
            "laclau_structured_json",
            "laclau_raw_response",
            "laclau_status",
            "laclau_error",
            "laclau_prompt_version",
            "laclau_model",
            "laclau_context_sha256",
            "laclau_generated_at",
            "formula_of_populism_codebook_context_json",
            "formula_of_populism_codebook_fingerprint",
        ),
        notes="Evidence-linked document-level Laclau/Palonen candidates. Rich JSON is canonical; historical element^affect columns are deterministic compatibility projections.",
    ),
    _s(7, "discourse_network_analysis", "roihu_identity.py", (),
       notes="Statement extraction; writes through the DNA module rather than df columns."),
    _s(8, "social_network_analysis", "roihu_enrich.py",
       ("ep24_codebook_fingerprint", "ep24_seed_entities_json",
        "ep24_memory_entity_ids", "ep24_memory_sentiment_target_ids_json",
        "ep24_sentiment_targets_json")),
    _s(9, "rdf", "roihu_rdf.py", (), gpu=False,
       notes="Deterministic CPU-only export; emits RDF, not dataframe columns."),
)

STAGES_BY_NUMBER: dict[int, Stage] = {stage.number: stage for stage in STAGE_CONTRACT}

# Country processing priority (issue #64): the first three are required for the
# demo samples; the remainder follows one deterministic documented order.
COUNTRY_PRIORITY: tuple[str, ...] = ("Finland", "Poland", "Portugal")
COUNTRY_PROCESSING_ORDER: tuple[str, ...] = (
    "Finland",
    "Poland",
    "Portugal",
    "Germany",
    "Spain",
    "Hungary",
    "Croatia",
    "France",
    "Bulgaria",
    "Sweden",
)

# Lowercase tokens as they appear in the private input filenames
# (analysis/ep24_reprocess/data/to_reprocess/ep24_<token>.csv).
COUNTRY_TOKENS: dict[str, str] = {
    "Finland": "finland",
    "Poland": "poland",
    "Portugal": "portugal",
    "Bulgaria": "bulgaria",
    "Croatia": "croatia",
    "France": "france",
    "Germany": "germany",
    "Hungary": "hungary",
    "Spain": "spain",
    "Sweden": "sweden",
}


def stage(number: int) -> Stage:
    """Return the contract for one stage, or raise a clear error."""
    try:
        return STAGES_BY_NUMBER[number]
    except KeyError:
        known = ", ".join(str(s.number) for s in STAGE_CONTRACT)
        raise KeyError(f"unknown EP24 stage {number!r}; known stages: {known}") from None


def country_order(available: list[str]) -> list[str]:
    """Order known EP24 countries exactly as required by issue #128.

    Unknown countries are never dropped: they follow the ten known countries in
    deterministic alphabetical order.
    """
    present = list(dict.fromkeys(available))
    known = [country for country in COUNTRY_PROCESSING_ORDER if country in present]
    unknown = sorted(country for country in present if country not in COUNTRY_PROCESSING_ORDER)
    return known + unknown


def all_contracted_columns() -> tuple[str, ...]:
    """Every column the contract says is present, source columns first."""
    ordered: list[str] = list(SOURCE_COLUMNS)
    ordered.extend(BOOTSTRAP_ADDED_COLUMNS)
    for stage_ in STAGE_CONTRACT:
        for column in stage_.appends:
            if column not in ordered:
                ordered.append(column)
    return tuple(ordered)