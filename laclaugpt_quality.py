"""Shared deterministic OK / REPROCESS / DELETE quality gate for EP24 steps 2-5."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

QUALITY_STATUSES = ("OK", "REPROCESS", "DELETE")
_STATUS_PRIORITY = {"OK": 0, "REPROCESS": 1, "DELETE": 2}
_STATUS_RE = re.compile(r"\b(OK|REPROCESS(?:ED)?|DELETE(?:D)?)\b", re.IGNORECASE)

QUALITY_COLUMNS = (
    "processing_status",
    "processing_status_reason",
)

STEP_QUALITY_COLUMNS = (
    "frame_quality_status",
    "frame_quality_reason",
    "video_quality_status",
    "video_quality_reason",
    "summary_quality_status",
    "summary_quality_reason",
)


def normalize_status(value: object, *, default: str = "OK", strict: bool = False) -> str:
    """Coerce a value to a valid quality status.

    Unrecognised text is common in real data -- an older schema, a model that
    wrote prose into a status field, a hand-edited CSV. Raising here is *not* a
    safe default: Step 5 normalizes the whole incoming column before routing, and
    that call sits outside any per-row error handling, so one unexpected value
    aborts the entire step for every country.

    The default is therefore to fall back to `default`, which routes the row
    onward for review rather than deleting it from the dataset. Pass
    `strict=True` where the caller wants to reject bad input explicitly, such as
    validating a model response before trusting it.
    """
    text = "" if value is None else str(value).strip().upper()
    aliases = {
        "REPROCESSED": "REPROCESS",
        "DELETED": "DELETE",
    }
    text = aliases.get(text, text)
    if not text:
        return default
    if text not in _STATUS_PRIORITY:
        if strict:
            raise ValueError(f"invalid processing status: {value!r}")
        return default
    return text


def normalize_reason(value: object, *, fallback: str = "") -> str:
    text = "" if value is None else " ".join(str(value).strip().split())
    return text or fallback


def status_from_text(text: object, *, default: str = "OK") -> str:
    """Best-effort extraction from model prose. Explicit stronger labels win."""
    raw = "" if text is None else str(text)
    statuses = []
    for match in _STATUS_RE.findall(raw):
        token = match.upper()
        if token.startswith("REPROCESS"):
            statuses.append("REPROCESS")
        elif token.startswith("DELETE"):
            statuses.append("DELETE")
        else:
            statuses.append("OK")
    if not statuses:
        return default
    return max(statuses, key=_STATUS_PRIORITY.__getitem__)


def merge_status(
    current_status: object,
    new_status: object,
    *,
    current_reason: object = "",
    new_reason: object = "",
) -> tuple[str, str]:
    """Escalate only: DELETE > REPROCESS > OK."""
    current = normalize_status(current_status or "OK")
    new = normalize_status(new_status or "OK")
    if _STATUS_PRIORITY[new] > _STATUS_PRIORITY[current]:
        return new, normalize_reason(new_reason)
    if _STATUS_PRIORITY[new] < _STATUS_PRIORITY[current]:
        return current, normalize_reason(current_reason)
    reason = normalize_reason(new_reason) or normalize_reason(current_reason)
    return current, reason


def is_empty_text(value: object) -> bool:
    text = "" if value is None else str(value).strip()
    return not text or text.casefold() in {"nan", "none", "null", "<none>"}


def evidence_is_meaningful(*values: object) -> bool:
    """Conservative signal check used only for the empty-transcript rule."""
    text = "\n".join(str(v or "") for v in values).casefold()
    if not text.strip():
        return False
    negative_markers = (
        "no meaningful content",
        "meaningless content",
        "blank frame",
        "black frame",
        "garbage",
        "unusable",
        "corrupt",
        "loading screen",
        "only ui",
        "no useful content",
    )
    positive_markers = (
        "person",
        "people",
        "speech",
        "text",
        "caption",
        "scene",
        "object",
        "politician",
        "flag",
        "logo",
        "rally",
        "interview",
        "meeting",
        "street",
        "building",
        "screen",
    )
    if any(marker in text for marker in positive_markers):
        return True
    return not any(marker in text for marker in negative_markers)


def quality_decision_from_analysis(
    analysis: object,
    *,
    failure: bool = False,
    failure_reason: str = "",
) -> tuple[str, str]:
    if failure:
        return "REPROCESS", normalize_reason(failure_reason, fallback="recoverable processing failure")
    status = status_from_text(analysis, default="OK")
    if status == "DELETE":
        return status, "analysis identifies the source as unusable/garbage"
    if status == "REPROCESS":
        return status, "analysis identifies a recoverable extraction/splitting/processing problem"
    return "OK", "analysis found usable content"


def summary_quality_decision(
    *,
    transcript: object,
    frame_analysis: object,
    video_analysis: object,
    prior_status: object = "OK",
    prior_reason: object = "",
) -> tuple[str, str]:
    """Apply the explicit issue #204 multimodal empty-transcript rule."""
    status = normalize_status(prior_status or "OK")
    reason = normalize_reason(prior_reason)

    if not is_empty_text(transcript):
        return status, reason or "transcript and multimodal evidence available"

    meaningful = evidence_is_meaningful(frame_analysis, video_analysis)
    frame_status = status_from_text(frame_analysis, default="OK")
    video_status = status_from_text(video_analysis, default="OK")

    if frame_status == "DELETE" or video_status == "DELETE" or not meaningful:
        return merge_status(
            status,
            "DELETE",
            current_reason=reason,
            new_reason="empty transcript and no meaningful visual/video evidence",
        )
    return merge_status(
        status,
        "REPROCESS",
        current_reason=reason,
        new_reason="empty transcript but meaningful visual/video evidence; transcription likely failed",
    )


def partition_by_status(frame):
    """Return (OK, REPROCESS, DELETE) dataframe partitions without mutating input."""
    if "processing_status" not in frame.columns:
        working = frame.copy()
        working["processing_status"] = "OK"
    else:
        working = frame.copy()
    statuses = working["processing_status"].map(lambda value: normalize_status(value or "OK"))
    return (
        working.loc[statuses == "OK"].copy(),
        working.loc[statuses == "REPROCESS"].copy(),
        working.loc[statuses == "DELETE"].copy(),
    )


def reprocess_output_path(output: str | Path) -> Path:
    path = Path(output)
    name = path.name
    if "step_05" in name:
        return path.with_name(name.replace("step_05", "step_05_reprocess", 1))
    if "step_5" in name:
        return path.with_name(name.replace("step_5", "step_5_reprocess", 1))
    return path.with_name(f"{path.stem}_reprocess{path.suffix or '.csv'}")


def delete_audit_output_path(output: str | Path) -> Path:
    path = Path(output)
    return path.with_name(f"{path.stem}_delete_audit.csv")


def safe_delete_local_paths(row, columns: Iterable[str] = ("frame_file", "vllm_video_local_path", "vllm_video_analysis_path")) -> list[str]:
    """Delete only local intermediate files. Never touch URLs, s3://, swift, or Allas source keys."""
    deleted: list[str] = []
    for column in columns:
        raw = str(row.get(column, "") or "").strip()
        if not raw or "://" in raw:
            continue
        path = Path(raw)
        try:
            if path.is_file():
                path.unlink()
                deleted.append(str(path))
        except OSError:
            continue
    return deleted
