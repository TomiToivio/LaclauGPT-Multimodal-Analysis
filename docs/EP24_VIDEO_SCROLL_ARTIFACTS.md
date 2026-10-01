# EP24 video scroll artifacts and analysis contract

The EP24 TikTok and Instagram material was recorded during the 2024 European
Parliament elections by researchers doing digital ethnography on GrapheneOS
mobile phones. Researchers captured continuous social-media feeds as mobile
screen recordings. A scroll-detection algorithm later split those recordings
into per-item clips.

That historical split has two consequences that every current and future media
analysis stage must respect.

## Mandatory initial 1.0 second exclusion

Each split clip begins with approximately one second of the scrolling transition
from the previous feed item. This is collection/splitting noise, not evidence
belonging to the current item.

The canonical rule lives in ep24_video.py:

    VIDEO_INITIAL_SKIP_SECONDS = 1.0

The value can be overridden for controlled experiments with the
LACLAUGPT_VIDEO_INITIAL_SKIP_SECONDS environment variable, but the EP24 default
is 1.0 seconds.

All analysis must operate on the logical interval from original t=1.0 seconds to
the end of the clip. The original source video is never modified.

Current handling:

- Whisper / ASR receives a non-destructive derived analysis clip beginning at
  t=1.0s.
- Keyframe extraction seeks the original source at t=1.0s, then 31.0s, 61.0s,
  and so on, up to six frames.
- OCR inherits the same exclusion because it operates only on those keyframes.
- Whole-video vLLM analysis receives a derived clip beginning at t=1.0s.
- Future frame, audio, embedding, scene-sampling, summary, fusion, or video
  modules must import the same helper instead of adding a separate magic number.

Videos with duration less than or equal to 1.0 seconds have no analyzable content
under this contract. They must fail gracefully and be marked as too short rather
than analyzed from the known scroll artifact.

Historical cache rows are preserved for reproducibility. New processing obeys the
skip rule and records additive status/provenance fields; no legacy column is
removed.

## Additional scroll detection

The original splitter occasionally failed, leaving more than one feed item in a
single clip. Whole-video VLM analysis must inspect the content after the initial
artifact and distinguish an actual TikTok/Instagram feed scroll from ordinary
motion such as a pan, zoom, cut, animation, camera movement, or scrolling inside
a post.

The structured fields are:

    SCROLL: TRUE | FALSE
    SCROLL_SECONDS: [...]

SCROLL is FALSE with an empty list when no additional feed scroll is present.
The known first-second transition must never make SCROLL true.

Timestamps are on the original source timeline. Because the derived vLLM input
begins at original t=1.0s, the prompt explicitly tells the model to add one
second to timestamps observed in that derived clip.

The human-readable Markdown analysis remains the primary descriptive output. The
VLM appends a small JSON object containing SCROLL and SCROLL_SECONDS, and the
harness also writes normalized CSV columns with those values.

## Re-splitting workflow

Additional scrolls set needs_resplit=TRUE when the clip has not already reached
the configured recursion limit. Automatic destructive editing is deliberately
not performed on first-pass model output.

ep24_video.py provides deterministic helpers for the next stage:

- normalize_scroll_metadata removes duplicates and ignores boundaries at or
  before the mandatory initial skip.
- split_plan creates ordered non-overlapping segment boundaries.
- derived_segment_id creates stable child identifiers carrying the parent ID,
  segment number, and start/end timestamps.
- MAX_RESPLIT_DEPTH defaults to 1, preventing an infinite re-split loop.

A re-split implementation must create derived clips only, never overwrite the
source, and must copy all source URL and legacy dataframe metadata into each
derived row. Every derived clip is itself subject to the same initial-scroll
analysis rule.

The explicit needs_resplit state is intentional. It keeps the VLM detection
auditable before creating more research artifacts while making the subsequent
operation deterministic.

## CSC Allas and Roihu workflow

Video storage remains simple and unchanged:

1. Read the source URL/video reference from the EP24 dataframe.
2. Download or stage the required video from CSC Allas using the configured
   Allas environment on CSC Roihu.
3. Keep the source clip unchanged.
4. Create temporary/derived analysis media where required to enforce the
   first-second exclusion.
5. Run frame, OCR, ASR, VLM, summary, and later multimodal stages.
6. Write backwards-compatible Pandas CSV output and preserve all legacy fields.
7. Use MongoDB/Redis only for the separately specified memory, RAG, backup, or
   orchestration roles. This video contract does not redesign storage.

## Compatibility

The legacy branch remains the historical publication archive and is not modified
by this work. The main pipeline adds these rules around the existing legacy
contract.

Do not remove legacy stages, legacy dataframe fields, source URLs, Markdown
summaries, or CSV/Pandas artifacts. New provenance and quality fields are
additive.
