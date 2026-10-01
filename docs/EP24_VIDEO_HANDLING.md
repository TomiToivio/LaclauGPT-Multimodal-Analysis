# EP24 mobile video handling

This document records the collection-specific video rules used by the active Roihu pipeline.

## Collection method

The EP24 videos were recorded on GrapheneOS phones by researchers conducting digital ethnography on TikTok and Instagram during the 2024 European Parliament elections. Continuous feed recordings were later split into per-item clips with a scroll-detection algorithm.

That collection method creates two important conditions:

1. the first second of each split clip contains the tail of the scroll from the previous feed item;
2. the splitter can occasionally miss a later scroll, leaving more than one feed item inside one clip.

## Canonical implementation

The active implementation lives in `ep24_video.py`. Do not create a second competing video-rule module.

The canonical constant is:

```python
VIDEO_INITIAL_SKIP_SECONDS = 1.0
```

The module provides the shared helpers used by preprocessing and whole-video analysis, including:

- `analysis_start_seconds()`
- `analysis_frame_times()`
- `is_too_short()`
- `prepare_analysis_clip()`
- `parse_scroll_metadata()`
- `needs_resplit()`
- `split_plan()`
- `derived_segment_id()`
- `derived_rows()`

## Mandatory first-second exclusion

Every EP24 media-analysis path must exclude the first 1.0 second.

The original source video is never destructively rewritten. Current readers either seek past the first second or create a deterministic derived analysis clip. The active preprocessing stage uses the same shared contract for frame extraction and ASR.

This applies to:

- frame and image analysis;
- OCR based on extracted frames;
- Whisper / speech-to-text;
- whole-video VLM analysis;
- later media-derived representations.

Short clips with no content remaining after the initial skip are not treated as valid analyzable media.

## Scroll-detection failures

Whole-video VLM output can report:

```text
SCROLL: TRUE | FALSE
SCROLL_SECONDS: [...]
```

The known first-second transition must never itself count as an additional scroll.

`ep24_video.parse_scroll_metadata()` normalizes the model output and removes timestamps at or before the mandatory initial skip. `needs_resplit()` and `split_plan()` then provide a bounded deterministic route for later re-splitting.

The active pipeline preserves source provenance and legacy CSV/Pandas fields. Derived rows use additive re-split metadata rather than replacing source identifiers.

## CSC Allas / Roihu workflow

The source-media workflow remains:

1. obtain the Allas object/path from the EP24 dataframe;
2. download the original video from CSC Allas;
3. preserve that downloaded source unchanged;
4. analyze only the post-skip derived interval;
5. keep human-readable CSV/Pandas outputs and legacy-compatible fields.

See `docs/VLLM_VIDEO_TEST.md` for the isolated native-video Roihu experiment and `docs/EP24_VIDEO_SCROLL_ARTIFACTS.md` for the detailed scroll-artifact contract.

## Compatibility

No historical legacy stage, field, prompt, or source object should be removed merely to implement these rules. The `legacy` branch remains the historical research record.
