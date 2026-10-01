# EP24 mobile video handling: the first-second rule and scroll re-splitting

This document describes how the EP24 videos were collected, the two video-quality
conditions that follow from that collection method, and how the pipeline handles
them. It is the reference for the mandatory first-second exclusion and for the
`SCROLL` / `SCROLL_SECONDS` fields.

## How the EP24 videos were collected

The EP24 material was gathered by researchers conducting **digital ethnography on
TikTok and Instagram during the 2024 European Parliament elections**. They
recorded the social-media feeds as continuous mobile screen video on
**GrapheneOS phones**.

Because the recording is continuous, the resulting files are not individual
posts. A **scroll-detection algorithm** later split each continuous recording
into per-item clips by detecting the scrolling transition between feed items.

That method has two consequences the rest of the pipeline must respect.

## 1. The first second of every clip is a transition, not content

The split boundary is placed at the scroll, so **the first second of each
resulting clip still contains the scrolling transition from the previous
social-media item**. It is an artefact of the splitting, not part of the item
being analysed.

**Every analysis path therefore ignores the first 1.0 second of every clip** and
analyses the interval `[1.0, original_end]`.

The skip is defined **once**, in `video_config.py`:

```python
VIDEO_INITIAL_SKIP_SECONDS = 1.0
```

Do not reintroduce a literal `1.0` or a bare `range(0, ...)` in a stage script.
Import from `video_config` so a change to the rule changes every consumer at
once:

| Helper | Purpose |
| --- | --- |
| `initial_skip_seconds()` | the effective skip (env-overridable) |
| `analysis_interval(duration)` | `(start, end)` of the analysed interval |
| `analysis_start_seconds(duration)` | just the start |
| `is_analysable(duration)` | whether the clip holds any analysed content |
| `sample_offsets_seconds(...)` | frame offsets inside the interval |
| `audio_extract_command(...)` | ffmpeg command seeking past the skip |
| `scroll_rule_text()` | the shared prompt/parser rule |

Override for testing or a special clip:

```bash
export LACLAUGPT_VIDEO_INITIAL_SKIP_SECONDS=1.0
```

The **original source video is never modified**. The skip is applied by the
reader: the frame sampler starts after it, the audio extractor seeks past it, and
the VLM prompt states it.

### Where the skip applies

| Analysis path | How the skip is applied |
| --- | --- |
| Whisper / speech-to-text | audio extracted with `ffmpeg -ss 1.0` before transcription |
| audio extraction / analysis | same extracted interval |
| frame extraction | sample offsets start at the skip, not at 0 s |
| OCR | runs over the extracted frames, so inherits the skip |
| frame / image analysis | runs over the extracted frames |
| VLM / video-language models | interval in the prompt; media read from the skip |
| embeddings from media | derived from the skipped frame/audio inputs |
| scene sampling | uses `sample_offsets_seconds` |
| summaries | consume the post-skip transcript and frames |
| multimodal fusion | combines post-skip inputs |
| future video modules | must import `video_config` |

Before this change, `roihu_preprocess.py` sampled keyframes from
`range(0, duration, 30)`, so the **first keyframe was the transition frame**, and
ASR transcribed the raw file including the transition. Both now start at the skip.

## 2. Scroll-detection failures and the `SCROLL` fields

The scroll-detection splitter occasionally fails, leaving a clip that contains
**more than one** TikTok/Instagram item separated by additional scrolling
transitions. The VLM analysis step reports this.

### Fields

```text
SCROLL: TRUE | FALSE
SCROLL_SECONDS: [...]
```

- `SCROLL = FALSE` — no additional scroll transition after the mandatory
  first-second artefact; the clip continues normally. `SCROLL_SECONDS` is `[]`.
- `SCROLL = TRUE` — one or more further feed-scroll transitions occur inside the
  clip; `SCROLL_SECONDS` holds their approximate timestamps.

```json
{ "SCROLL": true, "SCROLL_SECONDS": [8.4, 17.9] }
```

**The known first-second transition can never set `SCROLL = TRUE`.** Timestamps
at or below the skip are stripped when the report is parsed, so the artifact
cannot masquerade as a genuine boundary.

Parsing is deliberately fail-safe (`video_scroll.parse_scroll_report`):

- a missing or unparseable block is `SCROLL = FALSE` with an empty list, so an
  unhelpful model cannot flag every clip for re-splitting;
- near-duplicate timestamps are collapsed, out-of-range ones dropped;
- `SCROLL = TRUE` with no usable timestamps sets `flagged_without_timestamps`
  and produces a `needs_resplit` state rather than a guessed split.

### VLM prompt

`scroll_rule_text()` instructs the model to ignore the first second, inspect the
remainder for feed scrolling, distinguish motion inside a post from an actual
scroll, and report the two fields. Existing legacy fields and the human-readable
Markdown output are unchanged.

## 3. Re-splitting

When `SCROLL = TRUE`, the clip is flagged for re-splitting. `video_scroll.plan_resplit`
turns the report into a deterministic plan:

- segment boundaries come from `SCROLL_SECONDS`; each segment runs from the
  previous cut to the next, so **no frame is lost at a cut** (the boundary ends
  one segment and starts the next);
- every segment also begins after the skip, so derived clips obey the same rule;
- derived names are deterministic (`<stem>_scrollNN_<start>_<end>_<hash>.mp4`), so
  a re-run cannot create a second, differently-named set;
- the source clip name is carried in the derived names and is never overwritten;
- provenance from the source row and original clip is preserved in the plan.

### Bounds that make an infinite loop impossible

| Guard | Effect |
| --- | --- |
| `depth >= MAX_RESPLIT_DEPTH` (3) | a plan at the depth limit is **refused** |
| `MAX_SCROLL_BOUNDARIES` (12) | more boundaries than this is a model artefact; refused |
| boundaries outside the interval | nothing to split; `needs_resplit` for review |
| clip not analysable | never split |

Where automation is not provably safe the code emits the explicit
**`needs_resplit`** state (with a reason) and leaves the next step deterministic
and documented, rather than guessing at a boundary.

## CSC Allas / Roihu workflow

Video handling stays simple and unchanged in shape:

1. take the source URL / video reference from the EP24 dataframe;
2. download the video from **CSC Allas** using the configured Allas environment
   on CSC Roihu;
3. analyse it with the normal multimodal pipeline, applying the skip rules above;
4. keep the existing CSV/Pandas workflow and legacy fields;
5. use MongoDB/Redis only where already specified for memory/RAG/backup —
   video storage is **not** redesigned around a new database layer here.

## Compatibility

No legacy behaviour is removed: no stage removed, no legacy dataframe field
renamed or dropped, human-readable Markdown summaries preserved, CSV/Pandas
contracts preserved, source URLs and provenance preserved, and the `legacy`
branch untouched. The new handling is added around the existing pipeline.

## Related

- `docs/LEGACY_PIPELINE_CONTRACT.md` — stage and field contracts.
- `docs/ROIHU_MIGRATION.md` — Roihu architecture and public/private boundary.
- `docs/VLLM_VIDEO_TEST.md` — the standalone native-video smoke test, which
  applies the same skip rule.
- `tests/test_video_scroll_and_skip.py` — the tests for this behaviour.
