# EP24 country × stage coverage (#306)

Database-side audit of the shared MongoDB frame, 2026-10-27, computed **per video**.
**Counts only** — no row content, quotes, transcripts or identifiers.

## The finding that changes the numbers

Coverage depends entirely on the unit of analysis, and the frame stores **one row per ingest run**,
not one row per video:

| country | ingest rows | distinct videos |
|---|---:|---:|
| Finland | 3,700 | 2,125 |
| Poland | 3,643 | 1,781 |
| Portugal | 1,946 | 960 |

At ingest level the summary stage looks half-empty in Poland and Portugal. At video level, taking the
most complete run of each video, it is **99.3 % and 100 %**. The same corpus, the same stage, and a
45-point difference produced by a denominator.

That is why this report gives every percentage with its row count, and why the primary figures below
are per video.

## Four states, kept apart

| state | meaning |
|---|---|
| **completed** | at least one contracted output column carries something |
| **not started** | no output and no status |
| **failed** | the stage's own status says so |
| **claimed but empty** | status says success, every output column is empty |
| **other** | `REPROCESS` / `DELETE` markers — honest non-results |

`claimed but empty` is invisible in any denominator built on `status = ok`. A stage's status, reason
and error columns never count as its own output, or a stage would look complete because it wrote its
failure reason.

## Finland — 2,125 videos

| stage | completed | not started | failed | other | claimed but empty | coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 1 preprocess | 2,125 | 0 | 0 | 0 | 0 | 100.0 % |
| 2 frame | 2,125 | 0 | 0 | 0 | 0 | 100.0 % |
| 3 video | 1,276 | 0 | 0 | 155 | **694** | 60.0 % |
| 4 summary | 1,038 | 0 | 0 | 205 | **882** | 48.8 % |
| 5 postprocess | 1,050 | 0 | 0 | 202 | 873 | 49.4 % |
| 6 discourse | 707 | 1,418 | 0 | 0 | 0 | 33.3 % |
| 7 DNA | 660 | 1,465 | 0 | 0 | 0 | 31.1 % |
| 8 SNA | 623 | 1,502 | 0 | 0 | 0 | 29.3 % |
| 9 RDF | — | — | — | — | — | not auditable |

## Poland — 1,781 videos

| stage | completed | not started | failed | other | claimed but empty | coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 1 preprocess | 1,781 | 0 | 0 | 0 | 0 | 100.0 % |
| 2 frame | 1,781 | 0 | 0 | 0 | 0 | 100.0 % |
| 3 video | 1,781 | 0 | 0 | 0 | 0 | 100.0 % |
| 4 summary | 1,769 | 0 | 0 | 1 | 11 | 99.3 % |
| 5 postprocess | 1,781 | 0 | 0 | 0 | 0 | 100.0 % |
| 6 discourse | 10 | 1,771 | 0 | 0 | 0 | **0.6 %** |
| 7 DNA | 0 | 1,781 | 0 | 0 | 0 | **0.0 %** |
| 8 SNA | 0 | 1,781 | 0 | 0 | 0 | **0.0 %** |
| 9 RDF | — | — | — | — | — | not auditable |

## Portugal — 960 videos

| stage | completed | not started | failed | other | claimed but empty | coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 1 preprocess | 960 | 0 | 0 | 0 | 0 | 100.0 % |
| 2 frame | 960 | 0 | 0 | 0 | 0 | 100.0 % |
| 3 video | 960 | 0 | 0 | 0 | 0 | 100.0 % |
| 4 summary | 960 | 0 | 0 | 0 | 0 | 100.0 % |
| 5 postprocess | 960 | 0 | 0 | 0 | 0 | 100.0 % |
| 6 discourse | 714 | 246 | 0 | 0 | 0 | **74.4 %** |
| 7 DNA | 25 | 935 | 0 | 0 | 0 | **2.6 %** |
| 8 SNA | 0 | 960 | 0 | 0 | 0 | **0.0 %** |
| 9 RDF | — | — | — | — | — | not auditable |

## Findings

**1. `status = OK` does not mean there is a result.** At ingest level, 1,906 Finnish, 1,551 Polish and
693 Portuguese rows carry a success status with every summary output empty. Verified directly rather
than through the metric: those Finnish rows have `processing_status = OK`, no summary text and no
`metadata` field at all — while 631 neighbours are correctly marked `REPROCESS` and 26 `DELETE`. The
pipeline knows how to say "not done"; it also says `OK`.

**2. Discourse stages are not comparable across countries.** Finland reaches 33.3 / 31.1 / 29.3 % at
video level. Poland reaches 0.6 / 0 / 0; Portugal 74.4 / 2.6 / 0. A cross-country Laclau, DNA or SNA
comparison today compares *different pipelines*, not different countries.

**3. The unit changes the headline, so state it.** Poland's summary coverage is 50.8 % per ingest row
and 99.3 % per video. Neither number is wrong; a number without its unit is.

**4. Language mapping is clean.** The only `asr_language` values are `fi`, `pl`, `pt`, each confined
to its own country.

**5. Stage 9 is not auditable from this frame** and is reported as such rather than as 0 %, which
would read as a pipeline failure.

## Cross-check against the export-side reporter

`scripts/ep24/report_stage_coverage.py` computes coverage from the exported CSV, independently of this
database-side audit. Where the two use the same definition they agree:

| stage | this audit (per video) | export-side reporter | verdict |
|---|---:|---:|---|
| FI 3 video | 60.0 % | 60.0 % | **agree** |
| FI 4 summary | 48.8 % | 37.0 % | **disagree** |

The disagreement is definitional, not arithmetic: the reporter counts a summary as present from its
status column plus a text field, while this audit counts any populated output column, including
`metadata`. Both are defensible; two definitions for one word are not. **One definition should be
chosen and written into the stage contract**, so the dashboard and the report cannot disagree about
the same corpus.

## What would fix it

* **A stage must not write `OK` without output.** That invariant belongs at the stage boundary and is
  testable there, not in a dashboard.
* **Rerun stages 6–8 for Poland and Portugal** before any cross-country discourse claim is published.
  This is an execution gap, not a code defect, and it is the largest limitation on the data.
* **Declare the unit everywhere** (`video` or `ingest`) next to every percentage, as `#205` requires of
  the exporters.

## Limits

One snapshot (2026-10-27); no reruns were executed, so no before/after is claimed. The success and
failure status vocabularies are declared in `ep24_coverage_audit.py`, not read from the pipeline;
values outside them land in `other` and are listed rather than guessed at.
