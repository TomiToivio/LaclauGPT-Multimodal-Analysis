# EP24 bilingual codebook policy — the *review-state* metric

Tracking: #101.

> **This document describes ONE of two bilingual metrics.** There are two, they
> answer different questions, and they legitimately disagree by ~8x on the same
> corpus. Read both before quoting either number:
>
> | question | function | report | this doc? |
> |---|---|---|---|
> | Has every non-English-sourced entry had its **review state** recorded, even when the English form is identical? | `english_label_required` | `scripts/ep24/english_label_coverage.py` | **yes** |
> | Does the label need an English **translation**? (label is not already English) | `english_translation_required` | `scripts/ep24/check_bilingual_coverage.py` | no — see `docs/EP24_BILINGUAL_LABEL_POLICY.md` |
>
> The two are kept as separate named functions on purpose (#116/#120). The
> review-state metric exists so that "verified identical" stays distinguishable
> from "not reviewed yet"; the translation metric exists so a researcher has a
> bounded repair list. Which one a workflow should gate on is a project decision,
> not something either document settles.

EP24 codebooks preserve the local-language canonical label. English is a linked
retrieval/translation form, never a reason to replace or merge the local canonical
entity.

## Policy

For every entry whose source language includes a language other than English,
`english_label` is required. If the normal English form is spelled identically
(for example many personal names), write that identical value explicitly. This
keeps "verified identical" distinct from "not reviewed yet".

An English label may be omitted only when the entry contains
`metadata.english_label_exempt_reason` with a concrete reason, for example a
language-neutral identifier that genuinely has no translation. Exemptions are
counted and remain auditable.

### Entries with no language metadata

An empty `source_languages` means **unknown**, not "English". The `common` layer
records no `language` at file or entry level, so deciding on `source_languages`
alone exempted those entries *by omission*: they were neither present, nor
missing, nor counted as exemptions — 2694 entries per country simply uncounted,
26 of which are genuinely non-English (`Rassemblement National`, `Sinn Féin`,
`Moderaterna`).

For an entry with no language metadata the **label decides**:

| case | example | required? |
|---|---|---|
| label already English | `Abortion`, `Accessibility` | no — nothing to translate |
| label non-English | `Rassemblement National`, `Sinn Féin` | **yes** |
| person name | `Pedro Sánchez` | no — the English form is the same string |
| handle or URL | `@fundacjawosp`, `https://…` | no — language-neutral |
| documented exemption | `metadata.english_label_exempt_reason` | no — and auditable |

Organisations are **not** exempted by the person-name shape test: `Les
Républicains`, `Fianna Fáil` and `Partido Socialista` are two capitalised words
but do require glosses.

Measured effect: required labels rise from 4654 to **4914** across the ten books
(+26 per country, exactly the previously invisible common-layer gaps), while 2668
already-English common labels stay correctly exempt. The heuristic is a review
signal, not a verifier; residual error is deliberately biased toward
over-reporting, because a redundant gloss costs one line while a missing
translation costs silent retrieval failure.

Similar English translations never justify collapsing two distinct local
entities. Identity remains country/kind/local-label scoped and aliases stay
attached to the correct canonical entry.

## Enforcement

The runtime loader now returns `english_label_coverage` and `qa_state`.
Missing required labels produce `REVIEW_REQUIRED` by default, so existing EP24
processing remains runnable while the backfill is reviewed.

For QA/CI, use a hard gate:

```bash
export LACLAUGPT_CODEBOOK_ENGLISH_STRICT=1
```

or run the dedicated all-country report:

```bash
python3 scripts/ep24/english_label_coverage.py \
  --root ../LaclauGPT-Private/analysis/ep24 --strict
```

Exit 0 means all required English labels are present; exit 1 means one or more
countries need review; exit 2 means the private input is unavailable/unreadable.

## Cross-country coverage

The command above audits every country in `COUNTRY_PROFILES` (BG, DE, ES, FI,
FR, HR, HU, PL, PT, SE) and prints entries, required/present/missing/exempt
counts, coverage percentage, and state. This is the authoritative coverage
report, generated from the private books without publishing private rows.

The repository connector sees the private JSON codebooks as Git LFS pointer
files, so this public change deliberately does not copy private codebook contents
or fabricate a static table. Run the report in a smudged private checkout
(Roihu/local workstation) and use the resulting aggregate numbers for review.

The existing country audits already establish two actionable baselines:
France has 650/650 missing and Spain 470/470 missing. Both now resolve to
`REVIEW_REQUIRED` in the actual loader instead of silently exposing an unused
counter.

## Backfill rules

Use authoritative public sources and existing private research material where
appropriate. Preserve the local canonical label and aliases. Add established
English names/translations to `english_label`; do not translate by blindly
machine-replacing canonical labels. Keep sensitive notes and provenance in
`LaclauGPT-Private`.


## Two distinct coverage metrics (#116)

There are two useful questions and they must not be presented as rival answers to
one ambiguous phrase such as "missing English labels":

1. **Explicit-English review state**: implemented by `english_label_required()`,
   `english_label_coverage()`, and
   `scripts/ep24/english_label_coverage.py`. A non-English-sourced entry must
   carry an explicit `english_label` (or a documented exemption), even when the
   English spelling is identical. This distinguishes "verified identical" from
   "not reviewed yet" and is the provenance/QA metric.

2. **English translation work**: implemented by
   `english_translation_required()`,
   `english_translation_coverage_report()`, and
   `scripts/ep24/check_bilingual_coverage.py`. It counts only entries whose
   canonical label needs a distinct English rendering. Personal names and
   already-English labels do not count as translation work.

For example, a Polish person named `Adam Bielan` is **review-required** until an
explicit English label or exemption is recorded, but needs **no translation**
because the English rendering is the same name. `Rassemblement National` needs
both review and translation. `Abortion` needs neither when it has no
non-English source metadata.

The loader exposes the explicit-review metric as `english_label_coverage` and
`qa_state`, because that is the safer operational QA signal. The translation
metric is advisory planning information for backfill work and can be gated
separately by its dedicated CLI. Compatibility aliases from #110 remain only as
delegating wrappers; there is one implementation per policy.
