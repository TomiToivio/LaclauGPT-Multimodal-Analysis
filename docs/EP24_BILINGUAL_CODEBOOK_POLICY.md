# EP24 bilingual codebook policy

Tracking: #101.

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

## Measured effect of the policy (2026-10-02)

Across the ten EP24 books, measured with the canonical predicate:

| metric | value |
|---|---:|
| total entries | 31594 |
| entries requiring an English label | 4914 |
| entries missing one | 4914 |
| English-label coverage | 0% |
| fillable from an English alias already present | 134 |
| needing genuine translation work | 755 |

The 26 common-layer gaps are identical for every country (Orbán, Sánchez, Fidesz,
Rassemblement National, Les Républicains, Fianna Fáil, Sinn Féin, Moderaterna,
Vasemmistoliitto, `democracia`, `Demokratie`, ...), so repairing them benefits all
ten books at once. That is the highest-value backfill target.

## Known limitations of label-based classification

`label_looks_english` is a **review signal, not a verifier**. Residual error is
deliberately biased toward over-reporting: a redundant gloss costs one line, while
a missing translation costs silent retrieval failure. Three measured cases:

1. a **bare surname** (`Orbán`, `Sánchez`) is a single capitalised word, so the
   person-name exemption does not match and it is flagged — arguably correct, since
   a surname-only label still needs disambiguation;
2. a **hyphenated name with diacritics** (`Agnieszka Dziemianowicz-Bąk`) matches the
   person-name shape and is exempted, making the reported number a **lower bound**;
3. a label like **`Fidesz party`** is flagged via the organisation word list even
   though "Fidesz" is used as-is in English — harmless, since the gloss equals the
   label.

## One policy, one report

There is exactly **one** policy predicate: `english_label_required()`. The coverage
metrics, the strict gate and both CLI entry points derive from it, so they cannot
drift. This is deliberate: issue #116 was filed because two parallel predicates over
the same corpus reported 4914 and 889 for the same question, exposed as two tools
giving different answers. `bilingual_coverage_report()` and
`assert_bilingual_coverage()` remain as the aggregate report and the threshold gate,
but they now delegate to the canonical policy.
