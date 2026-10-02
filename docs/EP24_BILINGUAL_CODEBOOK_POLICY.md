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
