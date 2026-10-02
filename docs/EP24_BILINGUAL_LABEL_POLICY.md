# EP24 bilingual label policy (`english_label`)

Issue: #101. Origin: the France QA pass (#91, PR #100), where FR reported
`missing_english_count = 650` — 100% of its entries.

## The rule

> An entry needs an `english_label` when its **canonical label is not already
> English**.

That is the whole policy. It is deliberately about the *label text*, not about
which language a source happened to be in, and not about which layer an entry
came from.

### Why not "the entry has a non-English source language"

The original gate was:

```python
entry.source_languages and any(lang != "en" for lang in entry.source_languages)
```

Two things are wrong with it, both measured on the real books:

1. **The `common` layer carries no language metadata.** `ep24_common_private.json`
   has no file-level `language` key and its 2694 entries have no per-entry
   `language`, so every common-layer entry gets `source_languages == []` and is
   **silently invisible** to the gate. The gate therefore sees only the country
   layer (711 entries for PL) and cannot report on the other 2694.
2. **Most labels are already English.** `Abortion`, `Accessibility`,
   `Accountability and responsibility` need no gloss. A language gate cannot tell
   that, so it both misses real gaps and flags non-gaps.

### Measured effect of the fix

Before, the metric reported **14.7%** of entries (4654 of 31594) as missing an
English label, and 0% English-label coverage everywhere. After:

| country | entries | needs English | missing | missing % | by layer |
|---|---:|---:|---:|---:|---|
| HU | 3194 | 146 | 146 | 4.6% | common 26, country 120 |
| SE | 3266 | 139 | 139 | 4.3% | common 26, country 113 |
| FR | 3344 | 117 | 117 | 3.5% | common 26, country 91 |
| PT | 3050 | 97 | 97 | 3.2% | common 26, country 71 |
| PL | 3405 | 91 | 91 | 2.7% | common 26, country 65 |
| ES | 3164 | 82 | 82 | 2.6% | common 26, country 56 |
| DE | 3098 | 69 | 69 | 2.2% | common 26, country 43 |
| FI | 3019 | 67 | 67 | 2.2% | common 26, country 41 |
| HR | 3044 | 63 | 63 | 2.1% | common 26, country 37 |
| BG | 3010 | 29 | 29 | 1.0% | common 26, country 3 |
| **total** | **31594** | **900** | **900** | **2.8%** | |

Two corrections in one change: the gate stopped missing the common layer, and it
stopped counting labels that are already English. `english_label` coverage is
genuinely **0%** across all ten books — no entry anywhere has the field set — but
only **900** entries actually need one, not 31594.

The **26 common-layer entries** are the same set for every country (Orbán,
Sánchez, Fidesz, Rassemblement National, Les Républicains, Fianna Fáil, Sinn
Féin, Moderaterna, Vasemmistoliitto, `democracia`, `Demokratie`, …). They are the
highest-value repair target, because one fix benefits all ten countries.

## Exemptions

These require **no** English label, and flagging them would manufacture work and
inflate the count:

| class | example | why |
|---|---|---|
| already English | `Abortion Rights` | the label is the English label |
| person name | `Pedro Sánchez` | the English form is the same string |
| handle | `@fundacjawosp` | language-neutral identifier |
| URL | `https://example.org` | language-neutral identifier |

Organisations are explicitly **not** exempted by the person-name shape test:
`Les Républicains`, `Fianna Fáil`, `Partido Socialista`, `Rassemblement National`
and `Sinn Féin` are two capitalised words but do need glosses.

Removing the person-name exemption was worth **643 entries** on the real books
(1543 flagged → 900) — almost all of them false positives.

## Known limitations, stated rather than hidden

The policy is a **review signal, not a verifier**. Residual imprecision is
reported here instead of being buried in a confident-looking percentage:

1. A **bare surname** (`Orbán`, `Sánchez`, `Höcke`) is one capitalised word, so
   the person-name exemption does not match and it is flagged. Arguably correct —
   a surname-only label has no distinct English form but does need disambiguation.
2. A person with a **hyphenated surname and diacritics**
   (`Agnieszka Dziemianowicz-Bąk`) matches the person-name shape and is exempted.
   Intended, but it makes the reported number a **lower bound**.
3. A label like **`Fidesz party`** is flagged via the organisation word list even
   though "Fidesz" is used as-is in English. Harmless (the gloss equals the
   label) but slightly inflates the count.

The direction of the residual error is deliberate: **over-reporting costs a
redundant gloss; under-reporting costs a silently missing translation.**

## Gate vs warning

`missing`/`missing_pct` are **reported by default and never block the pipeline**.
EP24 research must keep running while labels are repaired, and a missing gloss
must never silently substitute a country-fallback string.

A QA run can opt into a threshold:

```bash
python3 scripts/ep24/check_bilingual_coverage.py \
    --root "$LACLAUGPT_EP24_PRIVATE_ROOT" --max-missing-pct 5
```

Exit code is 1 and the message names the offending countries:

```text
FAIL: bilingual coverage below threshold 5.0% for: HU=4.6%, SE=4.3%
```

Suggested policy once repair begins: set `--max-missing-pct` to the current worst
country and ratchet it down. Start at 5, target 0 for the 26 common-layer entries
first, since that is one fix for ten countries.

## Repairing labels

Not done in this change — the metric, the report and the gate are. When adding
labels:

- populate from **authoritative public sources** (Wikipedia in both languages,
  official party/EP material) with provenance, and from private research material
  kept in `LaclauGPT-Private`;
- **preserve the local-language canonical label and aliases**; `english_label` is
  additive;
- **never collapse two distinct local-language entities** because their English
  translations happen to be similar, and never auto-merge an ambiguous entity;
- add to the `common` layer when the entity is not country-specific.

## Where this is implemented

| file | role |
|---|---|
| `roihu_codebooks.py` | `label_looks_english`, `entry_needs_english_label`, `bilingual_coverage_report`, `assert_bilingual_coverage`; `missing_english_count` now uses the policy |
| `scripts/ep24/check_bilingual_coverage.py` | human/JSON report + optional threshold gate |
| `tests/test_ep24_bilingual_coverage.py` | 17 tests, synthetic fixtures only |

`load_profile` still returns `missing_english_count` and now additionally
`missing_english_entry_ids`, so a caller can act on the specific gaps rather than
just the number.

## Related

- #91 — the country QA pass that found this for FR.
- #100 — the FR PR where it was documented.
- #72 — the country-codebook issue; bilingual coverage is one of its dimensions.
