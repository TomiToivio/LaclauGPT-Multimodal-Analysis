# EP24 cross-country codebook coverage — all 10 countries

Tracking: issue #72. Produced by `scripts/ep24/codebook_coverage.py`, the auditor
added in #81. Every number below is reproducible with one command per country.

This is the cross-country view that a single-country pass cannot give: it shows
whether the Croatia findings were a local quirk or a systemic property of the
codebook layer.

## Method

```bash
python3 scripts/ep24/codebook_coverage.py --root <LaclauGPT-Private>/analysis/ep24/codebooks --country <ISO2>
```

Read-only. The auditor resolves each country's filename through the repo's own
`COUNTRY_PROFILES` map, because the filename is **not** derivable from the ISO2
code (Finland is `ep24_finland_private.json`, Poland `ep24_poland_private.json`).

> **Table refreshed 2026-10-02 (Sweden second pass, #91).** The static columns
> (entries, aliases, theme near-dups) are unchanged, but **every fragmentation
> count moved**, and the total went 599 → 623. The table was accurate as
> published: all ten rows reproduce exactly against the auditor as it stood in
> #86. It then went stale because #100 wired `fold_fix` into `_fold`, which
> changed what counts as one token — so the fragmentation column silently drifted
> while the doc kept promising reproducibility. `scripts/ep24/check_coverage_table.py`
> now re-runs the auditor and fails if this table no longer matches, so the next
> change to the auditor surfaces here instead of in a country pass months later.


The `countries/*.json` public-context seeds are excluded from this table: they are
Git LFS pointers in a normal checkout and hold 2–3 entries each, so they are
containers rather than codebooks. The auditor reports them as `UNREADABLE`
(with the `git lfs pull` hint) instead of crashing.

## The table

| Country | Entries | Without aliases | % | Aliases | Alias/entry | Fragmented groups | Theme near-dups |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PL | 711 | 649 | 91.3% | 76 | 0.11 | 98 | 17 |
| HU | 500 | 454 | 90.8% | 53 | 0.11 | 78 | 18 |
| SE | 572 | 514 | 89.9% | 83 | 0.15 | 64 | 1 |
| FR | 650 | 582 | 89.5% | 93 | 0.14 | 95 | 8 |
| BG | 316 | 271 | 85.8% | 91 | 0.29 | 37 | 5 |
| PT | 356 | 303 | 85.1% | 77 | 0.22 | 67 | 1 |
| ES | 470 | 398 | 84.7% | 104 | 0.22 | 64 | 3 |
| FI | 325 | 272 | 83.7% | 74 | 0.23 | 36 | 2 |
| DE | 404 | 337 | 83.4% | 83 | 0.21 | 47 | 7 |
| HR | 350 | 280 | 80.0% | 108 | 0.31 | 37 | 3 |
| **Total** | **4654** | **4060** | **87%** | 842 | 0.18 | 623 | 65 |

## What this establishes

### 1. Alias poverty is systemic, not a Croatian quirk

**87% of 4654 entries across 10 countries carry no aliases at all.** The best
country (Croatia, 80%) still leaves four in five entries matchable only by their
exact label. This is the single largest structural constraint on the codebook
layer, and it is uniform across languages and party systems.

The corollary is the important part: **new entries are not the bottleneck.** Every
country has hundreds of entries. What is missing is the *forms* — the acronyms,
inflected forms, English/local pairs and variants that let a corpus mention reach
an existing entry.

### 2. Entity fragmentation is proportional to entry count, not to language

`frag` scales with `entries`: PL 98, FR 95, HU 78 on the large books; FI 36, BG 37
on the small ones. That is the signature of a **structural** cause — entries being
created per observed surface form rather than a canonical entity being extended
with forms.

Poland is the extreme: **98 fragmented groups across 711 entries.**

### 3. Theme duplication is uneven, and the unevenness is a signal

`tdup` is not proportional to size: HU has 18 across 500 entries while SE has 1
across 572, and PT 1 across 356. If duplication were purely a size effect the
numbers would track entry count. They do not, which means some country books
passed through a normalized theme vocabulary and others did not. Worth a
maintainer look: is there a canonical EP24 theme list that some books were built
against?

### 4. The alias/entry ratio splits the countries into two groups

| Ratio | Countries |
| --- | --- |
| ≈ 0.11–0.15 | PL, HU, SE, FR |
| ≈ 0.21–0.31 | BG, PT, ES, FI, DE, HR |

The low group is the four largest books. That suggests aliasing was applied
per-entry as a manual step and simply did not scale with book size — again
pointing at automation rather than more hand work.

## Consequences for the parent issue

- The "usable bilingual/multilingual codebook for every country" progress clause
  is **not** met on the current books, and the gap is measurable rather than
  impressionistic: 4060 entries need at least one alternate form.
- The priority order for a backfill is mechanical: largest `no-alias` count first,
  within the lowest alias/entry ratio — i.e. **PL, HU, FR, SE**.
- A rule-level fix (R3/R4: variants are aliases, not entities) prevents the
  fragmentation from growing while the backfill runs. That rule is pinned by
  `tests/test_ep24_country_codebook_normalization.py` (#78).

## Limitations — read before quoting these numbers

- **These are counts, not quality judgements.** An entry with no aliases may be
  perfectly correct; it is merely unmatchable by anything but its exact label.
- **The fragmentation heuristic groups on the trailing name token.** It will
  over-group languages with different name orders and under-group names that
  differ in the middle. Treat `frag` as an indicator, not a merge list. The
  auditor never merges; ambiguity is reported, never resolved.
- **The theme-duplicate threshold is a single Jaccard value (0.6)** applied across
  ten languages. Croatian or Hungarian compounding may need a different
  threshold — flagged as an open question rather than settled.
- The books are private and were read locally; only aggregates appear here. No
  rows, notes or personal data are published.

## Reproduce

```bash
for c in BG DE ES FI FR HR HU PL PT SE; do
  python3 scripts/ep24/codebook_coverage.py --root "$PRIVATE_ROOT/analysis/ep24/codebooks" --country "$c"
done
```
