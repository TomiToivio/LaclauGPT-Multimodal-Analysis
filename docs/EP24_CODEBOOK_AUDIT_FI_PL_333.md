# EP24 codebook audit — Finland and Poland (issue #333)

**Date:** 2026-10-31 · **Scope:** the FI and PL country codebooks available in the private analysis
checkout, audited with `scripts/ep24/codebook_coverage.py`. **Privacy:** this document contains only
aggregate counts. No labels, aliases, actors, quotes, record ids or private paths are reproduced.

## Method

```bash
python3 scripts/ep24/codebook_coverage.py --root <private>/analysis/ep24/codebooks --country FI
python3 scripts/ep24/codebook_coverage.py --root <private>/analysis/ep24/codebooks --country PL
```

The auditor is read-only. It reports, per country: entry counts, alias coverage, kind breakdown,
entity fragmentation groups and near-duplicate themes. It never modifies a codebook or resolves an
ambiguity automatically.

The codebooks are Git LFS objects; the pointers were hydrated locally for the audit. Nothing from
them is committed here.

## Results

| Metric | FI | PL |
|---|---:|---:|
| Entries | 325 | 711 |
| Entries without aliases | 272 (83.7 %) | 649 (91.3 %) |
| Aliases total / per entry | 74 / 0.23 | 76 / 0.11 |
| Entities / themes / topics | 275 / 41 / 9 | 576 / 122 / 13 |
| Fragmented entity groups | 36 | 98 |
| Near-duplicate theme pairs | 2 | 17 |

Both codebooks are **alias-poor**: most entries can only be matched by their exact label, which is
the main driver of the fragmentation counts. PL is worse on every ratio despite having twice the
entries.

## Findings

1. **Alias coverage is the structural defect, not the entry count.** FI leaves 83.7 % of entries
   without an alias and PL 91.3 %. A label-only match cannot absorb spelling, language and
   institutional variants that the corpus demonstrably uses.
2. **Fragmentation is concentrated.** FI has 36 fragmented groups and PL 98, but the top groups
   account for a large share of observations. A small, reviewed alias pass on those groups buys more
   matching than a broad rewrite.
3. **Theme duplication is manageable in FI and significant in PL.** FI has two near-duplicate theme
   pairs; PL has seventeen, including exact-normalisation pairs (case/spacing) that are safe to fold
   and near pairs that need a researcher decision.
4. **The public cross-country table is stale for FI/PL.** `docs/EP24_CODEBOOK_COVERAGE_CROSS_COUNTRY.md`
   reports FI 83.7 % / 36 groups and PL 91.3 % / 98 groups, which matches this run. No drift was
   found, but the table only stays honest while `check_coverage_table.py` is run against the private
   codebooks; it cannot be reproduced from the public repository alone.
5. **Portugal has no codebook in this checkout.** PT cannot be audited here, so the three-country
   codebook comparison the issue asks for is incomplete; this is a data-availability gap, not a
   finding about PT.

## Recommended edits (for researcher review, not applied)

- FI: alias pass on the 36 fragmented groups, highest-observation first; fold the two theme pairs.
- PL: alias pass on the 98 groups; fold exact-normalisation theme pairs automatically and queue the
  semantic near-duplicates (EU relations, government criticism) for a researcher decision.
- Do not merge entities that are genuinely distinct actors or coalitions; the auditor reports
  ambiguity rather than resolving it.
- Add PT codebook coverage once the file is available.

## Not done

- No codebook was modified. These are recommendations requiring researcher review.
- No blinded model/prompt comparison was possible: that needs the private corpus and model runs.
- No live Steps 0–9 rerun; see #342 for the pipeline-side gate.
