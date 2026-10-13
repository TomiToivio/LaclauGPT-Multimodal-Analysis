# EP24 country audit: Sweden (SE)

Status: **completed first-agent QA pass for issue #91, plus an independent second pass**
(2026-10-02). The second pass is appended below under "Second pass"; it closed the
first pass's private-data gap and challenges some of its framing.

This document records public, non-sensitive findings only. Operational codebooks, researcher notes, and legacy rows remain in `TomiToivio/LaclauGPT-Private`.

## Runtime/settings checked

Public runtime profile:

- pipeline country code: `SE`
- country: Sweden
- configured languages: `sv`, `en`
- private country codebook: `ep24_se_private.json`
- canonical output translation language: English through the shared EP24 pipeline
- country identity is explicit and is not inferred from language

The public profile and the private builder agree on Sweden = `SE` / Swedish = `sv`. The private builder consumes `ep24_sv.csv` and writes `ep24_se_private.json`, so the language-oriented legacy filename and country-oriented runtime filename are intentionally different rather than inconsistent.

## Private material inspected

Available through the GitHub connector:

- `analysis/ep24/codebooks/ep24_se_private.json` — tracked through Git LFS; pointer verified (payload itself is not exposed by the connector)
- `analysis/ep24/codebooks/research_staging/SE.public_context.json` — tracked through Git LFS; pointer verified
- `analysis/ep24/codebooks/build_all_country_codebooks.py`
- `analysis/ep24/codebooks/research_staging/README.md`
- `analysis/ep24_reprocess/data/by_country/ep24_sweden_cleaning_report.md`

The Sweden cleaning report records 2,678 source rows, 2,591 rows retained for reprocessing, 59 explicit researcher deletes, 3 explicit dubious exclusions, 25 recut/split rows, and 10 rows retaining non-trivial researcher notes for review/context.

Because the large codebook/data payloads are Git LFS objects, this pass does **not** claim row-by-row inspection of their private contents through this connector. A later agent with a checked-out LFS working tree should do that deeper pass.

## Public-source verification

Checked in Swedish and English:

- Swedish Election Authority, Swedish results: https://www.val.se/valresultat-och-statistik/eu-val/valresultat-2024
- Swedish Election Authority, English results: https://www.val.se/english/election-results/elections-to-the-european-parliament/european-parliament-election-results-2024
- Swedish Election Authority, final 2024 result announcement: https://www.val.se/servicelankar/servicelankar/pressrum/nyheter--pressmeddelanden/pressmeddelande-nya/2024-06-14-valresultat-faststallt-i-valet-till-europaparlamentet
- Swedish Election Authority, elected MEPs: https://www.val.se/servicelankar/servicelankar/pressrum/nyheter--pressmeddelanden/pressmeddelande-nya/2024-06-14-valda-ledamoter-till-europaparlamentet
- Swedish Wikipedia election page: https://sv.wikipedia.org/wiki/Europaparlamentsvalet_i_Sverige_2024
- English Wikipedia election page: https://en.wikipedia.org/wiki/2024_European_Parliament_election_in_Sweden

The official result provides a useful canonical cross-check for the eight represented parties and their Swedish names:

- Arbetarepartiet-Socialdemokraterna
- Centerpartiet
- Kristdemokraterna
- Liberalerna
- Miljöpartiet de gröna
- Moderaterna
- Sverigedemokraterna
- Vänsterpartiet

The codebook should retain common shorter forms and English names as aliases rather than replacing these Swedish identities.

## Language and normalization findings

Swedish diacritics are analytically meaningful. `Å/Ä/Ö` must be preserved in canonical labels. The shared identity key already does this correctly by using Unicode NFC + casefold + whitespace collapse without ASCII folding.

English party names should be aliases / English labels, not duplicate canonical entities. For example, `Sverigedemokraterna` and `Sweden Democrats` should point to the same country-scoped entity.

The most important Sweden-specific gap is **short party abbreviations**. Swedish political discourse routinely uses forms such as `S`, `M`, `V`, `C`, `L`, `KD`, `MP`, and `SD`. The private codebook builder intentionally drops forms shorter than five characters because the historical retrieval matcher used unsafe substring matching. That protects precision, but leaves important Swedish aliases unavailable.

This is not safely fixable by loosening the identity key or globally allowing tiny substrings. Shared architectural follow-up: **#95**.

## Memory / RAG / evidence firewall

The shared runtime already provides the correct safety properties:

- codebook entries are country-scoped;
- aliases can be language/country scoped;
- canonical Memory resolution abstains rather than fuzzily merging ambiguous identities;
- background relations/context remain background and must not become observed evidence;
- strict boundary matching exists for identity/coverage checks.

For Sweden, retrieval should prioritize exact canonical/alias hits in `SE`, then use semantic retrieval. Short aliases should wait for #95 rather than being injected into the current substring scorer.

## Prompt/stage review

No Sweden-specific full prompt fork is justified. Useful targeted context is:

- `SE` and `sv` identification;
- bilingual entity aliases;
- election-specific party/list/candidate context;
- Swedish diacritic preservation;
- explicit instruction that English translations do not create a second entity;
- the shared background-context evidence firewall.

OCR/ASR should preserve Swedish Unicode and must not automatically convert `å/ä/ö` to ASCII forms. ASCII/OCR variants may be reviewed aliases when observed empirically, never replacements for canonical spelling.

## Fixes made in this pass

Public regression tests were added for:

1. preserving Swedish diacritics in identity normalization;
2. keeping English/local-language forms under one entry;
3. keeping Sweden aliases country-scoped;
4. documenting that short aliases need boundary-aware retrieval rather than unsafe substring matching.

The shared architectural gap for safe short abbreviations was split into #95.

## Remaining uncertainty / requested re-check

A second agent with a local checkout that has Git LFS materialized should inspect the full `ep24_se_private.json`, `SE.public_context.json`, and real Sweden legacy rows, measuring:

- alias coverage for the eight represented parties and 2024 candidates;
- whether official full Swedish names, common names, English names, handles and hashtags converge on one canonical entity;
- actual OCR/ASR variants in the corpus;
- theme duplication and recurring researcher corrections.

That deeper private-data pass is the main remaining Sweden-specific uncertainty.

---

## Second pass — scope

Public: measurements, code, tests, public-source provenance, aggregates. Private: everything else. No private row values, cells or researcher text appear below.

| | |
| --- | --- |
| ISO / pipeline code | `SE` |
| campaign language | Swedish (`sv`) |
| retrieval companion | English (`en`) |
| election | 2024 European Parliament election in Sweden |
| seats | 21 |

### Public sources

1. English Wikipedia, 2024 European Parliament election in Sweden — https://en.wikipedia.org/wiki/2024_European_Parliament_election_in_Sweden
2. Swedish Wikipedia, Europaparlamentsvalet i Sverige 2024 — https://sv.wikipedia.org/wiki/Europaparlamentsvalet_i_Sverige_2024
3. European Parliament results portal — https://results.elections.europa.eu/
4. Valmyndigheten (Swedish Election Authority) — the official national source

### Confirmed: the first pass's LFS finding is correct

The first pass's central claim holds. In this checkout `countries/se.json` is readable (2 entries), and the per-country payload `ep24_se_private.json` is materialized and loadable through the runtime. So the first pass was right that it could not read the payload, and the second pass now can.

The declared profile is internally consistent:

| Layer | country_code | country | languages |
| --- | --- | --- | --- |
| `COUNTRY_PROFILES['SE']` | — | Sweden | `sv`, `en` |
| `ep24_se_private.json` | `SE` | Sweden | `sv` |
| `countries/se.json` | `SE` | Sweden | `sv` |

One minor divergence: `countries/se.json` records `languages: ["sv"]` while `COUNTRY_PROFILES` records `["sv", "en"]`. Not a defect — the seed layer lists campaign languages and the profile adds the retrieval companion — but it is a second place where a language list lives, and nothing checks the two agree.

### Measured codebook state

`ep24_se_private.json`: **572 entries**, 476 `entity` / 83 `theme` / 13 `topic`; **514 (89.9%) carry no aliases**, 83 aliases total (0.15/entry); 50 entries (8.7%) carry zero observations.

### Cross-country configuration consistency

The issue asks to *"look for inconsistencies between different scripts/modules"*. This is the check nobody had done, and it holds — with one documented exception.

Every **active** `roihu_*` stage declares the same 11-language list (`fi sv pl pt de es hu hr fr bg en`), which exactly equals the union of `COUNTRY_PROFILES[*].languages` plus the `en` companion. Checked: `roihu_preprocess`, `roihu_frame`, `roihu_summary`, `roihu_postprocess`, `roihu_rdf`, and the `countries` lists in `roihu_populism`. **No drift.**

The legacy `puhti_*` modules still carry a 10-language list missing `bg`. That is **correct and must not be "fixed"**: `AGENTS.md` makes the legacy branch and its compatibility surface immutable, and the `puhti_*` files are the frozen historical record.

### Finding S1 — a contradictory party gloss in the SE codebook

`Socialdemokraterna (Sweden Democrats)` exists as a label. It joins the **Social Democrats (S)** to the **Sweden Democrats (SD)** — two different parties. This is not a paraphrase problem: the gloss is a match target, so a query for one party retrieves the other's entry.

Demonstrated, not asserted:

```
q='Sweden Democrats'  ->  [... 'Socialdemokraterna (Sweden Democrats)', 'Sverigedemokraterna (Sweden Democrats)', ...]
```

The Social Democrats entry is returned for a Sweden Democrats query.

**Blast radius, measured.** Labels of the `Native (Gloss)` shape are the exposure surface:

| Book | entries | `Native (Gloss)` | % |
| --- | --- | --- | --- |
| **SE** | 572 | **121** | **21.2%** |
| DE | 404 | 63 | 15.6% |
| PL | 711 | 94 | 13.2% |
| PT | 356 | 46 | 12.9% |
| HU | 500 | 53 | 10.6% |
| FI | 325 | 33 | 10.2% |
| HR | 350 | 34 | 9.7% |
| BG | 316 | 26 | 8.2% |
| FR | 650 | 51 | 7.8% |
| ES | 470 | 34 | 7.2% |
| **Total** | 4654 | 555 | 11.9% |

SE is the most exposed book in the corpus at 21.2%, roughly double the average — which is why this defect surfaced here. **Exactly one** contradiction was found in SE's 572 entries, so the error rate is low; the exposure surface, not the error rate, is the risk.

### Tooling added

`scripts/ep24/check_party_gloss_consistency.py` reports contradictions for one country against a party table derived from that country's public sources (SE table: `scripts/ep24/data/ep24_parties_se.json`). Read-only: it reports and never rewrites, because resolving which party a mention means is an author decision. 16 tests, each verified to fire on the real defect.

**The other nine countries are not covered yet** — each needs a party table from its own public sources, and inventing one would be exactly the guessing this issue forbids. SE is done; the rest are the obvious next step.

### Finding S2 — the cross-country coverage table had gone stale

`docs/EP24_CODEBOOK_COVERAGE_CROSS_COUNTRY.md` promises: *"Every number below is reproducible with one command per country."*

**Every fragmentation count in it was wrong.** The static columns (entries, aliases, theme near-dups) are exact, but the `Fragmented groups` column had drifted for all ten countries — total **599 → 623**.

This is not a careless error. Reproduced against the auditor **as it stood in #86**, all ten rows match *exactly*: the table was correct when published. It went stale because #100 wired `fold_fix` into `_fold`, changing what counts as one token. The doc kept promising reproducibility while the numbers moved underneath it.

A promise of reproducibility that nothing checks is worse than no promise: it makes a stale number look authoritative. So the promise is now a test — `scripts/ep24/check_coverage_table.py` re-runs the auditor and fails with a per-country diff if the table no longer matches. The table is refreshed and the prose that cited the old numbers is corrected. 12 tests.

### Findings that were checked and are NOT defects

Recorded so the next agent does not re-derive them:

- **`sv` is claimed by both FI and SE.** Checked whether this contaminates retrieval: it does not. `select_context` scopes by country first (`entry.country in {"", "COMMON", country.upper()}`), so a SE-scoped call returns only SE/COMMON and an FI-scoped call only FI/COMMON. Verified empirically: zero cross-country entries in either.
- **A SE-loaded entry list queried as FI returns COMMON entries.** Correct: the shared book is country-neutral, and scoping excludes the SE layer.
- **`Socialdemokraterna` in Swedish does not match the Finnish book.** Correct: that is the Swedish party's name; the Finnish one is SDP.

My own first runs produced false positives here — a naive gloss check flagged `Swedish Green Party` as contradictory for `Miljöpartiet`, and the party table's `_note`/`_election` metadata keys were read as party stems. Both are now pinned by tests.

### Remaining uncertainties

1. **A party table for each of the other nine countries** — the biggest open item, and the only way to know whether S1 is unique to SE.
2. The `countries/*.json` vs `COUNTRY_PROFILES` language-list duplication has no guard.
3. Whether the `new_entity` pruning behaviour found in the Finland pass (74.7% removal, 0% addition, organisations stripped) holds for SE — not measured here.
4. `#95` (short country-scoped abbreviations) remains open and is not duplicated by this pass.

### Status

Sweden is **not mature**. This pass closed the first pass's private-data gap, confirmed its LFS finding, added the cross-country configuration check, and fixed one identity defect plus one stale published table. Per #91, the country has a recorded pass; it does not have a second independent one that challenges *these* conclusions.

#91 stays open.
