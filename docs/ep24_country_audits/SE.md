# EP24 country audit: Sweden (SE)

Status: **completed first-agent QA pass for issue #91**.

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
