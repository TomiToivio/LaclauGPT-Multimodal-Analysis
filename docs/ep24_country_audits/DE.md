# EP24 country audit: Germany (DE)

Status: **first-agent QA pass** for issue #91.

## Scope

This public audit records public-source verification, runtime/profile checks, normalization rules, and synthetic regression tests. Private EP24 material remains in `TomiToivio/LaclauGPT-Private`.

Country profile checked:
- pipeline code: `DE`
- languages: German (`de`) + English (`en`)
- private codebook file expected by runtime: `ep24_de_private.json`
- election: 2024 European Parliament election in Germany
- election date: 9 June 2024
- seats: 96

## Public sources checked

German and English official sources were checked:
1. Bundeswahlleiterin, final 2024 European election result:
   https://www.bundeswahlleiterin.de/europawahlen/2024/ergebnisse/bund-99.html
2. Bundeswahlleiterin, English final-result release:
   https://www.bundeswahlleiterin.de/en/info/presse/mitteilungen/europawahl-2024/42_24_endgueltiges-ergebnis.html
3. European Parliament Germany 2024 country sheet:
   https://www.europarl.europa.eu/news/en/press-room/20240527BKG21656/european-elections-2024-country-sheets/10/germany
4. German-language Wikipedia and English-language Wikipedia election pages should remain secondary orientation sources, not the authority for canonical results.

The official result names provide a useful canonical baseline for major 2024 actors, including CDU, AfD, SPD, BÜNDNIS 90/DIE GRÜNEN, CSU, BSW, FDP, DIE LINKE, FREIE WÄHLER, Volt, Die PARTEI, Tierschutzpartei, ÖDP, FAMILIE and others.

## Runtime/profile QA

`roihu_codebooks.COUNTRY_PROFILES["DE"]` is internally consistent:
- country = Germany
- languages = `["de", "en"]`
- private file = `ep24_de_private.json`

The loader keeps Unicode NFC, casefolds only for identity comparison, preserves punctuation/diacritics in stored labels, and scopes context retrieval by country. These are the right defaults for German.

## Germany-specific normalization rules

### 1. Preserve umlauts and punctuation in canonical labels

Canonical labels should preserve forms such as:
- `BÜNDNIS 90/DIE GRÜNEN`
- `FREIE WÄHLER`
- `ÖDP`

Do not destructively rewrite `Ü/Ö/Ä/ß` in canonical identity. ASCII/transliterated forms such as `Gruene` or `Freie Waehler` may be useful observed aliases, but should not replace the canonical label.

### 2. Keep CDU and CSU separate

`CDU`, `CSU`, and the political shorthand `CDU/CSU` are not interchangeable canonical entities. Germany's EP2024 context needs explicit relations rather than collapsing the two parties into a single actor.

A mention of `CDU/CSU` can retrieve background context for both, but identity resolution should not silently rewrite it to CDU or CSU.

### 3. Treat acronyms as country-scoped aliases

Important short forms include `CDU`, `CSU`, `SPD`, `AfD`, `BSW`, `FDP`, `ÖDP` and others.

They must survive codebook building and merge steps. The existing cross-country audit has already shown that the old minimum-length guards can delete exactly these forms. Runtime matching should remain country-scoped and conservative.

### 4. Store official and common Greens forms together without flattening evidence

The official result uses `GRÜNE (BÜNDNIS 90/DIE GRÜNEN)`. Useful aliases can include `GRÜNE`, `Die Grünen`, and common ASCII social-media variants such as `Gruene` when actually observed.

The current post/video surface form should remain evidence. English labels such as `Alliance 90/The Greens` are context/translation, not replacements for German source text.

### 5. BSW needs temporal context

`BSW` / `Bündnis Sahra Wagenknecht - Vernunft und Gerechtigkeit` was new in the 2024 result compared with 2019. Relations, aliases, and descriptions should therefore be election/time scoped rather than projected backwards.

### 6. Person, party and EU-group identities stay separate

Candidate or leader names may retrieve party context, but must not be normalized into the party itself. Likewise, a German party and its European Parliament group are separate entities connected by a sourced, time-scoped relation.

## Memory/RAG checks

Germany context should be injected only when useful:
1. filter by country `DE`;
2. prefer exact canonical or alias hits;
3. use German + English expansion for retrieval only;
4. preserve election/time scope for 2024 alliances and memberships;
5. abstain on ambiguous short forms outside country/kind context;
6. repeat the evidence firewall: background context can disambiguate evidence present in the item, but cannot create evidence absent from the item.

## Private material check

The expected private codebook path exists in GitHub, but through this connector it resolves to a Git LFS pointer rather than the materialized 1.1 MB JSON object. Therefore I **did not claim row-level/private-codebook inspection**.

Checked:
- repository/path existence: yes
- materialized private JSON contents: no, LFS pointer only
- private legacy rows/researcher corrections: no

A later agent running in a checkout with LFS objects materialized should measure:
- missing aliases for CDU/CSU/SPD/AfD/BSW/FDP/GRÜNE/ÖDP;
- ASCII umlaut-loss variants;
- `CDU/CSU` coalition-shorthand conflation;
- person ↔ party confusion;
- local/English duplicate canonicals;
- 2024 temporal scoping for BSW and EU-group relations;
- recurrent OCR/ASR variants.

## Fix made in this pass

Added a public synthetic regression test for Germany that pins:
- DE profile/language/file mapping;
- Unicode-preserving identity keys;
- German acronym retrieval;
- country isolation for short aliases;
- distinct CDU and CSU canonical identities;
- German/English context rendering without replacing the source-language canonical label.

This is intentionally synthetic and contains no private research data.

## Completion record

- country: Germany (DE)
- agent: ChatGPT
- branch: `german-umlaut-bureaucrat-counts-acronyms`
- public sources: Bundeswahlleiterin (German + English), European Parliament
- private data inspected: **no, materialized LFS content unavailable**
- major problems/risk found: acronym-loss risk from shared builder/merge guards; CDU/CSU conflation risk; Unicode/ASCII variant risk; temporal BSW context requirement
- fixes made: DE regression tests + this audit
- remaining uncertainty: materialized private codebook and legacy-row QA still deserves an independent local/LFS re-check
- recommended re-check: yes, specifically private codebook/legacy data
