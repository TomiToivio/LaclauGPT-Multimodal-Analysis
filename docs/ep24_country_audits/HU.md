# EP24 country audit: Hungary (HU)

Status: **completed first-agent QA pass for issue #91**.

This public audit records non-sensitive findings only. Operational codebooks, researcher notes, and private corpus rows remain in `TomiToivio/LaclauGPT-Private`.

## Runtime/settings checked

Public runtime profile:

- pipeline country code: `HU`
- country: Hungary
- configured languages: `hu`, `en`
- private country codebook: `ep24_hu_private.json`
- canonical reporting/translation language: English through the shared EP24 pipeline

The public runtime and private builder agree on `HU` / `hu` and the `ep24_hu_private.json` path.

## Private material inspected

Available through the GitHub connector:

- `analysis/ep24/codebooks/ep24_hu_private.json` — Git LFS pointer verified; materialized payload is not exposed through the connector
- `analysis/ep24/codebooks/research_staging/HU.public_context.json` — Git LFS pointer verified
- `analysis/ep24_reprocess/data/by_country/ep24_hungary_cleaning_report.md`
- shared private codebook build/staging rules

The Hungary cleaning report records 1,894 source rows, 1,857 rows retained for reprocessing, 32 recut/split rows, 5 explicit researcher deletes, and no rows retaining non-trivial researcher notes in that derivative report.

Because the large HU codebook/context files are Git LFS objects, this pass does not claim row-by-row inspection of their materialized contents.

## Public-source verification

Checked:

- Nemzeti Választási Iroda (National Election Office), EP 2024 portal: https://www.valasztas.hu/ep2024
- National Election Office English archive/data page: https://www.valasztas.hu/en/korabbi-valasztasok-feldolgozhato-adatai
- Hungarian-language election/result material under the official 2024 portal
- Hungarian Wikipedia 2024 EP election coverage
- English Wikipedia, 2024 European Parliament election in Hungary
- European Parliament material where useful for English naming/context

The official election material confirms that list identities are first-class election objects. The codebook should therefore preserve election-list / coalition identity separately from constituent parties where applicable, using the shared electoral-relation layer rather than flattening a list into one party.

## Hungarian language / normalization findings

### Preserve Hungarian orthography

Canonical Hungarian labels must preserve accented letters including `á é í ó ö ő ú ü ű`. In particular, `ő` and `ű` are distinct letters and must not be destructively ASCII-folded.

The shared `identity_key()` is appropriate for canonical identity because it uses Unicode NFC + casefold + whitespace normalization without stripping accents.

ASCII-loss forms may be retained only as reviewed aliases when they are actually observed in OCR/ASR/social data.

### Hungarian vs English personal-name order

Hungarian personal names conventionally appear family-name first in Hungarian contexts, while English-language sources frequently use given-name-first order.

Example shape for a synthetic actor:

- Hungarian canonical/local form: `Kovács Anna`
- English form/alias: `Anna Kovács`

These must point to one canonical actor when explicitly linked by the codebook. The runtime must **not** globally sort or reverse two-token names, because token reordering would create false merges.

The same principle applies to real candidate/politician entries: use sourced aliases / `english_label`, not an automatic name-order heuristic.

### Party/list acronyms

Hungarian campaign language uses short and compact party/list forms. This is another instance of the shared short-alias architecture problem already tracked in #95. Do not weaken canonical identity or permit unrestricted substring matching to recover them.

### Coalition/list identity is temporal

Election alliances/list combinations are 2024 election facts, not timeless synonyms. Preserve the list object and scope membership relations to the relevant election/time window.

## Memory / RAG / evidence firewall

For HU, retrieval should remain:

1. exact canonical/alias hit inside `HU`;
2. country/kind/election-scoped relation context;
3. Hungarian + English query expansion;
4. semantic retrieval inside the same country/election scope;
5. abstain when ambiguous.

Codebook/RAG context is background knowledge. It may disambiguate an entity already evidenced in the current item but must not create observed evidence.

## Prompt/stage review

No separate Hungary-only pipeline is justified. Targeted country context is useful for:

- Hungarian language identification;
- Hungarian/English person-name aliases;
- party/list aliases and 2024 relations;
- preserving `ő/ű` and other diacritics through OCR/ASR/translation;
- explaining that translated/reordered English names are aliases, not new actors;
- keeping list membership and EU-group context out of the observed-evidence layer.

Do not inject the full country codebook into every prompt.

## Fixes made in this pass

Public regression tests were added for:

1. preserving Hungarian `ő/ű` in canonical identity;
2. representing Hungarian and English personal-name order as forms of one reviewed entry;
3. proving that token reversal is **not** a global normalization rule;
4. keeping HU aliases country-scoped;
5. preserving the shared evidence firewall.

The shared short-acronym issue remains #95 rather than being duplicated.

## Remaining uncertainty / requested re-check

A second agent with a checkout that has Git LFS materialized should inspect the full HU codebook, public-context file, and real Hungary legacy rows. In particular:

- measure aliases and canonicalization for 2024 party/list names and candidates;
- measure Hungarian-order vs English-order person duplicates;
- inspect OCR/ASR variants of `ő/ű`;
- check list/coalition flattening and temporal relations;
- inspect theme duplication and researcher corrections.

That private LFS pass is the main remaining Hungary-specific uncertainty.
