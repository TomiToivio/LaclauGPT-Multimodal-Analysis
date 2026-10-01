# EP24 country audit: Portugal (PT)

Status: **provisional first-agent pass** for issue #72. This is not a completion marker. Portugal must be re-checked independently by at least one additional agent before its codebook/context is treated as mature.

## Scope

This public audit records public-source research and generic normalization / prompt / RAG recommendations only. Private researcher notes, spreadsheets, legacy rows, derived private codebooks, and sensitive mappings remain in `TomiToivio/LaclauGPT-Private`.

Country profile:
- ISO / pipeline code: `PT`
- primary campaign language: Portuguese (`pt`)
- retrieval companion language: English (`en`)
- election: 2024 European Parliament election in Portugal
- election date: 9 June 2024
- seats: 21

## Public sources checked in pass 1

These are starting points, not an exhaustive bibliography:

1. Comissão Nacional de Eleições (CNE), Eleições para o Parlamento Europeu 2024:
   https://www.cne.pt/content/eleicoes-para-o-parlamento-europeu-2024
2. Portuguese Wikipedia, Eleições europeias de 2024 (Portugal):
   https://pt.wikipedia.org/wiki/Elei%C3%A7%C3%B5es_europeias_de_2024_%28Portugal%29
3. English Wikipedia, 2024 European Parliament election in Portugal:
   https://en.wikipedia.org/wiki/2024_European_Parliament_election_in_Portugal
4. European Parliament election/results and Portugal member material:
   https://results.elections.europa.eu/
   https://www.europarl.europa.eu/meps/en/search/advanced?countryCode=PT

Later agents should add official candidate/list documents, Portuguese party sites, contemporary 2024 Portuguese news, and suitable political-science sources.

## High-value normalization cases for Portugal

### 1. Coalition/list identity is not the same as constituent-party identity

The 2024 Portuguese result tables include electoral coalitions/list identities as well as individual parties. Important examples include:

- `AD - Aliança Democrática`
- `CDU - Coligação Democrática Unitária`

The public codebook/runtime should not silently flatten an observed coalition/list mention into one constituent party. Keep list/coalition identity separate and use sourced relations for membership when the relation layer from #74 is available.

### 2. Acronyms must be country- and kind-scoped

High-value Portuguese political acronyms include:

- `PS`
- `AD`
- `CH`
- `IL`
- `BE`
- `CDU`
- `PAN`
- `ADN`

These are too short and collision-prone for naive fuzzy or substring matching.

Recommended resolution order:
1. exact canonical label;
2. exact alias/acronym inside country `PT` and entity kind;
3. election/list relation context;
4. local-language + English retrieval context;
5. abstain if ambiguity remains.

### 3. Keep Portuguese canonical labels and English labels distinct

Examples:
- `Partido Socialista` ↔ `Socialist Party`
- `Iniciativa Liberal` ↔ `Liberal Initiative`
- `Bloco de Esquerda` ↔ `Left Bloc`
- `Coligação Democrática Unitária` ↔ `Unitary Democratic Coalition`

English translations are useful retrieval/display metadata. They should not overwrite Portuguese source evidence or become competing canonical Portuguese identities.

### 4. Preserve Portuguese orthography

Canonical identity must preserve diacritics and punctuation, including forms such as:

- `João`
- `António`
- `Coligação`
- `Não`
- `Cidadãos`

Do not make accent stripping part of the stable identity key.

ASCII forms such as `Joao` or `Antonio` may be useful as observed aliases when they are actually attested, but they should not silently replace the canonical Portuguese form.

### 5. Party/list/person are separate entity kinds

Portugal is another useful regression case for:
- party vs electoral coalition/list;
- candidate/list leader vs party;
- national party vs European Parliament group.

A person mention can retrieve party/list context, but background membership must not be emitted as directly observed evidence unless the current item actually contains it.

## Proposed PT RAG / Memory context packet

Keep it compact and source-backed:

- country = Portugal
- language = Portuguese + English retrieval companion
- election = EP2024 Portugal
- canonical party/list/person/institution names
- acronym aliases
- Portuguese ↔ English labels
- election-specific coalition/list relations
- temporally valid EU-group relations where needed
- ambiguity notes for short acronyms

Recommended retrieval order:
1. exact PT canonical/alias match;
2. election-scoped relation lookup;
3. bilingual query expansion;
4. semantic retrieval constrained to PT + EP2024;
5. abstention for unresolved ambiguity.

## Prompt evidence firewall

Every step receiving PT background context should preserve this distinction:

> Background codebook / Memory / RAG context can disambiguate evidence that is present in the current post/video. It cannot create evidence that is absent from the current post/video.

Do not inject the full country codebook into every stage.

Suggested stage-specific use:
- preprocessing/entity extraction: aliases, acronyms, Portuguese/English labels;
- summary: only context needed to resolve already-observed actors/terms;
- Laclau analysis: retrieved context for evidenced entities/themes only;
- postprocessing: canonical IDs, aliases, provenance, ambiguity state.

## Private-data audit still required

A local/second agent should inspect the materialized private files for Portugal, especially:

- `analysis/ep24_reprocess/codebook_sources/entities.xlsx`
- `analysis/ep24_reprocess/codebook_sources/persons.xlsx`
- `analysis/ep24_reprocess/codebook_sources/themes.xlsx`
- `analysis/ep24_reprocess/codebook_sources/research_notes.xlsx`
- Portugal rows/files under `analysis/ep24_reprocess/data/by_country/`
- `analysis/ep24/codebooks/ep24_pt_private.json`

Measure, do not guess:
- acronym misses/collisions;
- Portuguese ↔ English duplicate concepts;
- accent-loss variants;
- coalition/list flattening;
- person ↔ party/list confusion;
- malformed parser fragments;
- duplicate/near-duplicate themes;
- recurring researcher corrections that can become validated aliases or rules;
- source-language / provenance gaps.

Do not promote repeated model mistakes into aliases merely because they are frequent.

## Multi-agent review log

### Pass 1
- branch: `portuguese-codfish-rag-orchestra`
- focus: public-source baseline, PT/EN normalization rules, acronym safety, coalition/list identity, RAG/prompt design
- status: provisional
- independent pass 2: required

### Pass 2+
Add independent review findings here. Re-run checks and challenge pass 1 assumptions rather than simply approving them.

## Hypotheses for the next reviewer to falsify

1. Acronym coverage is likely a larger practical PT resolution risk than missing full party names.
2. AD/CDU make Portugal a clean test case for coalition/list identity preservation.
3. Accent-stripped forms should be observed aliases only, never canonical identity normalization.
4. Portuguese ↔ English labels should improve retrieval without creating duplicate entities.
5. Country context should be selectively retrieved per analysis step, not injected wholesale.
