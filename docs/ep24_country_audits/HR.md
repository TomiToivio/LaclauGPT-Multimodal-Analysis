# EP24 country audit: Croatia (HR)

Status: **provisional first-agent pass** for issue #72. This document is deliberately not a completion marker. Croatia must be re-checked by at least one additional independent agent before its codebook/context can be treated as mature.

## Scope

This public audit records only public-source research and generic normalization/prompt recommendations. Private researcher notes, spreadsheets, legacy rows, derived private codebooks, and sensitive mappings remain in `TomiToivio/LaclauGPT-Private`.

Country profile:
- ISO / pipeline code: `HR`
- primary campaign language: Croatian (`hr`)
- retrieval companion language: English (`en`)
- election: 2024 European Parliament election in Croatia
- election date: 9 June 2024
- seats: 12
- electoral system facts useful for context: single nationwide constituency, preferential voting, 5% threshold

## Public sources checked in pass 1

Start/re-check these rather than treating this list as exhaustive:

1. Croatian State Electoral Commission (DIP), official 2024 European Parliament election pages:
   https://www.izbori.hr/site/izbori-referendumi/izbori-clanova-u-europski-parlament-iz-republike-hrvatske/68
2. European Parliament Croatia 2024 country sheet:
   https://www.europarl.europa.eu/news/en/press-room/20240527BKG21656/european-elections-2024-country-sheets/3/croatia
3. Croatian-language Wikipedia election page (re-check availability/current title during later reviews).
4. English Wikipedia, 2024 European Parliament election in Croatia:
   https://en.wikipedia.org/wiki/2024_European_Parliament_election_in_Croatia
5. European Parliament member search scoped to Croatia:
   https://www.europarl.europa.eu/meps/en/search/advanced?countryCode=HR

Later agents should add Croatian-language party sites, official candidate/list material, contemporary 2024 Croatian news, and suitable political-science sources.

## Entity model: do not flatten lists into parties

Croatia is a useful regression case because an electoral list can contain several parties while also functioning as a campaign-level entity.

Represent separately when supported by the source:

- `party`: constituent political party
- `electoral_list`: submitted list / coalition/list identity
- `person`: candidate or political actor
- `eu_group`: European Parliament political group
- `institution`: election authority, parliament, government, etc.

Examples from the European Parliament country sheet include:

- Hrvatska demokratska zajednica / HDZ
- Domovinski pokret / DP
- Možemo!
- Socijaldemokratska partija Hrvatske / SDP
- Most
- Hrvatski suverenisti
- HSP
- a multi-party SDP-led list
- an IDS/NPS/SDSS/Socijaldemokrati/HSLS/PGS/Reformisti/... list

Do not silently replace a coalition/list mention with one constituent party. Keep a list-level canonical object and explicit membership relations where the private codebook schema permits it.

## Croatian/English normalization rules to test

### 1. Preserve diacritics in canonical identity

Canonical labels should preserve Croatian orthography: `č ć đ š ž`.

Do not make accent/diacritic stripping part of the canonical identity key. A diacritic-stripped form may be stored as a low-confidence observed alias only when it is actually attested in data or public sources.

### 2. Abbreviations are aliases, not sufficient identity by themselves

Examples such as `HDZ`, `SDP`, `DP`, `HSP`, `HSU`, `RF`, and `IDS` should resolve only inside country/kind context.

Short abbreviations must abstain on collision rather than cross-country matching.

### 3. Keep local and English labels distinct but linked

Store:
- canonical Croatian label,
- English label/translation where an established English form exists,
- observed aliases,
- source language,
- provenance.

Do not replace Croatian surface evidence with an English translation. Translation is context, not source-text rewriting.

### 4. Person aliases must not imply party identity

A leader/candidate name may retrieve relevant party/list context, but a person mention is not itself evidence that the party/list is mentioned.

This is especially important for prominent list leaders and candidates such as Andrej Plenković, Biljana Borzan, Ivan Penava, Gordan Bosanac, Božo Petrov, and others.

### 5. Model 2024 list membership temporally

Electoral-list membership and alliances are election-specific facts. Where possible, scope relations to EP2024 rather than making them timeless identity facts.

## Prompt/RAG context packet proposed for HR

A compact HR context packet should contain only enough background to disambiguate current evidence:

- country = Croatia / Hrvatska
- languages = hr + en
- election = EP2024 Croatia
- electoral list aliases and constituent-party relations
- canonical party/person/institution aliases
- election-specific alliance/list relations
- EU-group mappings only when sourced and temporally appropriate

Retrieval order:
1. exact canonical/alias hit within HR;
2. exact person/list/party relation within EP2024;
3. local-language + English query expansion;
4. semantic retrieval inside HR/EP2024;
5. abstain if ambiguous.

Every prompt receiving background context should repeat the evidence firewall: **background knowledge can disambiguate evidence present in the current item; it cannot create evidence that is absent from the item.**

## Legacy-data checks still required in LaclauGPT-Private

A later/local agent with access to materialized LFS objects should inspect:

- `analysis/ep24_reprocess/codebook_sources/entities.xlsx`
- `analysis/ep24_reprocess/codebook_sources/persons.xlsx`
- `analysis/ep24_reprocess/codebook_sources/themes.xlsx`
- `analysis/ep24_reprocess/codebook_sources/research_notes.xlsx`
- `analysis/ep24_reprocess/data/by_country/ep24_croatia_with_researcher_notes.csv`
- cleaned/decision/provenance outputs for Croatia

Specifically count/review:
- HDZ/SDP/DP/HSP/IDS/etc. abbreviation collisions;
- diacritic-loss variants;
- person vs party/list confusion;
- coalition/list flattening;
- local/English duplicate entities;
- same person represented with shortened/full names;
- themes that split only because of translation wording;
- researcher corrections that repeat often enough to become reusable aliases or prompt rules.

Do not promote repeated model errors into aliases without human/source validation.

## Multi-agent review log

### Pass 1
- agent branch: `croatian-diacritics-goblin-council`
- focus: public source baseline, entity ontology, Croatian/English normalization and prompt/RAG design
- result: provisional
- next independent reviewer: required

### Pass 2+
Add independent review entries here. Re-check sources and assumptions rather than merely approving pass 1.

## Open hypotheses for the next reviewer

1. Coalition/electoral-list identity deserves an explicit relation layer instead of being serialized only as aliases.
2. Country-scoped abbreviation resolution should be evaluated before any fuzzy matching.
3. Croatian diacritic-loss should be handled as observed alias evidence, not destructive normalization.
4. Country context should be stage-specific: entity extraction needs aliases/list relations; Laclau analysis needs only retrieved context relevant to entities/themes already evidenced in the item.
5. If legacy HR rows show systematic coalition flattening across stages, open a separate pipeline issue for list/coalition relation preservation.
