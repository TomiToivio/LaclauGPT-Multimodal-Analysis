# EP24 country audit: Spain (ES)

Status: **provisional first-agent pass** for issue #72. This document is deliberately not a completion marker. Spain must be re-checked by at least one additional independent agent before its codebook/context can be treated as mature.

## Scope

This public audit records only public-source research and generic normalization/prompt recommendations. Private researcher notes, spreadsheets, legacy rows, derived private codebooks, and sensitive mappings remain in `TomiToivio/LaclauGPT-Private`.

Country profile:
- ISO / pipeline code: `ES`
- campaign language: Spanish/Castilian (`es`)
- other languages present in the study: Catalan (`ca`), Basque (`eu`), Galician (`gl`)
- retrieval companion language: English (`en`)
- election: 2024 European Parliament election in Spain
- election date: 9 June 2024
- seats: 61
- electoral system facts useful for context: single nationwide constituency, closed lists, D'Hondt, no national threshold (3% was abolished in 2019 → now effectively the 5-seat minimum per constituency does the filtering; for EP it is one national constituency)

## Public sources checked in pass 1

Start/re-check these rather than treating this list as exhaustive:

1. Spanish-language Wikipedia, 2024 European Parliament election in Spain (re-check exact title on later passes):
   https://es.wikipedia.org/wiki/Elecciones_al_Parlamento_Europeo_de_2024_en_Espa%C3%B1a
2. English Wikipedia, 2024 European Parliament election in Spain:
   https://en.wikipedia.org/wiki/2024_European_Parliament_election_in_Spain
3. Official results (Ministerio del Interior / European Parliament results portal):
   https://results.elections.europa.eu/en/spain/
4. European Parliament Spain country sheet and MEP search:
   https://www.europarl.europa.eu/meps/en/search/advanced?countryCode=ES

Later agents should add Spanish party sites, official candidate/list material, Catalan/Basque/Galician-language list material, contemporary 2024 Spanish news, and suitable political-science sources.

## What pass 1 found in the current private ES material (aggregate only)

The runtime ES profile is `ep24_es_private.json` (`ep24-private-codebook-v4`): 470 country entries merged over the shared `ep24_common_private.json` (2,694 entries) for a total of 3,164 loaded entries under the current loader contract. The loader loads successfully and reports no alias collisions for ES — so the current ES book is *loadable*, and the defects below are quality defects, not load failures.

Measured defects (counts are aggregates; row-level private content stays in the private repository):

1. **Alias coverage is near-zero.** 398 of 470 ES entries (85%) have an empty `aliases` list. Because the entries are labelled with long parenthesised legacy strings, a reader who writes the bare surface form does not hit the entry.
2. **The five largest Spanish parties have no canonical label and no alias.** `Vox`, `PSOE`, `PP`, `Sumar` and `Podemos` match *zero* exact labels or aliases. They exist only as competing parenthesised variants — e.g. 11 competing PSOE labels, 6 for PP, 4 for Vox, 3 each for Sumar and Podemos. This is a canonicalization failure, not a missing-data problem: the information is present but split across variants.
3. **A minimum-length guard silently dropped the acronyms.** The build records `Vox (label shorter than 5)` and `ABC (label shorter than 5)` as skipped, and Spain's dropped-surface list contains `PP`, `PSOE`, `Vox`, `VOX`, `VOX)`, `Vox)`, `VÓX`, `Psoe`, `PNV`, `ERC`, `ETA`, `OTAN`, `NATO`, `EU`, `ECR`, `5G`, `CGPJ`, `RTVE`, `TV3` and more. Dropping is the correct *runtime* instinct (naive substring matching on `PP` is dangerous) but the acronym must survive as an explicit alias on a canonical party entry, with country/kind-scoped exact matching — not be discarded.
4. **Mechanical-parse fragments became entries.** Examples include `Podemos)`, `PSOE)`, `Sumar)`, `Felipe González)`, `Le Monde)` and the theme `6 mil)`. Five of the six are the same five parties again, confirming the split-parsing root cause.
5. **Language mislabelling on the public-context layer.** The 38 merged public-context entries include Catalan, Basque and Galician regional entities (Junts, ERC, EAJ-PNV, EH Bildu, BNG, Compromís, Coalició Compromís) and English-language EU-group entries, but they carry no `source_languages`; the book-level `language: "es"` then assigns every entry `es`. The country shell `countries/es.json` correctly declares `["es","ca","eu","gl"]`, so the two ES layers disagree about the country's own languages.
6. **`missing_english_count: 470`.** No ES entry has an English label, so the bilingual retrieval the issue asks for is not yet exercised for ES.
7. **`sourced_entry_count` equals total entry count and `source_languages: []`.** The manifest reports full sourcing while recording no source languages, so the two fields cannot currently be used as a QA signal.

## Entity model: do not flatten lists into parties, and do not flatten coalitions

Spain is a two-level regression case, and the 2024 election produced both kinds of confusion at once:

- A **party** that is also its own electoral list (PSOE, PP, Vox) — one identity, many surface forms (acronym, expanded Spanish name, English translation, social-media spelling).
- A **coalition running as a single list** whose member parties also exist as entities (`Sumar` in 2024; `Ahora Repúblicas` = ERC + EH Bildu + BNG + others; `Coalición por una Europa Solidaria` (CEUS) = EAJ-PNV + CCa + others; `Junts i Lliures per Europa` (Junts+)). A list-level mention must not be silently replaced by one member party, and a member-party mention must not be expanded into the whole list.

Represent separately when supported by the source:

- `party`: constituent political party
- `electoral_list`: submitted list / coalition / list identity
- `person`: candidate or political actor
- `eu_group`: European Parliament political group
- `institution`: election authority, parliament, government, etc.

Do not silently replace a coalition/list mention with one constituent party. Keep a list-level canonical object and explicit membership relations where the private codebook schema permits it.

## Nationalization rules to test

### 1. Preserve diacritics in canonical identity

Canonical labels should preserve Spanish orthography (`á é í ó ú ü ñ`, and `·` in Catalan like `col·legi`). Do not make accent-stripping part of the canonical identity key. A diacritic-stripped form may be stored as a low-confidence observed alias only when it is actually attested. Legacy ES rows show both diacritic-dropped and diacritic-bearing spellings of the same person, so this is live evidence, not theory.

### 2. Abbreviations are aliases, not identity by themselves

`PSOE`, `PP`, `Vox`, `IU`, `ERC`, `PNV`, `EAJ-PNV`, `EH Bildu`, `BNG`, `CCa`, `CEUS`, `JxCAT` should resolve only inside country/kind context. Short abbreviations must abstain on collision rather than cross-country matching. `PP` in particular is a two-letter form that naive substring matching will hit inside unrelated words.

### 3. Keep local and English labels distinct but linked

Store the canonical Spanish label, the established English label where one exists (note that the legacy private material used *English* party names such as "United Left", "More Madrid", "Canarian Coalition", "Together and Free for Europe", "The Party is Over" — these are translation artefacts and must be aliases linked to the Spanish canonical label, never the canonical label itself), observed aliases, source language and provenance.

The issue's own test: `Sumar` is not "Unite"; `Se Acabó La Fiesta` is not "The Party is Over"; `Ahora Repúblicas` is not "Republics Now". Legacy rows show these English renderings being used as if they were the entity's name.

### 4. Person aliases must not imply party identity

A leader/candidate name may retrieve relevant party/list context, but a person mention is not itself evidence that the party is mentioned. Legacy ES rows show person-and-party blobs (`estrella galán, psoe`; `jorge buxadé, the european union, the vox party`) being split by the normalizer into a person-only value, which means the party signal was dropped rather than kept as a separate entity with its own identity.

### 5. Model campaign-specific relations temporally

List membership and alliances are election-specific facts. Scope membership relations to EP2024 rather than making them timeless identity facts (the CNMV's 2024 lists will not survive to 2029 unchanged).

### 6. Separate candidate from list

Spain's closed national list means "candidate" and "list" are distinct entity types that legacy outputs conflate (legacy rows contain `politician candidate` as a single entity string).

## Prompt/RAG context packet proposed for ES

A compact ES context packet should contain only enough background to disambiguate current evidence:

- country = Spain / España
- languages = es (+ ca, eu, gl as they occur) + en
- election = EP2024 Spain
- canonical party entries with acronym aliases (nationwide parties)
- regional list entities (Junts+, Ahora Repúblicas, CEUS, Compromís) with their member parties
- canonical person/institution aliases
- election-specific alliance/list relations
- EU-group mappings only when sourced and temporally appropriate

Retrieval order:
1. exact canonical/alias hit within ES (acronyms and diacritic-exact forms);
2. exact person/list/party relation within EP2024;
3. local-language + English query expansion;
4. semantic retrieval inside ES/EP2024;
5. abstain if ambiguous.

Every prompt receiving background context should repeat the evidence firewall: **background knowledge can disambiguate evidence present in the current item; it cannot create evidence that is absent from the item.**

## Legacy-data checks still required in LaclauGPT-Private

A later/local agent with access to materialized LFS objects should inspect:

- `analysis/ep24_reprocess/codebook_sources/entities.xlsx`
- `analysis/ep24_reprocess/codebook_sources/persons.xlsx`
- `analysis/ep24_reprocess/codebook_sources/themes.xlsx`
- `analysis/ep24_reprocess/codebook_sources/research_notes.xlsx`
- `analysis/ep24_reprocess/data/by_country/ep24_spain_with_researcher_notes.csv`
- cleaned/decision/provenance outputs for Spain

Specifically count/review:
- acronym collisions (`PP`, `PSOE` shared with other languages);
- diacritic-loss variants;
- person vs party/list confusion;
- coalition/list flattening in `Sumar` / `Ahora Repúblicas` / `CEUS`;
- English-translation party names mistaken for canonical labels;
- same person represented with shortened/full names;
- themes that split only because of translation wording;
- researcher corrections that repeat often enough to become reusable aliases or prompt rules.

Do not promote repeated model errors into aliases without human/source validation.

## Multi-agent review log

### Pass 1
- agent branch: `spicy-spanish-squid-solves-signifiers`
- focus: public source baseline, entity ontology, Spanish/regional-language normalization and prompt/RAG design
- result: provisional
- next independent reviewer: required

### Pass 2+
Add independent review entries here. Re-check sources and assumptions rather than merely approving pass 1.

## Open hypotheses for the next reviewer

1. Party-acronym recovery deserves an explicit, country/kind-scoped exact-match alias layer, separate from the min-length-substring guard that currently drops them.
2. Canonicalization should collapse the *observed* legacy variants into one canonical party entry while retaining each variant as an attested alias with provenance — the fix belongs in the builder, not in ad-hoc runtime aliases.
3. Regional-language entities need explicit `source_languages`, and the book-level default language must not silently label them as Castilian.
4. English-translation party names should be modelled as `english_label`/alias, never as the canonical label.
5. If legacy ES rows show systematic coalition flattening across stages, open a separate pipeline issue for list/coalition relation preservation.
6. The `sourced_entry_count` / `source_languages` manifest fields should not be trusted as QA signals until a real language-provenance pass has run.
