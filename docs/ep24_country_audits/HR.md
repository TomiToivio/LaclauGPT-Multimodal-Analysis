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

### Pass 2 (independent re-check)
- agent branch: `croatian-hedgehog-eats-leftovers-too`
- focus: the legacy-data checks pass 1 explicitly deferred (this pass had the private material)
- result: pass 1 hypotheses 1 and 5 **confirmed with counts**; a pipeline-level defect found and escalated
- note: pass 1 is not merely approved here — its two testable hypotheses were tested against the real rows

Pass 1 said, correctly, that "a later/local agent with access to materialized LFS objects should inspect" the HR rows. This pass did that. Aggregates only; row-level private content stays in `TomiToivio/LaclauGPT-Private`.

## Pass 2 measured results (legacy HR data)

Legacy HR dataset: 1,376 rows, 95 columns. Researcher review is thin: `researcher_video_checked` is filled on 113 rows (8%); `researcher_dubious_video` 112, `researcher_delete_video` 112, `researcher_new_persons` 107, `researcher_note` 68. So no more than ~8% of HR rows carry a human correction — everything else rests on the unreviewed pipeline output.

**Hypothesis 1 — CONFIRMED.** Coalition/electoral-list flattening is real, and it shows up first as *surface-form fragmentation*:
- 1,193 non-empty `entities` cells contain **849 distinct** entity strings, and **852 cells (71% of non-empty) pack more than one entity into a single comma-joined string** (for example `european parliament, hdz, milanović, plenković`).
- All 1,193 non-empty entity cells are **lower-cased**, so proper-name casing is destroyed at extraction time.
- The same party appears under many variants: **HDZ 9** distinct surface forms (including both the bare acronym and the full Croatian *and* English names, sometimes glued together), **Domovinski pokret 8**, **SDP 4**, **Možemo 4**.

**Hypothesis 5 — CONFIRMED and escalated.** The flattening is not only present, it is **systematic across stages**, and it is worse than "flattening":
- `spacy_entities` is **completely empty** in this dataset (0 of 1,376 rows); the NER stage contributed nothing.
- `new_entity` (the normalization/correction output) is filled on only **186 of 1,376 rows**, and on **150** of those it *removes* entities rather than canonicalizing them — e.g. a three-entity cell becomes a single person. That is why **zero** HDZ variants appear in `new_entity`: the party was not normalized, it was dropped.
- **1190 of 1,376 rows keep the raw fragmented blob as their final value**, including 108 rows that still carry a multi-party blob. So the fragment the researcher sees in the analysis input is the one the extractor wrote, not a canonical entity.
- Net effect: the distinct-surface count collapses from **849** in `entities` to **56** in `new_entity`, but by deletion, not by canonicalization.

**Hypotheses 2, 3, 4 — not yet measurable from data alone.** They remain design rules; pass 1's reasoning is sound and this pass has nothing to refute.

## This is not a Croatia-specific defect (systematic, all 10 countries)

The same build defects reproduce across every EP24 country book, which is why this is escalated as a pipeline problem rather than fixed as a Croatia codebook tweak:

| metric | all 10 country books |
|---|---|
| country entries | 4,654 |
| entries with **empty** `aliases` | 4,060 (**87%**) |
| acronyms dropped by the min-length guard | **938** |
| mechanical-parse fragments (`Podemos)`, `PSOE)`, …) | 16 |
| labels beginning lowercase | 232 |

Per country, empty-alias share ranges 80% (HR) to 91% (PL, HU); dropped short forms range 76 (ES) to 134 (SE). The guard's own log for HR records `Most (label shorter than 5)` and `NATO (label shorter than 5)`, and the HR drop-list contains `HDZ`, `SDP`, `DP`, `HSP`, `IDS`, `SDSS`, `MOST`/`Most`, `ECR`, `EPP`, `EU`, `S&D`, `NATO`, `USA` and more. Every one of the main Croatian parties is in the *dropped* set, and **none of `hdz`, `sdp`, `dp`, `most`, `hsp`, `hsls`, `ids` resolves to any exact HR label or alias** (only `Možemo` does). HDZ's 10 competing codebook labels are therefore all unreachable by the acronym a post or a researcher would actually type.

## Escalation

Hypothesis 5 asked for a separate linked issue if the flattening was systematic. It is, and so is its cause: the **canonicalization/alias layer is missing from the builder**, end to end. That is a shared-pipeline defect affecting all countries and all stages, not an HR codebook fix, and it is tracked separately (see PR discussion for the issue link) rather than folded into #72.

The HR-specific, in-scope follow-ups remain as pass-1 designed them: acronym recovery as country/kind-scoped exact-match aliases, `english_label` instead of English canonical labels, and explicit `source_languages` for the public-context entries.

## Open hypotheses for the next reviewer

1. Coalition/electoral-list identity deserves an explicit relation layer instead of being serialized only as aliases. → **Pass 2: surface fragmentation confirmed (see above); the relation layer is still unbuilt.**
2. Country-scoped abbreviation resolution should be evaluated before any fuzzy matching. → **Pass 2: the acronyms are being dropped, not resolved; scoped exact-match is still the right fix.**
3. Croatian diacritic-loss should be handled as observed alias evidence, not destructive normalization. → **Pass 2: not measurable from data alone; design rule stands.**
4. Country context should be stage-specific: entity extraction needs aliases/list relations; Laclau analysis needs only retrieved context relevant to entities/themes already evidenced in the item. → **Pass 2: supported — `spacy_entities` is empty and `new_entity` deletes rather than canonicalizes, so there is currently no stage that produces canonical entities at all.**
5. If legacy HR rows show systematic coalition flattening across stages, open a separate pipeline issue for list/coalition relation preservation. → **Pass 2: CONFIRMED, escalated separately.**
6. **New for the next reviewer:** verify whether the 938 dropped acronyms across all 10 books have a second cause besides the min-length guard (e.g. a per-book owner map that gives up on a colliding form). If so, both causes need fixing before acronyms can be recovered.
