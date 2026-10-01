# EP24 country audit: Finland (FI)

Status: **provisional first-agent pass** for issue #72. This is not a completion marker. Finland must be re-checked independently by at least one additional agent before its codebook/context is treated as mature.

Pass 1 also fixed two defects in the shared coverage auditor (`scripts/ep24/codebook_coverage.py`) that made this and every other country's numbers incomparable. See "Tool defects found" below.

## Scope

This public audit records public-source research, measured aggregates and generic normalization / prompt / RAG recommendations only. Private researcher notes, spreadsheets, legacy rows, derived private codebooks and sensitive mappings remain in `TomiToivio/LaclauGPT-Private`.

Aggregates only: no private row values, cells or researcher text appear below.

Country profile:

- ISO / pipeline code: `FI`
- primary campaign language: Finnish (`fi`)
- second national language, materially present in the campaign: Swedish (`sv`)
- retrieval companion language: English (`en`)
- election: 2024 European Parliament election in Finland
- election date: 9 June 2024
- seats: 15 (up from 14; the 2020 Brexit reallocation plus a 2023 review added the 15th)

## Public sources checked in pass 1

Starting points, not an exhaustive bibliography:

1. English Wikipedia, 2024 European Parliament election in Finland:
   https://en.wikipedia.org/wiki/2024_European_Parliament_election_in_Finland
2. Finnish Wikipedia, Suomen europarlamenttivaalit 2024:
   https://fi.wikipedia.org/wiki/Suomen_europarlamenttivaalit_2024
3. Oikeusministeriö (Ministry of Justice) information and result service — the official result source cited by the above
4. European Parliament results portal: https://results.elections.europa.eu/
5. Yle election compass (vaalikone) — public, multilingual, and the campaign's most-used public information intermediary

Later agents should add official candidate lists, party programme documents, 2024 Finnish political journalism and Finnish political-science sources.

## Measured codebook state

From the shared auditor (`scripts/ep24/codebook_coverage.py`) after the resolution fix, `ep24_finland_private.json`:

| Measure | Value |
| --- | --- |
| entries | 325 |
| entries without aliases | 272 (**83.7%**) |
| aliases total / per entry | 74 / 0.23 |
| kind breakdown | `entity` 275, `theme` 41, `topic` 9 |
| entries carrying **zero** observations | 45 (**13.8%**) |
| fragmented entity groups (post-fix) | 36 |

Alias coverage is the headline weakness: **fewer than one alias per six entries**. An entry with no aliases can only be matched on its exact label, so every surface variant the legacy pipeline produced is unmatched — which is precisely the failure mode the legacy data demonstrates below.

## Findings from the private legacy data (aggregates only)

2,498 legacy FI rows, 95 columns. These are quality defects, not load failures.

### F1. Entity cells pack several entities together — 71.6%

Of 2,155 non-empty `entities` cells: **1,544 (71.6%)** contain a comma-separated list, mean **2.13** entities per cell. Any consumer treating the cell as one entity gets a compound string.

### F2. Entity strings are lower-cased in the majority of distinct forms — 68.0%

**620 of 912 (68.0%)** distinct entity parts are all-lowercase. Proper nouns are lost, which breaks exact matching against any correctly-cased codebook label and makes case a *source of mismatch* rather than something normalization can rely on.

### F3. The same party appears under many surface forms

Measured surface forms per party inside the legacy `entities` field:

| Party | Distinct forms | Most frequent |
| --- | --- | --- |
| Perussuomalaiset | **21** | `perussuomalaiset finns party` |
| Vihreät | 13 | `green party` |
| Kokoomus | 11 | `kokoomus` |
| Keskusta | 10 | `keskusta` |
| Vasemmistoliitto | 7 | `vasemmistoliitto` |

The variation is not only spelling: it includes **translation into English**, **suffix `party`**, **acronym only** (`ps`), and **both languages concatenated** (`kokoomus national coalition party`). A codebook whose Perussuomalaiset entry carries one alias cannot resolve 21 observed forms.

### F4. `new_entity` is a *pruning* field, not a correction field — 74.7% removal

This is the most consequential finding, and it contradicts the natural reading of the field names.

Comparing `entities` against `new_entity` on the **498 rows where both are filled** (token-set comparison, not substring):

| Relationship | Rows | Share |
| --- | --- | --- |
| identical | 126 | 25.3% |
| `new_entity` ⊂ `entities` (**pruned**) | **372** | **74.7%** |
| `entities` ⊂ `new_entity` (**added**) | **0** | **0.0%** |

**`new_entity` never adds an entity. It only removes.** What it removes is dominated by institutions and organisations rather than noise:

| Removed tokens | Occurrences | Kind |
| --- | --- | --- |
| `european`, `union` | 179 + 101 | EU institution |
| `party` | 89 | entity kind |
| `parliament` | 67 | institution |
| `perussuomalaiset` | 45 | **political party** |
| `kokoomus` | 20 | **political party** |
| `sdp` | 14 | **political party** |

Totals: **380** institution-token removals, **113** party-name removals, 662 other. The observed shape is always the same: a cell listing one or more institutions or parties alongside one person is reduced to the person alone. Where the legacy cell combined a party and a person, the party is what disappears.

**Why this matters for #72 and for the #33 seeding plan.** Issue #33 specifies seeding the next round from `entities = canonicalized union of new_entity + researcher_new_persons`. That produces a **person-only seed set**, because `new_entity` has already had the organisations and parties stripped out — and party and institution identity is exactly what the codebooks are for. The union is safe as a *person* seed and unsafe as a general entity seed.

Two readings are possible and the choice is the author's: either `new_entity` was intended as a person-only field, or the historical matching removed organisations it should have kept. The measurement above cannot distinguish intent, only effect. It should be decided explicitly rather than inherited by accident.

### F5. `researcher_new_persons` is 85% empty-container noise

| Value | Rows |
| --- | --- |
| empty | 2,457 |
| literal `[]` | **35** |
| genuinely informative | **6** |

A naive `non-empty` fill-rate reports 41 "filled" rows; the honest count of researcher-supplied persons is **6** across 2,498 rows. `researcher_new_themes` shows the same pattern (36 of 41 are `[]`), though its non-empty values are richer multi-theme lists.

**Normalization rule:** treat `[]`, `{}`, `[ ]`, `null`, `None`, `nan`, `-` as *absent*, never as a value. A fill-rate built on `!= ""` overstates researcher supervision by roughly 7× for these two columns. Any prompt or seed built from them must parse the container first.

### F6. Themes are not the fragmented axis — the long tail is short

2,293 `new_theme` cells, 114 distinct parts, mean 2.65 themes per cell, and only **1.8%** of distinct parts appear once. Theme vocabulary is largely closed and reused. Unlike entities, themes do **not** need the same urgency of alias work — the effort belongs on entity identity.

### F7. `spacy_entities` is empty in 100% of rows

0 of 2,498. Any pipeline stage or codebook builder that expects NER output from this column is reading an empty field. It should be treated as unavailable, not as evidence of absence of entities.

## Public-source coverage check

The FI book against the **14 parties contesting the 2024 EP election** (from the public result tables):

- **present: 7 / 14** — and these are exactly the 7 that won seats: National Coalition, Left Alliance, Social Democratic, Centre, Green League, Finns, Swedish People's Party.
- **absent: 7 / 14** — Christian Democrats, Freedom Alliance, Movement Now, Liberal Party – Freedom to Choose, Communist Party of Finland, The Open Party, Truth Party.

This is a defensible prioritization rather than a defect — the seat-winners carry most analytical weight — but it is worth recording that the codebook is **seat-weighted, not contest-weighted**, so a query about e.g. the Christian Democrats or Freedom Alliance will find nothing.

## Normalization recommendations (FI)

1. **Case is not a reliable signal.** 68% of legacy forms are lower-case. Match on a casefolded key and keep the observed surface form for provenance; never require cased agreement.
2. **Resolve `fi`/`sv`/`en` triples on one canonical ID.** Every major FI party has a Finnish name, a Swedish name and an English name, and the legacy data used all three. The book already hints at this (`Perussuomalaiset` → `Sannfinländarna`, `Kansallinen Kokoomus` → `Samlingspartiet`) but coverage is thin.
3. **Treat a trailing `party` as a kind marker, not part of the name.** The same lesson as the auditor fix: `Perussuomalaiset party` and `Perussuomalaiset` are one entity.
4. **Do not let a coalition/list acronym stand for a party** where lists exist. Finland used open-list PR in a single nationwide constituency with no electoral threshold, so the coalition-vs-party problem is milder than in HR/PT — but electoral alliances still appear in some countries' data and the rule is shared.
5. **Institutions are entities too.** Given F4, decide explicitly whether `European Parliament`, `European Union` and `Finnish Government` belong in the entity layer. If they do, they must be re-added deliberately, because the legacy pipeline removed them.

## RAG and prompt notes (FI)

- **Country- and language-scope every retrieval.** Finnish party acronyms are short and collision-prone (`ps`, `sdp`, `kd`, `rkp`, `vas`); resolve within `FI` and within entity kind, never globally.
- **Query expansion in both directions** (`fi` ↔ `en`, plus `sv`) is high-value here because the legacy data itself mixed the three.
- **Do not inject the full codebook.** With 325 entries and 83.7% alias-poor, whole-book injection would mostly add unmatched strings. Retrieve by entity, kind and country.
- **State the evidence boundary in the prompt.** Given F4 and F5, a prompt that says "these are known entities" without distinguishing *observed in this item* from *known in the world* invites the model to assert presence.

## Tool defects found and fixed in this pass

Both are in `scripts/ep24/codebook_coverage.py`, added by #81. Both are covered by new regression tests (`tests/test_ep24_codebook_coverage_resolution.py`), each verified to fail against the unfixed tool.

1. **Codebook resolution assumed the filename equals the ISO2 code.** `--country FI` failed outright with *"no codebook found"*: Finland's book is `ep24_finland_private.json`, Poland's is `ep24_poland_private.json`, and `countries/fi.json` / `countries/pl.json` do not exist. The auditor therefore could not open **2 of the 11 country books**. Resolution now reads the `country_code` / `country` fields recorded inside each file, falling back to filename matching, and the not-found error lists the books that do exist.
2. **`entity_fragmentation` keyed on the last token longer than 3 characters**, which for English-labelled entries is often a *kind* word. Finland's largest reported group was **25 unrelated entries collapsed under the key `party`** (`Brothers of Italy party`, `Centre Party`, `Finnish Social Democratic Party`, …) at `obs=209`. That is a false merge, and it **masked the real signal**: post-fix, Finland's genuine fragmentation surfaces as 36 groups, led by Perussuomalaiset (5 forms) and Kokoomus (4).

Blast radius of defect 2: it affects PT, FI, PL, DE and SE; HR, ES, BG, FR and HU are immune. #81's tests were verified against **Croatia** — one of the immune countries — which is why the verification looked clean.

## Open questions for the next FI reviewer

1. Decide F4 explicitly: is `new_entity` a person-only field, or did historical matching remove organisations that should be kept? This changes the #33 seed construction.
2. Are institutions (`European Parliament`, `European Union`) entities in this project's data model?
3. Should the FI book be extended to the 7 non-seat-winning 2024 parties, or is seat-weighting the intended policy?
4. Why do `Brothers of Italy party` and `Italian Prime Minister Meloni` appear in the FI book? They may be legitimate cross-border mentions in Finnish media, or a scoping leak. Not deleted in this pass — flagged.
5. Confirm the `fi` ↔ `sv` ↔ `en` triple coverage per party against the official party register.

## Status

Finland is **not finished**. One agent has now measured the state and fixed the two tool defects that made measurement unreliable. Per #72, at least one further independent agent must re-check FI — including the materialized private material — and should challenge these findings rather than accept them.

#72 stays open.
