# EP24 country audit: France (FR)

Issue: #91. Agent: Ai (愛), on Tomi's WSL box. Branch: `french-fennec-fixes-false-friends`.
Status: **first-agent QA pass — needs a second independent agent.**

## Scope

Country-scoped codebook, entity normalization, RAG/Memory context, prompt-settings and
language/runtime review for France, following the method and document structure of the
HR / ES / PT / PL / DE / SE / BG audits. Private researcher material is **not**
reproduced here; aggregates only.

Country profile:

- pipeline code `FR`; languages `fr` + `en`; private file `ep24_fr_private.json`
- election: 2024 European Parliament election in France
- election date: 9 June 2024; seats 81; turnout 51.4%
- 38 candidate lists contested (see §Public sources)

## Baseline measurement

Measured with the merged auditor (`scripts/ep24/codebook_coverage.py`, #81) so the
numbers are comparable with the other country passes.

```bash
python3 scripts/ep24/codebook_coverage.py \
    --root <private codebook root> --country FR
```

| metric | FR | published peer mean |
|---|---:|---:|
| entries | 650 | 465 |
| entries **without aliases** | **89.5%** | 87.0% |
| aliases per entry | 0.14 | 0.18 |
| fragmented entity groups | 95 | 60 |
| near-duplicate themes | 8 | 7 |

FR is the second-largest book and sits with PL/HU/SE at the bottom of the alias
distribution. `load_profile` additionally reports **`missing_english_count = 650`** — i.e.
**100%** of FR entries lack an English label, the highest of the ten countries (peer
range 10.5%–20.9%). The loader computes that signal; nothing in the pipeline consumes it,
so the gap is invisible downstream.

## Finding 1 — French acronyms do not exist as retrievable forms

FR party identification runs through short acronyms. Measured against the real FR book,
**none** of `NFP`, `RN`, `FN`, `LR`, `PS`, `LFI`, `EELV`, `MoDem`, `PCF`, `LO`, `NPA`,
`REC` exists as a label **or** as an alias of anything:

```text
NFP    label=[]  alias_of=[]        in_label=['Nouveau Front Populaire (NFP)']
RN     label=[]  alias_of=[]        in_label=['Rassemblement National (RN)', ...]
LR     label=[]  alias_of=[]        in_label=['Les Républicains (LR)', 'LR (Les Républicains)']
LFI    label=[]  alias_of=[]        in_label=['La France Insoumise (LFI)', ...]
EELV   label=[]  alias_of=[]        in_label=[]
```

Two distinct failure modes:

1. **Glued inside the label.** `Rassemblement National (RN)` is one string. No `forms`
   entry equals `RN`, so a post saying `le RN` matches nothing. There are **16** such
   parenthetical labels in the FR book and **129 across the ten books** — this is systemic,
   not French.
2. **Absent entirely.** `EELV`, `LO`, `NPA`, `REC`, `NFP` are not present in any form.

Real corpus cost, measured over the 2441 legacy FR rows (`entities` + `whisper_transcript`):

```text
RN   93 hits in 53 rows    retrieval selects 0 entries
LR   30 hits in 17 rows    retrieval selects 0
PS    8 hits in  4 rows    retrieval selects 0
LFI  28 hits in 19 rows    retrieval selects 7   (only because 'Insoumis' is a separate label)
```

This is the FR instance of the general short-alias/alias-poverty problem already tracked
as **#95** and escalated in the HR audit. It is recorded here with French counts rather
than fixed as an FR codebook tweak, because the fix is the shared builder/merge guard plus
a retriever that can carry country-scoped acronyms.

## Finding 2 — the auditor's `_fold` deletes every Cyrillic letter (systemic)

The PL pass (#88) added `scripts/ep24/fold_fix.py` with 14 regression tests. The FI
follow-up (#96) documented that the auditor **never imported it** and the defect was
therefore still live. This pass **fixed that**: `codebook_coverage._fold` now delegates to
`fold_fixed`.

The measured consequence was worse than the PL pass could see, because PL only had `ł`:

| country | labels | `_fold` → empty string |
|---|---:|---:|
| FI, SE, PL, PT, DE, ES, HU, HR, FR | — | **0** |
| BG | 316 | **52** |

52 of 316 Bulgarian labels (16%) folded to the **empty string**, because the old filter
`[^a-z0-9 ]` deleted every letter NFKD does not decompose — which is all ~330 non-decomposing
Cyrillic letters, i.e. the entire Bulgarian alphabet. Four BG themes were invisible to
duplicate detection for the same reason.

This is why the published cross-country table is **not comparable across countries** and why
its BG fragmentation figure is wrong in a way a single-country pass cannot see:

```text
BG entity fragmentation groups:  published 34
                                 auditor as shipped  32
                                 with fold_fixed     37
```

34 is reproducible by neither. Any country comparison built on this tool was computed with a
fold that silently dropped one country's writing system.

**Fixed in this pass.** `non_decomposing_letters()` also scanned only Latin Extended-A/B
(`U+0100`–`U+024F`), so it could not report Cyrillic at all; it now scans every script and
`non_decomposing_scripts()` breaks the count down per script. 14 new regression tests pin the
wiring, the negative direction (no letter may be silently deleted) and the Cyrillic case.

## Finding 3 — French ligatures: `œ`/`æ` are letters, and the corpus is full of them

`œ` and `æ` are single letters with no NFKD decomposition. Under a "strip non-ASCII" fold
they are deleted, splitting the word. This is not a corner case in French:

```text
corpus occurrences (2441 legacy FR rows):
  cœur 76   cœur/coeur both attested      œuvre 19    manoeuvre/œuvre
  sœur  6   soeur also attested           mœurs  8    œuf 9
  103 of 2441 rows (4.2%) contain œ/Œ; 16 contain æ/Æ (Hjælp, Næss, …)
```

So 4.2% of the French corpus carries a character that the codebook auditor corrupted before
counting. FR is the second country (after PL) hit by this class, and the fix is the same one:
keep the letter, never map `œ → oe` (both spellings are attested in real text, so that is a
reviewable alias decision with provenance, not a fold).

## Finding 4 — French elision splits identity on the apostrophe (FR-specific, new)

French elision is written with **either** the ASCII apostrophe (`l'Union`, what keyboards
and ASR produce) **or** U+2019 (`l’Union`, what word processors, iOS autocorrect and much
French news web output produce). Measured in the real corpus:

```text
ASCII  '  : 540 744 occurrences
U+2019 ’  :   5 826 occurrences
2 400 rows contain both
```

`identity_key` normalises NFC, casefolds and collapses whitespace and **nothing else** —
deliberately, because accents and punctuation are analytically meaningful (HR audit rule 1).
The two apostrophes are canonically unrelated (`Po` vs `Pf`), so **NFC cannot unify them**
and no normalising pass in the pipeline can know which was meant.

The damage is at **identity**, and the measurement matters because the obvious diagnosis is
wrong. `score_entry`'s token fallback splits on non-word characters, so the *scorer* already
bridges the two forms:

```text
score_entry("Besoin d’Europe", entry labelled "Besoin d'Europe") == 1.0   # already works
identity_key("Besoin d'Europe") != identity_key("Besoin d’Europe")        # does not
```

But identity is what everything downstream keys on. Two spellings of one entity become two
`CB-…` entry ids, two `A-…` memory object ids, and **two CANONICAL objects**:

```text
'Besoin d'Europe'  -> A-1b23218f83f451b5
'Besoin d’Europe'  -> A-9eb91724e91f80b9
```

Nothing flags this, because each spelling is individually valid.

**Tool added in this pass:** `scripts/ep24/apostrophe_hygiene.py`, the French generalisation
of the BG pass's `cyrillic_hygiene.py`. It reports, it does not repair — unifying the
apostrophes changes an identity key, and which form is canonical is a researcher decision.
It separates the families carefully:

| family | characters | unify? |
|---|---|---|
| elision | `'` `’` `‘` `ʼ` | **safe candidate** — same linguistic function, different keyboard |
| not apostrophes | `′` (prime), `` ` `` (grave) | **no** — prime means minutes/feet |
| dashes | `-` `‐` `–` `—` `−` | reported only — a hyphen can join a double-barrelled surname |

Current FR state: **8 split groups, all of them in the shared `ep24_common_private.json`
layer, not the FR book** — e.g. `Conseil Constitutionnel` / `Conseil constitutionnel`,
`Lutte Ouvrière` / `Lutte ouvrière`, `Parti Animaliste` / `Parti animaliste`. The FR book
itself writes the ASCII apostrophe consistently (**0 typographic forms**), so this is a
latent risk for incoming FR data rather than damage already present in the FR codebook.
It is live in the corpus, though: 5 826 occurrences.

## Finding 5 — the shared scorer has no stopword filtering, and French articles are labels

`score_entry`'s token fallback uses tokens `len > 2` with **no stopword list**. A French
entity label that begins with an article (`Les Républicains`, `Les Écologistes`,
`La France insoumise`) accumulates the token `les` — so *any* French sentence containing
`les` scores against it. Measured on the real 650-entry FR book:

```text
'les options possibles'  -> 8 selected: ['Les Goguettes','Les Republicains','Les Républicains (LR)','les supporters']
'les gens votent'        -> 8 selected: (same)
'les jeunes'             -> 8 selected: ['Les Jeunes Républicains', ...]
'il y a du monde'        -> 1 selected: ['Le Monde']
total spurious selections across 8 neutral French probes: 27/8 probes all retrieve
```

26 of 650 FR labels begin with a French article/preposition. FR is the worst case in the
set only mildly (FI 2, SE 1, PL 2, PT 0, DE 2, ES 0, HU 2, HR 1, BG 0 begin with a short
lowercase token) — but a stopword token is *inherently* a language-dependent false positive,
so the fix belongs in the shared scorer as per-language stopword lists, not in this audit.
**Recorded, not patched** (see §Escalation).

A related measurement that *narrows* a suspicion: the short-alias word-boundary guard works.
`PS` is matched as a word, so `apside`/`options possibles` do **not** fire the alias. The
false positives above are the token fallback, not the acronym path. That distinction matters
for #95, which is about the acronym path.

## Memory / RAG checks

- Country filtering works: an FR alias does not retrieve DE/ES context (pinned in tests).
- Exact canonical retrieval works; **alias retrieval is the bottleneck** (Findings 1 and 5).
- Local-language + English retrieval is structurally blocked for FR: `missing_english_count
  == 650`, so there is no English form to retrieve on (Finding 1 / baseline).
- Election/time filtering is not implemented at the FR layer; the codebook carries `FR` but
  no `valid_from`/`valid_to` on any entry.
- **Cross-country contamination:** none found in FR. Ambiguous forms = 0, conflicts = 0.
- Evidence firewall restated: codebook/Memory/RAG content is a normalisation aid, never
  evidence that an item contains an actor or theme.

## System prompts and step-specific settings

Checked the FR entry in every stage that is language-aware:

| stage | FR present | note |
|---|---|---|
| `roihu_preprocess.py` (EasyOCR + ASR) | yes | `'fr'` in the EasyOCR list and in the active language list |
| `roihu_frame.py` | yes | |
| `roihu_summary.py` | yes | |
| `roihu_postprocess.py` | yes | |
| `roihu_populism.py` | yes | `"fr": ("FR", "fr")` country map |
| `roihu_rdf.py` | yes | `LANGUAGES` |
| `step_7_…discourse_network_analysis.py` | yes | via `LACLAUGPT_LANGUAGES` default |
| `step_8_…social_network_analysis.py` | yes | via `LACLAUGPT_LANGUAGES` default |
| `roihu_enrich.py` | yes | `"FR": ("ep24_fr.csv", "fr")` |
| `roihu_clean.py` | yes | `KNOWN_COUNTRY_CODES` / `KNOWN_COUNTRY_NAMES` |
| `roihu_codebooks.py` | yes | `COUNTRY_PROFILES["FR"]` |

**No inconsistency found.** FR is complete across the runtime, which matches the BG pass's
conclusion that the late-addition risk is behind the project. The stage-level gap is not FR's
language list — it is that no stage consumes `missing_english_count`, so the bilingual gap in
Findings 1 and "Memory/RAG" cannot surface anywhere.

## Language/runtime configuration

- `country_code`: `FR` (consistent everywhere)
- `languages`: `["fr", "en"]` — correct
- private codebook file: `ep24_fr_private.json` — exists, resolves through `COUNTRY_PROFILES`
- EasyOCR: `'fr'` present; ASR: French via Whisper's `whisper_language` (100% filled)
- `whisper_language` is populated on all 2441 rows; `whisper_translated` on 2249 (92.1%)
- **No country-specific exclusions or special cases** for FR. Unlike BG (Cyrillic) or PL
  (`ł`), FR has no script-level special case — its risk is typographic (apostrophe, ligature).

## Private material inspected

**Yes.** `LaclauGPT-Private` was available in a full checkout with LFS objects materialized.

- `analysis/ep24/codebooks/ep24_fr_private.json` — 650 entries, 75 `public-context`, 575
  `researcher-grounded`; kinds `{entity: 495, theme: 142, topic: 13}`
- `analysis/ep24/codebooks/ep24_common_private.json` — 2694 entries (shared layer)
- `analysis/ep24/codebooks/countries/fr.json` — 2-entry public-context seed
- `analysis/ep24_reprocess/data/by_country/ep24_france_with_researcher_notes.csv` — 2441
  rows × 95 columns
- `analysis/ep24_reprocess/codebook_sources/*.xlsx` — present (entities, persons, themes,
  research_notes)

Aggregate findings from the legacy rows:

```text
entities column filled on 2140/2441 (87.7%)
  distinct cell strings     1455
  cells packing >1 entity   1600 (74.8% of non-empty)
  fully lowercase cells     2034 (95.0%)   <-- proper-name casing destroyed at extraction
new_entity filled on 769 (31.5%); distinct forms 70 vs 970 in `entities`
  -> normalisation collapses by DELETION, same as the HR finding, not by canonicalisation
spacy_entities filled on 0 rows (0.0%)     <-- NER stage contributed nothing
researcher_video_checked 133 (5.4%)        <-- thin human review
researcher_new_persons 2, researcher_new_themes 2
```

The FR rows reproduce the HR/PL structural defects exactly: multi-entity blobs, lowercase
extraction, an empty NER stage, and a normalisation step that removes rather than canonicalises.
`entities` → `new_entity` collapses 970 distinct forms to 70, by deletion.

## Public sources checked

French **and** English, per the issue:

1. French Wikipedia, 2024 European Parliament election in France
   https://fr.wikipedia.org/wiki/Élections_européennes_de_2024_en_France
2. English Wikipedia, 2024 European Parliament election in France
   https://en.wikipedia.org/wiki/2024_European_Parliament_election_in_France
3. European Parliament official results, France
   https://results.elections.europa.eu/en/france/
4. French Ministry of the Interior (election organiser)
   https://www.resultats-elections.interieur.gouv.fr/

Verified against the codebook where FR entries exist:

- the 38-list field and the main list names and heads of list
  (`Besoin d'Europe` / Hayer, `La France revient ! Avec Jordan Bardella…` / Bardella,
  `Réveiller l'Europe` / Glucksmann, `La France insoumise – Union populaire` / Aubry,
  `La droite pour faire entendre la voix de la France en Europe` / Bellamy,
  `Europe Écologie` / Toussaint, `La France fière…` / Maréchal, `Lutte ouvrière…` / Arthaud)
- coalition compositions: `Besoin d'Europe` = Renaissance + MoDem + Horizons + Parti radical
  + UDI + Fédération progressiste; `Réveiller l'Europe` = PS + Place publique;
  `Europe Territoires Écologie` = régionalistes + Volt Europa
- EU groups: RN → ID, LFI → GUE/NGL, LR → PPE, PS/Place publique → S&D, Renaissance → RE,
  Les Écologistes → Verts/ALE

The FR codebook's 75 `public-context` entries are **correct wherever they are populated**
(spot-checked `Besoin d'Europe`, `Réveiller l'Europe`, `Mouvement démocrate`,
`Union des droites pour la République`, `affaire des assistants parlementaires`, the
campaign slogans). The defect is coverage, not accuracy: major 2024 actors are absent
(§Finding 1), not wrong.

**One documented inconsistency:** the codebook contains both `Gouvernement français` and
`Gouvernement de la République française / Matignon`, and both `Ministère de l'Intérieur`
and `Ministère de l'Intérieur et des Outre-mer`, from two different generations of the
public-context layer.

## Fixes made in this pass

Scoped, verified, committed:

1. **`scripts/ep24/codebook_coverage.py`** — `_fold` now delegates to
   `scripts/ep24/fold_fix.fold_fixed`. This is the wiring #96 documented as missing. It stops
   the auditor from deleting every Cyrillic letter and every French ligature before counting.
2. **`scripts/ep24/fold_fix.py`** — `non_decomposing_letters()` now scans every script
   instead of Latin Extended-A/B, so the blast-radius report can name Cyrillic; added
   `non_decomposing_scripts()` for a per-script breakdown.
3. **`scripts/ep24/apostrophe_hygiene.py`** *(new)* — French generalisation of the BG
   pass's `cyrillic_hygiene.py`; reports elision apostrophe variants, distinguishes them from
   primes/grave accents/dashes, and reports identity splits. Report-only.
4. **`tests/test_ep24_french_codebook.py`** *(new, 28 tests)* — FR profile, ligature fold,
   apostrophe identity split, acronym retrieval, acronym word-bounding, country scoping,
   parenthetical-label unreachability, coalition-vs-party, person-vs-party.
5. **`tests/test_ep24_apostrophe_hygiene.py`** *(new, 19 tests)* — detector positive and
   negative cases, the not-an-apostrophe family, `identity_split` noise rejection, CLI.
6. **`tests/test_ep24_codebook_fold_defect.py`** — extended with the auditor-wiring pin, the
   Cyrillic blast-radius case and the FR ligature case.

Deliberately **not** fixed here (escalated instead): the alias/acronym coverage gap
(Finding 1, → #95), the stopword-token false positives (Finding 5), the missing
`english_label` coverage, and the builder/merge min-length guard.

## Verification

```text
ruff check (my files)                        All checks passed
pytest tests/test_ep24_french_codebook.py    28 passed
pytest tests/test_ep24_apostrophe_hygiene.py 19 passed
pytest tests/test_ep24_codebook_fold_defect.py  (extended) passed
full suite                                   431 passed, 4 failed
```

The 4 failures are **pre-existing and unrelated**: `tests/test_roihu_storage.py` 4×,
`AttributeError: 'FakeCollection' object has no attribute 'bulk_write'` at
`roihu_storage.py:284`. Verified by running the full suite on a pristine `origin/main`
worktree: **4 failed, 376 passed**, identical failures. Reported on #91 previously by
another pass; the storage test double is out of date with the real access path.

Note the count moves 376 → 431 because this branch adds 47 tests; the 4 failures are the same
4.

## Open hypotheses for the next reviewer

1. **H1** — the 8 apostrophe split groups in `ep24_common_private.json` are the *shared*
   layer, so every country inherits them. Confirm the same 8 appear for other countries and
   whether the shared layer is the right place to fix them once.
2. **H2** — `missing_english_count == 650` for FR but the loader computes it and nothing
   consumes it. Confirm no stage reads it (grep says none do) and decide where the gate belongs.
3. **H3** — the FR legacy `new_entity` collapse (970 → 70 distinct forms) is by deletion, as
   in HR. Confirm whether any FR legacy row's normalisation is a genuine canonicalisation.
4. **H4** — `spacy_entities` is empty on all 2441 FR rows. Confirm whether NER ever ran for FR
   or whether the column is vestigial for every country.
5. **H5** — the two parallel public-context generations (`Ministère de l'Intérieur` vs
   `…et des Outre-mer`; `Gouvernement français` vs `…/ Matignon`) suggest two builders writing
   the same layer. Confirm and reconcile.
6. **H6** — stopword tokens: is the false-positive rate materially reduced by a per-language
   stopword list, or does the fix need a length/IDF weight? Someone should measure on the
   real books before designing the fix.
7. **H7** — a *French speaker* should check the 75 public-context definitions for register and
   accuracy; this pass verified names, dates and relationships, not prose quality.

## Escalation

No new issue opened. The findings that need an architectural change are already tracked or
belong to existing threads:

- **alias/acronym coverage, parenthetical labels, min-length guard** → already **#95** and the
  HR escalation. FR adds counts (16 parenthetical labels in FR, 129 across ten books).
- **electoral-list/coalition layer** → already **#74** / merged #93. FR adds the instance
  (`Besoin d'Europe` = a list, not a party).
- **missing `english_label` coverage** → new. Not a codebook tweak: it is the bilingual layer
  of the builder. Flagged here; the maintainer should decide whether it is its own issue or
  part of #95.
- **stopword-token false positives in the shared scorer** → new. Cross-country by nature
  (it is a per-language property), so it deserves its own design/test cycle rather than an
  FR-local patch.
- **auditor `_fold`** → **fixed in this pass**, which closes the FI follow-up (#96) that
  flagged it, and corrects the published cross-country table's BG row.

## Completion record

- **country:** FR — France
- **agent:** Ai (愛), on Tomi's WSL box
- **branch:** `french-fennec-fixes-false-friends`
- **files/settings inspected:** `ep24_fr_private.json` (650), `ep24_common_private.json`
  (2694), `countries/fr.json`, `ep24_france_with_researcher_notes.csv` (2441×95),
  `codebook_sources/*.xlsx`, `roihu_codebooks.py`, `roihu_enrich.py`, `roihu_clean.py`,
  `roihu_preprocess.py`, `roihu_frame.py`, `roihu_summary.py`, `roihu_postprocess.py`,
  `roihu_populism.py`, `roihu_rdf.py`, `step_7_*`, `step_8_*`, `scripts/ep24/*`
- **private data inspected:** **yes** — full checkout with LFS materialized; aggregates only
  are reported here
- **public sources checked:** fr.wikipedia, en.wikipedia, results.elections.europa.eu,
  Ministère de l'Intérieur
- **major problems found:** acronyms unreachable as forms (RN 93 corpus hits → 0 retrieval);
  auditor deleted every Cyrillic letter and every French ligature before counting, making the
  cross-country table non-comparable and the BG fragmentation figure irreproducible;
  the apostrophe split creates two CANONICAL memory objects per entity; 100%
  `missing_english_count`; stopword tokens produce 27 spurious selections over 8 neutral probes
- **fixes made:** auditor wired to `fold_fixed`; blast-radius scan widened to all scripts;
  new `apostrophe_hygiene.py`; 47 new/extended tests
- **PR:** see PR **#100**
- **remaining uncertainties:** the 8 shared-layer apostrophe splits and their per-country
  reach; where the bilingual gate belongs; whether NER ran at all for FR; the two
  public-context generations
- **recommended re-check:** yes — a French speaker on the 75 definitions (H7), and an
  independent agent on H1/H6.
