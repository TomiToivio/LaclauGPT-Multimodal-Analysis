# EP24 country audit: Poland (PL)

Issue: #72 (long-lived; **not** closed by this pass).
Agent: Ai (愛) on Laskin. Branch: `polish-phoenix-counts-signifiers`.
Status: **provisional — needs a second independent agent.**

## Scope

Country-scoped codebook, entity normalization, RAG/Memory context and prompt-context
review for Poland, following the method and document structure of the HR, ES and PT
audits. Private Excel/researcher contents are **not** reproduced here; see
[Private-data audit](#private-data-audit-still-required).

## Baseline measurement

Measured with the merged auditor (`scripts/ep24/codebook_coverage.py`, #81) so the
numbers are comparable with the other country passes rather than newly invented.

```bash
python3 scripts/ep24/codebook_coverage.py \
    --root <private codebook root> --country PL
```

| metric | PL | peer mean (9 countries) |
|---|---:|---:|
| entries | 711 | 428 |
| entries **without aliases** | **91.3%** | 85.9% |
| aliases per entry | **0.11** | 0.20 |
| fragmented entity groups | **97** | 58 |
| near-duplicate themes | 17 | 5 |

Full cross-country comparison (same auditor, same run):

```text
ctry   entries  no-alias%  alias/ent  fragGrp  themeDup
PL         711       91.3       0.11       97        17   <-- worst on 3 of 4
HU         500       90.8       0.11       74        18
SE         572       89.9       0.15       63         1
FR         650       89.5       0.14       96         8
BG         316       85.8       0.29       34         5
PT         356       85.1       0.22       58         1
ES         470       84.7       0.22       61         3
FI         325       83.7       0.23       31         2
DE         404       83.4       0.21       46         7
HR         350       80.0       0.31       39         3
```

**PL is the worst country on entries, alias coverage and fragmentation.** It is also
the largest codebook (711 entries), so the problem is not missing content — it is
**unconsolidated** content.

> This is the same shape as the ES finding (`es.json` was structurally thinner than
> its peers). Here the defect is inverted: PL has *more* raw material than anyone and
> *less* structure.

## Finding 1 — the auditor cannot audit PL at all (defect in the merged tool)

`scripts/ep24/codebook_coverage.py` resolves a country as
`ep24_<iso>_private.json` or `countries/<iso>.json`. For Poland **neither exists**:

```text
PL: ep24_pl_private.json=NO   countries/pl.json=NO
HR: ep24_hr_private.json=yes  countries/hr.json=yes
```

The Polish file is named **`ep24_poland_private.json`**, breaking the ISO convention
every other country follows. `ep24_finland_private.json` has the same defect, and
`countries/fi.json` and `countries/pl.json` are the only EP24 countries missing from
a 25-entry directory.

```text
private codebooks, by naming convention:
  OK   ep24_bg_private.json        OK   ep24_hr_private.json
  BAD  ep24_common_private.json    OK   ep24_hu_private.json   (common: intentional)
  OK   ep24_de_private.json        BAD  ep24_poland_private.json
  OK   ep24_es_private.json        OK   ep24_pt_private.json
  BAD  ep24_finland_private.json   OK   ep24_se_private.json
  OK   ep24_fr_private.json
```

**Consequence:** the reusable auditor that HR/ES/PT used **silently excludes PL and
FI**. Anyone re-running the cross-country comparison gets a table that omits two of
ten countries without an error — the worst failure mode for a comparison tool.
Workaround used for this audit (read-only, no private files modified): a staging
directory of convention-correct symlinks.

**Recommendation:** the auditor should either accept a country→filename override, or
fail loudly with the *actual* filenames it found, rather than reporting only the two
names it looked for. Filename normalization in the private repo belongs to its own
change, not this audit.

## Finding 2 — Polish `ł` silently breaks diacritic folding (language-specific)

`_fold` strips accents via NFKD + dropping combining marks, then deletes anything not
`[a-z0-9 ]`. **Polish `ł` has no NFKD decomposition** — unlike `ż`, `ź`, `ó`, `ń`, `ć`,
`ś`, `ą`, `ę`. It therefore survives the accent strip and is then **deleted** by the
punctuation filter, splitting the word:

```text
Arłukowicz  ->  'ar ukowicz'      (one token becomes two)
Łępkowska   ->  'epkowska'        (leading letter deleted)
```

Measured blast radius across the ten EP24 codebooks:

```text
PL   711 entries    50 tokens split by the bug    74 entries affected
HR   350 entries     1 token  split by the bug     1 entry  affected
FI/PT/ES/DE/SE/FR/HU/BG    0
```

This is why PL's fragmentation count is inflated relative to peers: the tool corrupts
PL's tokens before counting them. It also means any retrieval or matching built on
`_fold` would fail to match `Arłukowicz` against itself.

**The fix must be conservative.** Folding `ł -> l` is *not* safe as an identity rule:
Polish `ł` is /w/ (English "w"), while `l` is /l/. `Łępkowska` and `Lepkowska` are
**different names**. So the correct change is to *keep* the letter, not to map it:

| input | broken fold | corrected fold |
|---|---|---|
| `Arłukowicz` | `ar ukowicz` | `arłukowicz` |
| `Łępkowska` | `epkowska` | `łępkowska` |
| `Żukowska` | `zukowska` | `zukowska` (unchanged) |
| `Kraków` | `krakow` | `krakow` (unchanged) |

Implemented in `scripts/ep24/fold_fix.py` with 14 regression tests
(`tests/test_ep24_codebook_fold_defect.py`), mutation-checked in both directions:
reverting to the broken regex fails one test, and over-folding `ł -> l` fails a
different one. The same non-decomposing class affects other Latin scripts
(`đ`, `ı`, `œ`, `ħ`, …), so this is a general defect that PL merely exposed.

## Finding 3 — coalition/list flattening (same systematic gap as HR/PT)

Poland contests EP elections through *komitety wyborcze*, and 2024 coalitions are
entities distinct from their members. The PL codebook fragments this badly:

```text
Koalicja Obywatelska       9 labels   (… + '(Citizens' Coalition)' + 'party' variants)
Konfederacja              14 labels   (+ 'Criticism of Konfederacja')
Lewica                    13 labels   (+ 'KOALICYJNY KOMITET WYBORCZY LEWICA')
Trzecia Droga              3 labels   ('Trzecia Droga PSL', 'Trzecia Droga)')
Zjednoczona Prawica        3 labels
```

The codebook has **no entity kind for coalition or list** — every entry is
`entity`, `theme` or `topic`:

```text
kinds present: {'entity': 576, 'theme': 122, 'topic': 13}
```

This is **not a new PL defect**: the HR pass escalated it as
[issue #74](https://github.com/TomiToivio/LaclauGPT-Multimodal-Analysis/issues/74)
(electoral lists/coalitions as entities in their own right). PL is additional
evidence for #74, and the alias-only approach cannot repair it — no amount of alias
enrichment turns a flattened party into a list.

## Finding 4 — entity-kind vocabulary is too coarse for country work

`entity` currently covers parties, coalitions, persons, institutions, media, places
and abstract relation labels. Observed consequences in PL:

- `Opposition parties in Poland` is stored as `kind: entity` — a category, not an entity.
- Placeholder text from the model is stored as entities:
  `a specific campaign or organization`, `the person featured in the video`,
  `The main subject (man in yellow shirt)` — 10 such entries.
- `PSL` resolves to five labels spanning `PSL party`, `PSL (Polish People's Party)`
  and `PSL (Polskie Stronnictwo Ludowe)`.

A finer kind vocabulary (`party`, `coalition`, `list`, `person`, `institution`,
`media`, `place`) would make both auditing and retrieval safe, but it is a **data-model
change** and therefore belongs in the linked escalation, not in this audit.

## Finding 5 — diacritic variants are real and unevenly handled

Nine label groups collide when accents are stripped, i.e. the same entity exists under
an accented and an unaccented spelling:

```text
['Anna Maria Zukowska', 'Anna Maria Żukowska']
['Dariusz Jonski', 'Dariusz Joński']
['Joanna Kaminska', 'Joanna Kamińska']
['Krakow', 'Kraków']
['Ewa Zajaczkowska', 'Ewa Zajączkowska']
['Katarzyna Pełczynska-Nałęcz', 'Katarzyna Pełczyńska-Nałęcz']
```

These are safely foldable (`ż`, `ń`, `ó`, `ą` all decompose). But note the ones that
are **not** foldable and need explicit aliases instead, because they are misspellings
rather than accent variants:

```text
'Krystof Bosak'           -> 'Krzysztof Bosak'      (missing letter)
'Bartosz Arlukowicz'      -> 'Arłukowicz'           (ł deleted, not folded)
'Alicja Lepkowska-Golas'  -> 'Łępkowska'            (ł deleted)
'Bożena Przyluska'        -> 'Bożena Przyłuska'     (ł deleted)
```

**Rule for PL:** fold only what NFKD actually decomposes; everything else needs a
reviewable alias with provenance. Do not let a fold silently become a merge.

## Proposed PL RAG / Memory context packet

Country-scoped context for prompts and retrieval. **Every entry is background
context, never evidence** — see the evidence firewall below.

- **Election:** 2024 European Parliament election in Poland, 9 June 2024; 53 seats;
  turnout 40.65%.
- **Result order:** Civic Coalition 37.06%, Law and Justice 36.16%, Confederation 12.08%.
- **Party system:** PiS / Law and Justice; KO / Civic Coalition (containing Platforma
  Obywatelska, Nowoczesna, Zieloni, Inicjatywa Polska); Trzecia Droga / Third Way
  (Polska 2050 + PSL); Lewica / The Left (Nowa Lewica + Razem); Konfederacja;
  Bezpartyjni Samorządowcy.
- **Institutions:** PKW (Państwowa Komisja Wyborcza) and KBW (Krajowe Biuro Wyborcze)
  are the official electoral bodies; Sąd Najwyższy rules on election validity.
- **Campaign-period controversies:** rule-of-law dispute (`spór o praworządność`) and
  EU funds conditionality; abortion/reproductive-rights dispute (`spór o aborcję`);
  farmers' protests (`protesty rolników`) over the Green Deal and Ukrainian grain;
  the Poland–Belarus border crisis (`kryzys na granicy polsko-białoruskiej`).
- **Locally specific vocabulary:** `praworządność` (rule of law), `suwerenność`
  (sovereignty), `KPO` (National Recovery Plan), `europosłowie` (MEPs), `eurowybory`.
- **Temporal note:** campaign period 26 April – 8 June 2024; government formed
  13 December 2023 (KO + Trzecia Droga + Lewica, PM Donald Tusk).

Sources for the above are public: Polish and English Wikipedia 2024 EP election
pages, `results.elections.europa.eu`, PKW/KBW official material, and 2024 news
reporting. Provenance is already carried in the private codebook under
`provenance_class: public_context`.

## Prompt evidence firewall

Carried over from the other country passes and **required** for PL:

> Codebook, Memory and RAG content are **contextual normalisation aids**. They are
> never evidence that a given post, video or frame contains that actor, theme, claim
> or grievance. Background context must not be cited as an observation, and an entity
> appearing in the codebook is not evidence that it appears in the current item.

Poland has a specific version of this risk: the codebook contains placeholders such
as `The current Polish government`, which are *descriptions of a referent*, not names.
Injected into a prompt they invite the model to attribute them to any item discussing
any government. Recommend excluding placeholder-shaped entries from retrieval
entirely.

## Private-data audit still required

Not performed in this pass; needs a shell with access to
`LaclauGPT-Private/analysis/ep24_reprocess/`:

- [ ] `codebook_sources/*.xlsx` — sheets, columns, researcher additions, naming
      conventions, recurring categories, aliases, country-specific concepts.
- [ ] `data/by_country/` old PL analysis output — recurring entity mistakes,
      party/person confusion, inflectional and diacritic failures, theme
      over-generation, researcher corrections that imply a reusable rule.
- [ ] Verify the §Finding 5 pairs against researcher corrections before any alias is
      added: which spelling did the researcher treat as canonical?
- [ ] Confirm whether `Krystof Bosak` / `Arlukowicz` / `Lepkowska` are ASR/OCR errors
      or genuine alternative forms in the data.

## Multi-agent review log

### Pass 1 — Ai (愛), Laskin, 2026-10-02

Findings 1–5, baseline measurement, fold defect with regression tests. Measured
numbers and the tool defect are the contributions intended to be reusable.

### Pass 2+

**Not done. PL is provisional.** A second independent agent should falsify:

1. That the auditor filename defect is real for FI as well, and whether any other
   country in the wider 25-entry directory has the same problem.
2. That the `ł` fold defect is the *only* script-specific fold bug; the
   non-decomposing letter class is larger than Polish (`đ`, `ı`, `œ`).
3. Whether the 97 fragmentation groups are genuinely one-entity-many-labels, or
   whether some are legitimately distinct (e.g. `Criticism of PiS` is arguably a
   theme, not a fragment of `PiS`).
4. Whether folding accents at all is safe for Polish person names in the corpus —
   `Zukowska`/`Żukowska` may be two different researchers' people.
5. Local-language accuracy of the proposed context packet (a Polish speaker should
   check the vocabulary and the institution descriptions).

## Open hypotheses (for the next reviewer)

- H1: the 50 split tokens are the largest single cause of PL's inflated
  fragmentation; fixing `_fold` would materially reduce PL's 97.
- H2: `countries/<iso>.json` and `ep24_<iso>_private.json` are produced by different
  generators, and PL/FI fell through a gap in the ISO mapping.
- H3: theme over-generation is systematic; PL's 17 near-duplicates will grow as
  more countries are audited with the same auditor.
- H4: kind vocabulary coarseness will block safe retrieval for every country, not
  just PL.

## Escalation

No new issue opened: the coalition/list data-model gap is **already** tracked as #74,
and the kind-vocabulary finding is part of the same change. This audit adds PL
evidence to #74 rather than fragmenting the discussion. If the maintainer prefers a
separate issue for the entity-kind vocabulary, this section is the draft.

**This audit does not close #72.** Country remains provisional pending independent
review.
