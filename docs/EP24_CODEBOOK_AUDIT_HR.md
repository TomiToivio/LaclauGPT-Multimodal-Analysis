# EP24 country codebook audit: Croatia (hr)

Tracking: issue #72. Working pass by agent Ai (愛) on Laskin, 2026-10-02.
Review branch: `grumpy-croatian-hedgehog-sings-eurovision`.

This document records **what was measured** against the live Croatia codebook and
legacy Croatia analysis output, and the reusable rules derived from it. It is a
public, non-sensitive summary: no private rows, researcher notes, source lists or
personal data appear here.

## Scope and method

Two layers were inspected for Croatia (HR):

| Layer | Path (private repo) | Size |
| --- | --- | --- |
| Researcher-grounded codebook | `analysis/ep24/codebooks/ep24_hr_private.json` | 350 entries |
| Public-context country seed | `analysis/ep24/codebooks/countries/hr.json` | 2 entries |
| Legacy analysis output | `analysis/ep24_reprocess/data/by_country/ep24_croatia_*` | 1 267 rows |

The public seed is explicitly labelled `public-context-seed-needs-researcher-reconciliation`
with empty `party_registry`, `account_registry` and `candidate_signifiers`. That
label is accurate: it is a container, not a country codebook.

Legacy rows analysed: **1 267** usable rows (from 1 376 source rows; 97 recut, 11
researcher deletes, 1 dubious exclusion). 677 Instagram / 590 TikTok, all
`synthetic` accounts, `cleaning_status=KEEP` throughout.

## Measured defects

### D1. Entity fragmentation — one person, many entries

The private codebook holds **six separate entries for one politician**, all
`researcher-grounded`, with no aliases linking them:

```
label                              observations
Andrej Plenković                   64
Plenković                          26
Andrija Plenković                   3     <- misspelling as its own entity
Andreja Plenković                   2     <- misspelling as its own entity
Croatian Prime Minister Andrej P.   2     <- role+name as its own entity
```

Fragmenting one actor across six identities splits its observation mass and makes
any per-actor aggregate wrong. The same shape repeats for other actors
(`Kolakušić` / `Mislav Kolakušić`; `Milanović` / `Zoran Milanović`; `Penava` /
`Ivan Penava`).

### D2. Party fragmentation — one party, ten surface forms

`Croatian Democratic Union` (HDZ) appears under **ten** distinct labels, several
of them malformed (unbalanced parenthesis, trailing word):

```
Croatian Democratic Union (HDZ)
Croatian Democratic Union (HDZ) party
HDZ (Croatian Democratic Union)
HDZ (Hrvatska Demokratska Zajednica          <- unbalanced parenthesis
HDZ (Hrvatska Demokratska Zajednica)
HDZ party
Hrvatska demokratska zajednica
Hrvatska demokratska zajednica (HDZ)
Hrvatska demokratska zajednica (HDZ) party
Criticism of HDZ                             <- a theme mis-typed as an entity
```

(The last is a theme recorded in the entity layer — a separate type-confusion defect.)

The same happens to Homeland Movement / Domovinski pokret (`Domovinski pokret`,
`Domovinski pokret (DP)`, `DP (Domovinski Pokret)`, `Domestic Movement`,
`Homeland Movement`, `Domovinski Pokret (Home Movement)`,
`Domovinski pokret movement` …).

### D3. Alias coverage is the binding constraint

```
entries with NO aliases : 280 / 350   (80%)
total aliases           : 108
```

Alias work is the highest-leverage change in this codebook: four out of five
entries cannot be matched on any form other than their exact label, which is
precisely what produces D1 and D2.

### D4. Missing 2024 electoral actors

Checked against the official 2024 EP election lists for Croatia. Of the 20
party/coalition actors on those lists, **7 are absent from the entity layer
entirely**:

```
HSP        Croatian Party of Rights          (Most-led list partner)
PGS        Alliance of Primorje-Gorski Kotar (Fair Play List 9)
NPS        Independent Platform of the North (Fair Play List 9)
GLAS       Civic Liberal Alliance            (Rivers of Justice)
DO i SIP   Dalija Orešković and People with a First and Last Name (Rivers of Justice)
PiP        Law and Justice                   (0 seats in 2024 — NOT a seat-winner)
Fokus      Fokus                             (did not contest the 2024 EP election)
```

> **Correction note (2024 seats).** This table originally annotated `PiP` as
> "2 seats, Non-Inscrits" and `Fokus` as "minor list". Both were wrong, and both
> were corrected in the Croatia pass-3 re-check against the official DIP results:
> **`PiP` won 0 seats in 2024** (22,214 votes, 2.99%, 8th) — the 2 Non-Inscrit
> seats were won in **2019** by the *Independent list of Mislav Kolakušić*, a
> different vehicle, and the "2 seats" figure was the 2019 result carried into a
> 2024 table. **`Fokus` does not appear in the DIP results at all**; the party did
> not contest this election. The structural finding below stands, but the
> "sharpest case" argument built on `PiP` being a seat-winner does not: the 2024
> seat-winners were HDZ (6), the SDP-led list (4), DP (1) and Možemo! (1).

The original argument for `PiP` as the sharpest case was that a **seat-winning**
party had no codebook entry. That was wrong: `PiP` is not a seat-winner. The
coverage gap is real — a party polling 2.99% and contesting nationally should be
representable — but it is a *contender* gap, not a representation gap, and the
case should be made on that basis.

Coverage is also uneven by layer: some present parties exist only as a
`public-context` entry (e.g. `Centar`, sourced to the party's Wikipedia page)
while others are `researcher-grounded`. The two layers are not yet reconciled.

> **Correction note.** An earlier revision of this document stated "13 of 20
> missing". That count was wrong — it came from a keyword match that both
> false-positived and false-negatived. The list above was re-derived
> by enumerating the entity labels and reading each candidate. The lesson is part
> of the finding: **substring matching over entity labels produces confident wrong
> coverage numbers.** Coverage diffs must match on exact labels/aliases, and a
> human must read the survivors.
>
> **Correction note (pass 3).** The two examples originally given for that
> false-positive were `Možemo` matched `Mozemohr` and `SDSS` matched
> `Republika Srpska`. Only the first is real. The second does **not** reproduce
> under any matcher in the tree — raw substring (either direction), the
> auditor's fold (either direction), word-boundary regex, token overlap, or
> initials all return false:
>
> ```text
> score_entry("SDSS", <entry Republika Srpska>)          = 0.0
> boundary_matches("SDSS", "Republika Srpska")           = False
> "sdss" in "republika srpska"                           = False
> fold_fixed("SDSS") in fold_fixed("Republika Srpska")   = False
> ```
>
> What does reproduce is a weaker case: the **full name** `Samostalna
> demokratska srpska stranka` token-overlaps the entry `Republika Srpska` at
> `score_entry = 0.5`, via the shared tokens `srpska`/`demokratska`. The acronym
> was quoted where the full name was the actual trigger. The lesson is unchanged,
> but the right examples are `Most` ⊂ `Mostar` and `HDZ` ⊂ `HDZx`, which
> reproduce cleanly. `Možemo`/`Mozemohr` is also real but belongs to the
> **auditor's fold path** (which strips diacritics), not the runtime matcher,
> which preserves them.

### D5. Legacy output contains no party entities at all

Across the 1 267 legacy rows, `new_entity` is populated in 171 rows (13%) and
contains **41 distinct values — all natural persons, zero parties or
organisations**. Party/coalition mentions in that corpus went unrecorded, or were
folded into person entities.

### D6. Theme duplication and near-duplicate labels

```
legacy theme cells containing a repeated theme : 67 / 1267
  worst repeat                                : "climate, sustainability and environmentalism" x27
distinct theme tokens in legacy output         : 559 (101 after splitting cells)
```

Near-duplicate codebook themes that would fragment the same concept:

```
Croatian interests          ~ Croatian National Interests     (Jaccard 0.67)
Croatian identity           ~ Croatian National Identity      (Jaccard 0.67)
Politics in Croatia         ~ Politics and governance in Croatia (0.60)
```

## Derived rules (reusable beyond Croatia)

**R1 — Normalize before aggregating, never after.** A legacy entity field mixing
`Plenković`, `Andrej Plenković` and `Croatian Prime Minister Andrej Plenković` is
three identities to any downstream counter. Normalization must run *before*
counting, or every per-actor number is wrong.

**R2 — A bare surname and a full name are a merge CANDIDATE, not a merge.** In a
corpus containing several politicians who share a surname — and Croatia's 2024
list does — an unqualified surname must stay explicitly ambiguous until the
evidence resolves it. The memory model already has the right shape for this
(`AMBIGUOUS` decisions); the codebook should carry the alias link and let memory
adjudicate, rather than the codebook pre-deciding.

**R3 — Misspellings are aliases, not entities.** `Andrija`/`Andreja Plenković`
should be alias forms on one canonical entry. Creating an entry per variant *is*
the fragmentation defect.

**R4 — Role-prefixed forms are alias forms.** `Croatian Prime Minister Andrej
Plenković` is a surface form of the person, not a distinct actor.

**R5 — Duplicate themes within one cell are a generation bug, not a data choice.**
67 cells repeat a theme; the split-and-dedupe must happen at write time.

**R6 — Deduplicate themes on a normalized key, not on the raw string.**
`Croatian identity` and `Croatian National Identity` are the same retrieval
concept; they must collapse to one canonical theme with the other recorded as an
alias, or retrieval silently fragments.

**R7 — Entity type must be explicit, and "no parties found" is a signal.**
D5 was invisible in the aggregate output; it only appeared on a type breakdown.
Coverage reporting should always print the entity-type distribution.

**R8 — Check the country's actual electoral actor set against the codebook.** The
2024 EP official lists are the ground truth for *which actors exist*; a coverage
diff against that list is a cheap, high-value audit (it found the 13 missing
parties in seconds).

## Reproducible review sample

The measurements above are reproducible from the paths in "Scope and method".
A follow-up implementation slice should add a synthetic fixture under
`tests/` encoding R2/R3/R4 (surname ambiguity, misspelling-as-alias,
role-prefix-as-alias) so the rules are enforced rather than documented.

## Status and open questions for a second reviewer

- D1–D6 are **measured**, not inferred; the counts come from the files named above.
- Whether `Andreja`/`Andrija Plenković` are true misspellings of Andrej Plenković
  or a genuinely different person is **not** asserted here. The observation
  counts and the shared surname make a merge *candidate*; a Croatian-language
  reviewer should confirm before any merge is applied. This is exactly the R2 case.
- No merge, rename or delete was applied to the private codebook in this pass.
  This pass is an audit plus reusable rules; changes to the researcher-grounded
  layer need the reconciliation the seed's own status line asks for.
- Per the parent issue's closure rule, issue #72 is left open.

## Provenance

- Official 2024 results and party lists:
  https://results.elections.europa.eu/en/croatia/
- Electoral system (12 seats, single nationwide constituency, semi-open list,
  D'Hondt, 5% threshold): the English Wikipedia page for the 2024 EP election in
  Croatia, corroborated against the official results portal above.
- Wikipedia is used here as a **discovery/alias** source only, per the codebook
  protocol; contested identities are flagged for review rather than asserted.
