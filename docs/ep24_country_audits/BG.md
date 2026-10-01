# EP24 country codebook and settings audit: Bulgaria (bg)

Tracking: issue #91. Agent Ai (愛), Laskin, 2026-10-02. Branch
`bulgarian-bagpipe-baffles-byzantines`.

Public, non-sensitive summary. No private rows, researcher notes, source lists or
personal data appear here.

Bulgaria is the only Cyrillic-script country in the EP24 set, and that turns out
to matter: it produces failure modes the nine Latin-script countries cannot have.

## Scope and method

| Layer | Path (private repo) | Size |
| --- | --- | --- |
| Private codebook | `analysis/ep24/codebooks/ep24_bg_private.json` | 316 entries |
| Public-context seed | `analysis/ep24/codebooks/countries/bg.json` | 2 entries (LFS pointer in a plain checkout) |
| Legacy analysis output | `analysis/ep24_reprocess/data/by_country/ep24_bulgaria_*` | 1 700 rows (from 1 709 source; 9 researcher deletes) |

Measured with `scripts/ep24/codebook_coverage.py` (added in #81) plus targeted
Unicode analysis.

## Measured defects

### BG1. Cyrillic and Latin forms of the same entity are separate entries

The same person exists as **four** entries across the two scripts:

```
Boiko Borisov          obs=5     (Latin, transliteration A)
Borisov                obs=42    (Latin, bare surname)
Boyko Borisov          obs=33    (Latin, transliteration B)
Бойко Борисов          obs=2     (Cyrillic, with alias Борисов)
```

The same for Peevski — **five** entries:

```
Delian Peevski  obs=7      Delyan Peevski  obs=11     Mr. Peevski  obs=2
Peevski         obs=12     Делян Пеевски   obs=0 (alias Пеевски)
```

`identity_key` is **not** the bug: it folds Cyrillic case correctly
(`Бойко Борисов` == `бойко борисов`, `ГЕРБ` == `герб`). The bug is that **a
Cyrillic form and a Latin transliteration produce different keys** and neither is
recorded as an alias of the other, so they can never match:

```
identity_key("Борисов") = "борисов"
identity_key("Borisov") = "borisov"     -> DIFFER
identity_key("ДПС")     = "дпс"
identity_key("DPS")     = "dps"         -> DIFFER
```

This is the Cyrillic analogue of the alias-poverty finding from #81/#86, and it is
strictly worse: even a perfectly complete *Latin* alias set will not match a
Cyrillic corpus mention, and vice versa.

### BG2. Two transliteration schemes for the same name, with no alias link

`Boiko` and `Boyko` are both present as `researcher-grounded` entries for one
person, with no alias connecting them. Bulgarian transliteration is not
standardised (the official streamlined system differs from common English usage),
so a single person reliably yields several Latin spellings in a multilingual
corpus. Each spelling became its own entity — the fragmentation defect the Croatia
pass named (R3), amplified by transliteration.

### BG3. A Latin/Cyrillic homoglyph is silently corrupting an entity label

```
'European Parliament (SEМ)'
                        ^ this is CYRILLIC CAPITAL LETTER EM (U+041C), not Latin M (U+004D)
```

NFC normalisation does **not** fix it — the two are distinct characters that are
canonically unrelated. The label renders identically to `SEP`, so nothing in
review would catch it by eye, and it can never match a correctly-typed form:

```
identity_key("European Parliament (SEМ)") = "european parliament (seм)"   (Cyrillic м)
identity_key("European Parliament (SEM)") = "european parliament (sem)"   (Latin m)
-> DIFFER
```

For a Cyrillic-script country this is a **systematic** risk class: a reviewer
mixing keyboard layouts, or an OCR/ASR pipeline emitting Cyrillic where Latin was
meant, introduces lookalike characters that defeat exact matching while looking
correct. A homoglyph guard is the fix, and it is cheap.

### BG4. Entity-type coverage: parties absent from the legacy output

Across 1 700 legacy rows, `new_entity` is populated in 98 rows and contains **28
distinct values — all natural persons, zero parties, zero institutions, and zero
Cyrillic forms**:

```
 18 maya manolova     15 radostin vasilev   12 kiril petkov    12 boyko borisov
 12 donald trump       9 jordan bardella     8 ursula von der leyen ...
```

The same shape as Croatia (D5) and now confirmed in a second, unrelated party
system: **the legacy entity layer captured personalities, not organizations.**
Bulgaria's 2024 lists — GERB-SDS, PP-DB, DPS, Vazrazhdane, BSP, ITN and the rest —
are absent from it entirely.

### BG5. Alias coverage and theme duplication

```
entries                 : 316
without aliases         : 271 (85.8%)     <- in line with the 87% cross-country rate
aliases total           : 91 (0.29/entry)
kind breakdown          : {entity: 257, theme: 55, topic: 4}
entity fragmentation    : 32 groups
theme near-duplicates   : 5
```

The five theme pairs are mostly **purely lexical** variants that a normalizer
should collapse, and two are near-identical:

```
0.83  "Bulgaria's entry into the Eurozone"          ~ "Bulgaria's potential entry into the Eurozone"
0.67  "Bulgaria's role in EU"                       ~ "Bulgaria's role in the EU"
0.67  'Bulgaria-Russia relations'                   ~ 'EU-Bulgaria relations'   <- NOT a duplicate
0.67  'Bulgarian identity'                          ~ 'Bulgarian National Identity'
0.60  'Bulgarian politics and governance'           ~ 'Politics and governance in Bulgaria'
```

The first is a **hedge** ("potential") that changes meaning — whether Bulgaria
*would* join or *will* join is analytically different, so that pair should NOT be
auto-merged. The third is a genuine false positive of the Jaccard heuristic.
Recorded rather than merged, per the abstain-don't-merge rule.

## Fixes applied in this pass

Scoped, safe, and tested — no destructive change to researcher-grounded data.

1. **A homoglyph/mixed-script detector** (`scripts/ep24/cyrillic_hygiene.py`) with
   tests. It flags labels that mix Latin and Cyrillic, reports each confusable
   character with its codepoint, and refuses to auto-rewrite: it reports, because
   rewriting a researcher-grounded label is a review decision.
2. **`docs/ep24_country_audits/BG.md`** — this document.

## Recommended follow-ups (not done here, and why)

- **A script-variant alias convention.** Every Cyrillic entity should carry its
  Latin transliteration(s) as aliases and vice versa. That is a data change across
  60 Cyrillic entries and needs a native-speaker review of which transliteration
  scheme the project standardises on — `Boiko`/`Boyko` shows the project currently
  has both. This is the single highest-value BG fix.
- **Transliteration is a shared-pipeline question**, not a BG-only one: if more
  Cyrillic or Greek countries are added, the normalizer needs a documented
  transliteration policy rather than per-country aliases. Worth a linked issue
  before the country set grows.
- **`bg.json` public seed is an LFS pointer** in a normal checkout. The auditor now
  reports this as `UNREADABLE` with the `git lfs pull` hint rather than crashing
  (fixed in #86), so it is a usability note, not a defect.

## Verification and honesty notes

- Every count above was produced by running the tooling in this pass, not from
  memory. The `identity_key` behaviour was verified by evaluating it, not assumed.
- The homoglyph finding was confirmed by **printing the codepoint** (`U+041C`),
  not by eyeballing the label — it renders as a perfectly normal `M`.
- The theme-duplicate list is reported **with** the false positive and the
  meaning-changing pair called out, rather than presented as five clean duplicates.
- `Andreja`/`Andrija`-style and `Boiko`/`Boyko` pairs are recorded as **merge
  candidates**. They are not merged here: confirming a transliteration or a
  misspelling needs a Bulgarian-language reviewer.
- No private data published. No merge, rename or delete applied to the
  researcher-grounded codebook.

## Remaining uncertainty for a re-check

**A Bulgarian-language reviewer should confirm:** (a) which transliteration scheme
the project should standardise on; (b) that `Делян Пеевски` and `Delyan Peevski`
are the same person (they are, but both entries are `researcher-grounded` and this
pass does not overrule that); (c) whether
`Democratic Pole (Демократическия Полс) party` is a correctly spelled party name —
that label carries a suspicious form and is a good candidate for the review queue.
