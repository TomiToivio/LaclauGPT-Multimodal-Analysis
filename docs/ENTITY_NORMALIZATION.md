# Researcher codebook sources as an entity-registry layer

`roihu_codebook_sources.py` reads the researcher-maintained replacement workbooks
(`entities.xlsx`, `persons.xlsx`, `themes.xlsx`) under
`LaclauGPT-Private/analysis/ep24_reprocess/codebook_sources/` and turns them into
`CodebookEntry` objects, so they can seed the entity resolution layer in
`ep24_entities.py` alongside the JSON country codebooks.

The workbooks are `old` → `new` replacement tables, not JSON registries:

```text
file            sheet    shape         meaning
--------------  -------  ------------  --------------------------------------
entities.xlsx   Sheet3   old, new      surface mention -> canonical entity
persons.xlsx    Sheet3   old, new      surface mention -> canonical person
themes.xlsx     Taul1    old, new      surface mention -> canonical theme
```

Each `new` value is a canonical entity; every `old` sharing it is an alias of that
entity. Themes collapse thousands of surface forms into a small canonical set
(one theme carries ~160 alias forms), so this is real alias data, not fixtures.

## Why this layer is needed

The JSON country codebooks carry **no person entries at all** — measured across
FI/PL/PT/DE/FR/ES they contain only `entity` and `topic` kinds, zero
`person`/`actor`. Persons are reachable only through `persons.xlsx`. Registering
the workbooks is therefore what makes entity resolution work for people, which is
the case the issue leads with (`Pääministeri Orpo` / `Orpo` / `Petteri Orpo` /
`Orpon` → one canonical actor).

## Two measured properties that shape the loader

**1. `new` embeds the disambiguation in parentheses.** Rows read
`Canonical Name (Party; Country)`. Taking `new` verbatim as the canonical label
means a plain mention of that person does not match their own canonical form, so
the parenthetical is parsed into structured party and country fields. Unrecognised
trailing parts are preserved in `extra` — researcher annotation is never silently
dropped.

**2. `persons.xlsx` is a strict subset of `entities.xlsx`** — all 418 person rows
also appear in the entity rows. A naive per-file load produces two entries per
person, one typed `ORGANIZATION` by the filename rule. That splits the aliases
across two ids and makes people unreachable under a person-scoped lookup. Groups
are therefore merged by canonical name + country, with type precedence
`PERSON` > `ORGANIZATION` > `THEME`, and every contributing workbook is recorded in
`metadata["source_workbook"]` as a list so provenance is not lost.

The merge key includes the **country**, so the same name in two countries stays two
entities. Measured against the real files: 700 canonical entities, 5,512 alias
forms, 418 PERSON / 160 ORGANIZATION / 122 THEME, all ten EP24 countries.

## API

```python
from roihu_codebook_sources import load_registry_from_dir
from roihu_codebooks import load_profile
import ep24_entities

workbook_entries, report = load_registry_from_dir(codebook_sources_dir)
profile_entries, _ = load_profile(private_root, "FI", strict_english=False)

registry = ep24_entities.EntityRegistry.from_codebooks([*profile_entries, *workbook_entries])
result = registry.resolve("Pääministeri Orpo", country="FI", language="fi")
```

Entries are ordinary `CodebookEntry` objects with `review_state="CANONICAL"`,
`origin="researcher_private"` and `locked=True`: a researcher-maintained canonical
mapping is not auto-editable. Nothing in this module resolves identity — it only
supplies the registry's canonical names, aliases, countries and type, and leaves
resolution policy to `ep24_entities.py` / `roihu_identity.py`.

A workbook that fails to parse is reported in the load report and skipped, so one
broken file cannot suppress the rest of the registry.

## Scope note

This module supplies the **input** to entity resolution. It deliberately does not
re-implement resolution, normalization or matching: the merged
`ep24_entities.py` already owns that policy, and duplicating it would create two
competing answers to the same question.

## Tests

`tests/test_ep24_codebook_sources.py` — 19 tests, synthetic/public fixtures only.
They reproduce the *structure* of the private resources (parenthetical
disambiguation, alias groups, a person duplicated across two workbooks,
cross-country homonyms) without private data, and they exercise the integration
through the merged resolver, including that persons are unreachable without this
layer.
