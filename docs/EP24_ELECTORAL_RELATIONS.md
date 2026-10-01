# EP24 electoral-list and coalition relations

Issue #74 adds a source-backed relation layer for election structures that must not be flattened into parties.

## Data model

The SQLite memory keeps electoral lists, parties, people, and EU groups as normal memory objects. Their relationships are stored separately in `relations`:

- `electoral_list --has_member--> party`
- `party --member_of_list--> electoral_list`
- `person --candidate_on--> electoral_list`
- `party/person --eu_group--> eu_group`

Each relation can carry country, election, validity dates, source type/reference/language, publication date, and an evidence locator. The same table is exported as `memory_relations.csv`, so the Roihu CSV/Pandas workflow remains usable without a service database.

## Evidence firewall

A relation is **background context, never source-item evidence**.

When an observed mention resolves to an electoral-list object, `resolve_with_relations()` returns that list as the observed object and attaches constituent parties/candidates under `related_context`. Every related row is marked:

- `evidence_role = background_relation_not_observed_evidence`
- `observed_in_current_item = false`

Downstream prompts must use relations only to disambiguate already-observed evidence. A constituent party, person, or group must not be emitted as directly observed unless the current source item itself mentions/resolves it.

## Temporal and election scope

`related_context()` can filter by country, election identifier, and date. This prevents a 2024 coalition from becoming a timeless party identity.

## Backward compatibility

The change is additive:

- existing object, alias, affiliation, provenance and proposal tables are preserved;
- schema migration creates the usual pre-migration SQLite backup;
- legacy CSV columns are untouched;
- relations export to their own CSV;
- synthetic public tests contain no private EP24 rows or researcher notes.
