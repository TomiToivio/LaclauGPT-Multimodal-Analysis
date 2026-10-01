# EP24 Memory and bilingual codebooks on CSC Roihu

This is the public runtime contract for issue #11. Operational codebooks, researcher
notes, settings, memory databases and real EP24 data stay in `LaclauGPT-Private`
or private CSC storage.

## Distributed memory and codebook persistence

For the active EP2024 reprocess, MongoDB + Redis supersede the earlier SQLite-first runtime assumption:

- MongoDB is the durable shared store for accepted/proposed memory, codebooks, researcher notes, RAG records/embeddings, identity mappings and provenance.
- Redis is the transient coordination/cache/messaging layer.
- SQLite may still be used for frozen local snapshots, review/export compatibility, tests or job-local checkpoints, but it is not the canonical distributed backend.
- Pandas CSV remains the mandatory readable interchange and must retain all legacy fields.
- Each enrichment/analysis step must add a human-readable Markdown summary in addition to structured output.
- Country collections use `laclaugpt_ep2024_reprocess_<country_name>_<collection_name>`.
- Live credentials come from `LaclauGPT-Private/analysis/ep24_reprocess/.env`; public code refers only to environment-variable names.
- Videos remain in Allas and are fetched on demand by URL/object identifier after `allas_conf` setup.
- PostgreSQL is not used.

## Upstream mapping

Port reviewed against `TomiToivio/LaclauGPT-Data-Analysis` commit
`bf80f435c0390a1d96b3cec21bda09b6c83e7ca0`.

| Upstream behavior | EP24/Roihu adaptation |
| --- | --- |
| SQLite memory with stable IDs and review states | `roihu_memory.py`; adds country/language/disambiguation scope, audit log, redirects, proposals and ID crosswalks |
| accepted-only automatic normalization | retained; `roihu_enrich.py` resolves only `CANONICAL` objects |
| provisional model discoveries | retained as proposals; never auto-promoted |
| label-normalized IDs | compatibility IDs retained in `id_crosswalk`; new EP24 IDs preserve Unicode, country, language and disambiguation |
| generic codebook loader | `roihu_codebooks.py` adapts the existing EP24 private JSON shapes |
| last-write-wins merge | replaced by lock-aware layering; locked human entries cannot be silently overwritten |
| lexical context retrieval | retained and made bilingual; exact acronym matches use token boundaries |
| codebook context is not evidence | retained explicitly in every context block and provenance record |

The historical `legacy` branch is not modified.

## Actual EP24 country/language inventory

The private EP24 builders identify 10 legacy country datasets. Country identity is
taken from this explicit dataset manifest, never inferred from language alone.

| Country | Dataset language | English analysis/output view | Private file |
| --- | --- | --- | --- |
| Finland | Finnish; Swedish may occur; English may occur | yes | `ep24_finland_private.json` |
| Sweden | Swedish; English may occur | yes | `ep24_se_private.json` |
| Poland | Polish; English may occur | yes | `ep24_poland_private.json` |
| Portugal | Portuguese; English may occur | yes | `ep24_pt_private.json` |
| Germany | German; English may occur | yes | `ep24_de_private.json` |
| Spain | Spanish; English may occur | yes | `ep24_es_private.json` |
| Hungary | Hungarian; English may occur | yes | `ep24_hu_private.json` |
| Croatia | Croatian; English may occur | yes | `ep24_hr_private.json` |
| France | French; English may occur | yes | `ep24_fr_private.json` |
| Bulgaria | Bulgarian; English may occur | yes | `ep24_bg_private.json` |

The private repository already contains these generated country codebooks plus
`ep24_common_private.json`, the source protocol, builders, researcher-grounded
FI/PL material and public-context research staging. This public repository does
not duplicate those files.

Run `python roihu_codebooks.py --manifest` for the machine-readable manifest.
Each loaded profile also reports ambiguous surface forms, source-language coverage,
source-backed entry counts, and missing English labels. Missing English coverage is
a recorded gap, never a reason to silently substitute another country's profile.

## Runtime

The canonical compatibility sequence remains exactly:

`preprocess -> frame -> summary -> postprocess -> populism`

When `LACLAUGPT_ENRICHMENT_ENABLED=1`, the runner inserts `enrich` immediately before
`populism` without changing the canonical default stage list. With enrichment
disabled, no enrichment stage is inserted and historical stage ordering is unchanged.

Enable it only when the private codebooks and accepted memory snapshot are ready:

```bash
export LACLAUGPT_MULTIMODAL_PRIVATE_ROOT=/scratch/project_x/LaclauGPT-Private/analysis/ep24
export LACLAUGPT_ENRICHMENT_ENABLED=1
export LACLAUGPT_MEMORY_DB=/scratch/project_x/run-accepted-memory.sqlite3
python roihu_enrich.py
```

The stage appends only:

- `ep24_codebook_fingerprint`
- `ep24_codebook_context_json`
- `ep24_memory_entity_ids`
- `ep24_memory_topic_ids`
- `ep24_memory_unresolved_json`

Legacy values such as `entities`, `topics`, `summary_analysis` and all earlier
columns remain present and are not normalized in place.

## Prepare accepted memory

Use a controlled single writer outside parallel inference:

```bash
python roihu_enrich.py --private-root /private/ep24 \
  --memory-db /private/ep24/database/accepted-memory.sqlite3 \
  --prepare-memory
```

Only reviewed/locked codebook entries are seeded as accepted objects. Aliases may
remain ambiguous. Ambiguity is an abstention, not a silent merge.

Then freeze a transactionally consistent batch snapshot with SQLite's backup API:

```bash
python roihu_memory.py \
  --db /private/ep24/database/accepted-memory.sqlite3 \
  snapshot /job-local/ep24-memory.sqlite3
```

Point `LACLAUGPT_MEMORY_DB` at that frozen job-local snapshot. Do not run a
multi-node workload with concurrent writers against one SQLite database on shared
storage. SQLite WAL is not the cross-host coordination mechanism.

## Slurm shard discipline

1. Build/review accepted memory with one writer.
2. Create a consistent snapshot.
3. Give each Slurm job its own local snapshot/checkpoint path.
4. Record unresolved/model-discovered candidates separately per job.
5. Merge proposals after the batch with a deterministic single-writer review step.
6. Promote valuable outputs atomically back to private project storage.

`roihu_memory.py export-csv` creates inspectable CSV exports of objects, aliases,
redirects, ID crosswalks, source provenance, affiliations, proposals and audit history.

Two public Slurm templates implement the single-writer workflow:

- `scripts/roihu/memory_prepare.sbatch` seeds reviewed private entries, creates a transactionally consistent job-local snapshot, checksums it and atomically promotes the snapshot to private project storage.
- `scripts/roihu/memory_merge.sbatch` snapshots the canonical DB before merging proposal shards, merges shards deterministically and exports an inspectable CSV/provenance bundle.

The memory schema is versioned with `PRAGMA user_version`. A pre-migration SQLite
backup is created before an existing older schema is upgraded.

## Identity policy

The upstream AI26 ID helper deliberately strips accents and terminal parenthesized
qualifiers. That is useful for some generic normalization, but unsafe as the only
identity boundary for EP24. The Roihu adaptation therefore keeps original
Unicode/diacritics/script, country and language, entity type and disambiguation,
original and English labels, plus a compatibility crosswalk to older upstream IDs.

Two people, parties or concepts with the same surface label are not merged merely
because their text matches.

### The legacy crosswalk can be ambiguous, and says so

The compatibility crosswalk is keyed by `upstream_stable_id()`, which is derived
from the accent- and qualifier-stripping helper. That hash is **deliberately
unchanged** — existing upstream ids are already in use and moving them would
silently re-point historical references. But it means one upstream id can
legitimately be claimed by more than one object:

```text
upstream_stable_id("actor", "Puolue (A)")  ==  A-e569d2c4ea4a
upstream_stable_id("actor", "Puolue (B)")  ==  A-e569d2c4ea4a
upstream_stable_id("actor", "Grüne")       ==  A-1cef4e6f3e2e
upstream_stable_id("actor", "Grune")       ==  A-1cef4e6f3e2e
```

Previously a lookup for such an id returned both objects with no indication that
the id was ambiguous — i.e. it resolved by insertion order. That is the silent
merge the identity policy above forbids.

The behaviour now:

- the legacy hash is unchanged, so no existing id moves;
- every claimant is retained (the second is not dropped, and the two are **not**
  merged into one object);
- the collision is recorded in `crosswalk_collisions` and the audit log;
- `resolve_upstream_id()` returns **every** claimant plus `ambiguous: true`, so a
  caller must decide, rather than receiving a winner chosen by write order;
- `crosswalk_collisions()` lists them for operator review.

A genuinely unique upstream id is unaffected: it resolves to exactly one object
and is not flagged.

## Layering and authority

Runtime precedence is common -> EU -> country -> language -> researcher overlay.
The current private files mainly expose common/country/researcher layers. A locked
human entry wins over an unlocked addition. Conflicting locked entries are emitted
as review conflicts instead of using last-write-wins behavior.

Background codebook matches are context only. They do not establish a speaker's
beliefs, a post's discourse category, or agreement with a mentioned actor. When
enrichment is enabled, the legacy `roihu_populism.py` Laclau stage receives the
selected bilingual context as an appended, explicitly non-evidentiary block and
stores the exact selection/fingerprint in additive columns. Existing cached
historical results are retained and labeled as not having received that context.

## Current validation boundary

Public synthetic tests cover Unicode IDs, cross-country homonyms, accepted-only
resolution, ambiguous aliases, locked human entries, rejected/provisional states,
redirects and ID crosswalks, bilingual retrieval, migration backups, SQLite
snapshotting, deterministic proposal-shard merging, disabled-mode no-op behavior,
and additive CSV enrichment.

This implementation consumes the already-populated private EP24 codebooks. It does
not claim that every private entry has independently completed the issue's full
local-language + English 2024 source-audit requirement. That remains a private
research QA task and should be reported from the private source ledgers, not
papered over in public code.
