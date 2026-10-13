# EP24 Step 7: Leifeld Discourse Network Analysis (DNA)

Step 7 is the bridge between the Laclau/Palonen discourse analysis (Step 6) and the
social network layer (Step 8). Its canonical implementation is
`step_7_roihu_discourse_network_analysis.py`, with the rDNA compatibility layer in
`ep24_dna.py`.

## Methodology

The coding follows Philip Leifeld's Discourse Network Analysis:

> Leifeld, Philip (2017). *Discourse Network Analysis: Policy Debates as Dynamic
> Networks*. In: Jennifer N. Victor, Mark N. Lubell and Alexander H. Montgomery
> (eds.), *The Oxford Handbook of Political Networks*, Chapter 25. Oxford University
> Press. Preprint: <https://eprints.gla.ac.uk/121525/>

The **unit of analysis is the statement**: one actor making one claim about one
concept at one time, with an explicit positive/support or negative/oppose qualifier
where the source supports one. The four classic core variables are:

1. **actor** — person or organization making the statement;
2. **concept** — claim, belief, policy position, justification, narrative, frame;
3. **agreement / qualifier** — positive/support vs negative/oppose;
4. **time / date**.

### Evidence discipline

The stage codes only actor-concept claims supported by the **current** record's
evidence. It does not infer positions from topic co-occurrence, and it does not
confuse actor extraction with concept extraction.

Memory (`ep24_memory.py`), RAG (`ep24_rag.py`), codebooks and prior records are
normalization/disambiguation context only. They are labelled
`normalization_context_not_source_evidence` / `prior_analysis_context_not_source_evidence`
in the prompt, and are **never** evidence that the current actor made the current
claim. A remembered actor cannot enter a statement; only the current source can.

### Agreement is not sentiment polarity

`agreement` is a DNA qualifier, not sentiment. Two actors referring to the same
concept with opposite positions must remain distinguishable. The model's raw stance
wording is preserved in `stance`; only explicit markers (`support`/`oppose`, and the
documented synonym sets) map onto the binary qualifier.

An ambiguous statement is **never** forced into positive/negative. It keeps
`agreement = null` and the label `uncertain`, stays fully available in
`dna_statements_json`, and is **excluded from the strict rDNA event list** — because
rDNA's qualifier is binary. This is the documented behaviour on `dna_network`:

> If the upstream DNA schema requires strict binary qualifier values for export,
> ambiguous cases should be omitted from the strict DNA event export ... while
> remaining available in the richer LaclauGPT JSON.

### Temporal data

Leifeld's method is explicitly longitudinal. Every statement carries the best
available timestamp (`date_time`, ISO-8601 UTC). Unparseable values are left empty
rather than invented, and the stage never reduces the discourse to a timeless graph.

## Output contract

Step 7 appends these columns (declared in `ep24_stage_contract.py`):

| column | meaning |
| --- | --- |
| `dna_analysis_markdown` | human-readable DNA interpretation |
| `dna_statements_json` | canonical rich statement records (all fields, uncertainty, provenance) |
| `dna_prompt_version` | coding prompt version |
| `dna_model_metadata_json` | provider/model/num_ctx/num_predict/temperature |
| `dna_generated_at`, `dna_runtime_seconds` | run provenance |
| `dna_context_sha256`, `dna_context_truncated` | deterministic context identity |
| `dna_statement_count`, `dna_binary_count`, `dna_uncertain_count` | statement accounting |
| `dna_actor_unresolved_count`, `dna_concept_novel_count` | normalization accounting |
| `dna_codebook_fingerprint` | codebook version used |
| `dna_memory_context_json`, `dna_rag_context_json` | exactly what context was supplied |
| `dna_raw_response` | raw model output (retained on parse failure) |
| `dna_status`, `dna_error` | per-row status |
| `dna_persistence_status` | MongoDB outcome |

Every incoming column is preserved unchanged (verified by
`assert_source_metadata_preserved` through `write_cumulative_csv`), and the stage
never truncates the canonical CSV with a head-only frame.

### Statement record

Each entry in `dna_statements_json` carries:

`statement_id`, `source_record_id`, `document_id`, `network_layer`,
`organization`, `actor_name_raw`, `actor_id`, `actor_type`, `concept`,
`concept_label_raw`, `concept_id`, `concept_provenance`, `proposition`, `stance`,
`agreement`, `date_time`, `evidence_quote`, `evidence_fields`, `confidence`,
`uncertainty_notes`, `counter_evidence`, `evidence_role`, `provenance`,
`extraction_prompt_version`, `context_sha256`, `source_country`.

- `statement_id` is a deterministic SHA-256 prefix over
  (source record, actor, concept, agreement, timestamp, proposition), so it is
  stable across runs and reorderings — which is what makes duplicate policies
  meaningful.
- `actor_id` comes from the shared entity resolver (`ep24_entities`) or the codebook
  memory lookup. **The LLM never mints a canonical ID.**
- `concept_id` is a codebook entry id when a match exists, otherwise a deterministic
  `concept:<sha256>` id with `concept_provenance = "novel"`. Novel concepts are kept
  with provenance for review rather than being forced into a codebook category.

## Entity and concept normalization

- **Actors** use the existing entity-resolution layer, so `Orpo`,
  `Pääministeri Orpo` and `Petteri Orpo` resolve to the same canonical actor when
  evidence supports it. Aliases come from `ep24_entity_resolution_json` (stable IDs)
  and `entity_normalization_json` (canonical labels).
- **Concepts** use `theme_normalization_json` / codebook memory for a canonical
  label, falling back to the observed surface form. Every concept keeps its raw
  label, canonical label, id, language context and provenance.

## rDNA compatibility layer (`ep24_dna.py`)

### Interchange route (documented decision)

`sample.dna` is a **password-protected SQLite database**, and section 9D of the
issue requires preferring the official API/interchange path over reverse-engineering
it. Section 9 also notes that rDNA's `networkType = "eventlist"` *assembles* an event
list from statements already in DNA's database — it is an output mode, not an import
format.

The selected route is therefore:

1. **A canonical rich statement JSON** — `dna_statements_json` (above).
2. **A DNA-compatible event-list CSV** — one row per binary statement, with the
   rDNA statement type `"DNA Statement"` and the `organization` / `concept` /
   `agreement` variable names, plus document ids, timestamps and provenance.
3. **A generated rDNA import script** — `dna_import_<country>.R`, which loads the
   event list and calls the **official batch API**
   (`dna_addDocuments` + `dna_addStatement`) via `dna_init()` /
   `dna_openDatabase()`. The pipeline itself needs no R installation; the script is
   a generated artefact the operator runs.
4. **Network-ready exports** — two-mode actor×concept, actor congruence, actor
   conflict, concept congruence, and GraphML.

We do **not** write `.dna` files.

### Qualifier aggregation, normalization, duplicates

The pure-Python helpers reproduce the semantics documented on `dna_network` so the
data can be verified before it reaches R:

- `qualifier_aggregation`: `ignore`, `congruence`, `conflict`, `subtract`, and
  two-mode `combine` (1 = positive, 2 = negative, 3 = mixed).
- `normalization`: one-mode `no` / `average` / `jaccard` / `cosine`; two-mode
  `no` / `activity` / `prominence`.
- `duplicates`: `include`, `document`, `week`, `month`, `year`, `acrossrange`.

The network helpers are a **reproducibility/verification layer, not a replacement
for rDNA**. The primary interoperability requirement is that our event data feeds
rDNA cleanly; rDNA remains the authority for production network construction.

### Signed concepts

Concept + qualifier are treated as distinct signed concepts
(`signed_concepts()`, e.g. `climate policy|0` vs `climate policy|1`), so
pro-support and anti-support of the same policy position are separable. Actor
congruence counts shared concepts coded identically; actor conflict counts shared
concepts coded oppositely.

### Duplicate handling

Deterministic, per rDNA policy: statement identity is the actor/concept/qualifier
tuple, optionally bucketed by document or calendar period. Duplicate policies are
only meaningful because statement ids are stable.

## Relationship to Step 6 and Step 8

- **Step 6 (Laclau/Palonen)** interprets discursive formations, nodal points,
  equivalence/difference, antagonism and populist frontier logic. Step 7 does not
  replace it: it consumes Step 6 output as analytic context for concept naming and
  interpretation while remaining empirically grounded in source evidence. Not every
  Laclau category becomes a DNA concept.
- **Step 8 (SNA)** handles evidenced social/communication relations between actors.
  DNA congruence is **not** proof of friendship, coordination, communication,
  affiliation or influence. Every statement carries `network_layer = "discourse"`
  so downstream code can distinguish discourse edges from social edges
  (`network_layer = "social"`).

## Persistence, coordination, context

- **MongoDB** (`ep24_db.country_storage`) is the durable source of truth. Step 7
  patches its fields into the cumulative `dataframe` collection (never replacing
  prior stages) and persists statements to a `dna_statements` collection. A
  run is only resumed when the stored prompt version, context hash and model all
  match exactly.
- **Redis** (`ep24_redis`, `RedisCoordinator(country, step=7)`) is optional
  coordination: per-record lock plus `running` / `completed` / `failed` status.
  Absence of Redis never breaks execution.
- **RAG**: Step 7 retrieves prior analysis as context and upserts its own
  representation through the shared `ep24_context.update_retrieval`.
- **CSV** remains a first-class checkpoint and is written atomically with the
  standard cumulative helper.

## Running it

```bash
python step_7_roihu_discourse_network_analysis.py --country finland
python step_7_roihu_discourse_network_analysis.py --country finland --limit 10
LACLAUGPT_INPUT_CSV=ep24_finland.csv python step_7_roihu_discourse_network_analysis.py
```

Also works under sbatch, the orchestrated numbered pipeline, and cron; no
interactive input is required. `--help` works with `ollama`/`pydantic` absent
(issue #152 contract).

### Demo / verification

```bash
python examples/step7_dna_demo.py /tmp/step7_demo
```

Writes the event list, two-mode matrix, congruence/conflict edge lists, GraphML and
the generated R importer for a synthetic four-actor fixture — no model, Mongo,
Redis or R needed.

## Tests

`tests/test_issue182_step7_dna.py` covers the qualifier (support / oppose /
ambiguous), actor canonicalization, concept normalization and novel concepts,
multiple actors and concepts per record, timestamps, duplicate policies,
determinism of statement ids, the event-list schema, two-mode / congruence /
conflict / normalization semantics, prompt provenance discipline, stage-contract
declaration, offline operation with Mongo and Redis absent, and the export layer.
