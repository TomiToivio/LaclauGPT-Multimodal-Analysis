# EP24 Step 7: Leifeld Discourse Network Analysis compatibility

Step 7 turns the evidence-preserving discourse-analysis pipeline into a relational
**discourse** layer. It does not replace Step 6 (Laclau/Mouffe/Palonen) and it
does not create Step 8 social ties.

## Methodological basis

The implementation follows:

Philip Leifeld (2017), “Discourse Network Analysis: Policy Debates as Dynamic
Networks,” in *The Oxford Handbook of Political Networks*, chapter 25.
Preprint: https://eprints.gla.ac.uk/121525/

The unit is a statement connecting an actor to a concept with an agreement
qualifier at a time. Step 7 therefore preserves:

- actor / organization
- concept / proposition
- positive/support or negative/oppose agreement where explicit
- source/document identity
- time
- evidence span and source fields
- uncertainty and provenance

Ambiguous statements stay in `dna_statements_json`, but are excluded from the
strict binary DNA event-list export. Memory, RAG, codebooks, and Step 6 outputs
are normalization/interpretive context, never evidence that a claim exists in
the current source.

## Pipeline boundary

- **Step 6**: Laclau/Palonen discourse analysis and evidence-linked theoretical
  interpretation.
- **Step 7**: Leifeld DNA statement coding, actor ↔ concept.
- **Step 8**: evidence-supported social/communication relations, actor ↔ actor.

A discourse congruence/conflict relation must never be promoted into a social
tie.

## Internal schema and strict event list

The rich internal schema is stored per cumulative row in
`dna_statements_json`. Every statement gets deterministic IDs and
`network_layer = "discourse"`.

The strict CSV export uses the rDNA-oriented names:

| EP24 event-list column | DNA/rDNA role |
| --- | --- |
| `organization` | actor variable (rDNA `variable1`) |
| `concept` | concept variable (rDNA `variable2`) |
| `agreement` | binary qualifier |
| `time` | statement time |
| `document` | stable document/source identifier |
| `statement_id` | deterministic EP24 statement identifier |

Additional provenance columns are intentionally retained for reproducibility.

## Current upstream rDNA semantics

The compatibility target is the current Leifeld Lab implementation:
https://github.com/leifeld-lab/dna

`rDNA::dna_network()` supports:

- `networkType = "twomode"`, `"onemode"`, or `"eventlist"`
- `variable1 = "organization"`
- `variable2 = "concept"`
- `qualifier = "agreement"`
- one-mode qualifier aggregation `ignore`, `congruence`, `conflict`,
  `subtract`
- two-mode qualifier aggregation including `combine`
- normalization `no`, `average`, `jaccard`, `cosine`
- duplicate policies `include`, `document`, `week`, `month`, `year`,
  `acrossrange`
- temporal start/stop filters and moving windows
- CSV, UCINET DL, and GraphML output

EP24 deliberately does not reimplement the full DNA network engine. The
`ep24_dna.actor_projection` helper is a small reference implementation used to
test congruence/conflict/subtract and normalization semantics.

## Official import path

Do not reverse-engineer the `.dna` SQLite schema. Current rDNA exposes official
data-access functions including `dna_addDocuments()` and
`dna_addStatement()`. After opening/creating the target DNA database through
the supported DNA/rDNA workflow, load the EP24 event list and add statements
using those APIs.

Conceptual R mapping:

```r
events <- read.csv("..._dna_eventlist.csv", stringsAsFactors = FALSE)

# Create/get DNA documents with dna_addDocuments(), retaining a lookup from
# EP24 document -> DNA numeric document ID.

for (i in seq_len(nrow(events))) {
  e <- events[i, ]
  dna_addStatement(
    documentID = document_id_lookup[[e$document]],
    statementType = "DNA Statement",
    organization = e$organization,
    concept = e$concept,
    agreement = as.logical(e$agreement)
  )
}
```

The official `dna_addStatement()` API converts logical qualifier values to
the integer representation required by DNA. Document timestamps belong on the
DNA documents, so the importer must create documents with the event/source time
via `dna_addDocuments(..., date_time=...)`.

Once imported:

```r
congruence <- dna_network(
  networkType = "onemode",
  statementType = "DNA Statement",
  variable1 = "organization",
  variable2 = "concept",
  qualifier = "agreement",
  qualifierAggregation = "congruence",
  normalization = "average"
)

conflict <- dna_network(
  networkType = "onemode",
  variable1 = "organization",
  variable2 = "concept",
  qualifier = "agreement",
  qualifierAggregation = "conflict"
)
```

This official API route is preferred to generating a `.dna` database directly.

## Persistence

When Mongo is enabled, Step 7:

1. patches the full cumulative row into the standard country-scoped
   `dataframe` collection;
2. upserts statement-level records into the country-scoped
   `dna_statements` collection;
3. upserts the stage into standard RAG;
4. still checkpoints the cumulative CSV and strict event-list CSV.

Redis is optional and used only for per-record locks/status. Mongo is durable
shared storage; CSV remains the inspectable/reproducible interchange backup.
