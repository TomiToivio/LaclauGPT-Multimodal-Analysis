# EP24 FI/PL/PT export QA and producer contract (#330–#333)

The EP24 Mongo export is **one canonical parent row per video** when
`--record-unit video` is selected. DNA statements persisted by Step 7 are
**ingest-level children**: child `_storage_id` is a statement identifier and
**must not** be joined to the parent's `_storage_id`.
The child `source_record_id` refers to the source ingest row.

For freshly generated exports, `children/<country>_dna_statements.csv`
contains `parent_record_id` (same identity as the canonical parent's
`record_id`) and `parent_country`. Only child statements belonging to the
selected canonical ingest parent are emitted. This prevents a child statement
from a discarded ingest run from being counted against a different video
projection. Other child roles retain their historical format pending an
equivalent collection-specific identity audit. Avoid treating their counts
as video-unit counts without checking joins.

The manifest includes `children_unit`, `child_parent_join_key`,
`child_join_qa` and `heavy_fields`. A `child_join_qa` exclusion is
explicit data loss from the **canonical projection**, not a claim that an
upstream statement was invalid. Consumers should report excluded numbers.
`children.finland.dna_statements` reports only matched exported child rows,
not all ingest-level Mongo documents.

## Long text / coverage

Index CSV columns with string values over the configured
`--text-threshold` (default 1,000 characters) are omitted from that
record's CSV index cell. Their full values are retained in
`<country>_text.jsonl.gz`, keyed by `record_id`. The manifest's
`heavy_fields.<country>.<field>` gives the number of elided records.
An empty CSV cell therefore does **not** imply a pipeline stage failed.
The DNA coverage calculation recognizes a positive
`dna_statement_count` as evidence of Step 7 output even if the JSON
was moved to the bundle. It does not count zero as a substantive result.
The manifest coverage is calculated before text splitting.

## Verification and remaining gates

This patch contains synthetic regression tests; there is **no assertion**
that the historical private 1,583 FI child rows have become 1,490.
To verify on an authorized MongoDB endpoint: rerun the read-only exporter,
compare counts in the new manifest, inspect `child_join_qa`, confirm the
expected FI/PL/PT per-stage matrix, then check the Visualization Step 7
reader against the new `parent_record_id` key. The downstream
Visualization consumer must support that key before relying on the new
contract. Keep the original export for reproducibility.

Issue #333 additionally requires a genuine stratified quality audit of
language-specific codebooks, prompts, grounding and missing stages across
Steps 0–9. No responsible prompt or codebook change can be claimed
without authorized corpus samples and a before/after researcher evaluation.


## v2 follow-up and defect register

The authoritative v2 contract is now in `EP24_MONGO_EXPORT.md`. It supersedes the
initial PR notes above: `children_unit` is now per role, and requested DNA coverage
is reconciled against emitted children, not only a positive source counter.

| Defect | Verified source cause | Repair / remaining gate |
| --- | --- | --- |
| Unreachable DNA | Child storage ID differs from source ingest; video projection drops ingests | Explicit compound parent key; selected-ingest filtering; exclusion QA |
| Duplicate/ambiguous joins | Initial PR used last-write-wins parent map and retained duplicate statements | Quarantine ambiguity and duplicate identities; synthetic tests |
| Silent text loss in index | Long cells omitted without row-level marker | `_text_bundle_fields`, field counts, threshold metadata, protected identity fields |
| False global child grain | Entities/themes can contain registries/codebooks; SNA stores graph IDs | Per-role units; preserve source collections without inventing video joins |
| Coverage disagreement | Row counters can describe children that are absent/unjoinable | Child-derived DNA evidence coverage; preserve source coverage and flag counter mismatches |
| Stale child files | Reusing destination can retain files when a collection disappears | Require empty export destination |
| Unknown stage states | Missing output and unknown status do not establish failed/not-run | Conservative status matrix, including explicit unknown buckets |

### Live acceptance gates still open (#333)

No authorized Mongo/Redis/Roihu configuration or corpus export was present in the
implementation workspace. No real record, private codebook, log or RDF file was
inspected; no model call or reprocessing job was run. The reported historical
counts in the issues are context, not a newly verified measurement. #333 remains open.

1. In private storage, identify the latest complete intended run using job/run
   records and pipeline commit, not newest row timestamp or canonical stage score.
   Record collection/checkpoint and prompt/model/schema versions. Reject mixed-run
   claims until the operator has verified the selection.
2. Export to a new private directory and inspect manifest `stage_states`,
   `stage_coverage`, `child_join_qa`, `heavy_fields`, provenance and hashes. Reconcile
   distinct parent counts separately from statement counts and inspect every exclusion.
3. Stratify FI/PL/PT samples by language/platform/stage state. Link private failure
   logs to source input and checkpoints; verify RDF files and provenance separately.
4. Diagnose confirmed producer failures before choosing records/steps to rerun.
   Preserve the original export and historical outputs; record new run IDs and
   compare failures, usable output, cost and unintended changes before deployment.
5. Evaluate proposed prompt/codebook edits on a fixed researcher-reviewed sample:
   quote fidelity, attribution, evidence-linked statements, abstention and disagreement.
   Keep restricted aliases and codebooks private. No prompt change is justified yet.
6. Verify the new export with Visualization #222's successor reader before closing
   the live acceptance gates. Frequency/centrality alone is not hegemony or a real tie.
