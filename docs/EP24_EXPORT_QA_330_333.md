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
