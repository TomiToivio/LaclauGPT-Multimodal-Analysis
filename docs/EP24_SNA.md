# Step 8: basic Social Network Analysis (and Castells interpretation)

Step 8 is a deliberately modest, inspectable network layer for the EP24 pipeline:
**Node – Edge – Node + basic metrics + a Castells-informed human interpretation.**
Advanced SNA (community detection, temporal or multiplex networks) is explicitly
out of scope for now and belongs in a later issue, once EP24 runs end-to-end on
Roihu.

Implementation: `roihu_sna.py` (library) and `step_8_roihu_social_network_analysis.py`
(CLI / batch entry point).

## Two layers, kept separate

1. **Relation extraction.** An LLM proposes evidence-supported ties between actors
   and writes them to `sna_edges_json`. This is the original Step 8 behaviour and is
   preserved unchanged.
2. **Deterministic graph construction.** Node and edge tables, metrics and the
   Markdown report are derived from the accumulated record plus the layer-1 edges.
   No theory is involved.

The **Castells interpretation** is applied only to the computed summary, is labelled
as interpretation in the report, and cannot add nodes or edges. `castells_interpretation`
takes a summary dict, so there is no code path by which the theory can change the
graph — a test asserts this.

## Graph model

Nodes carry: stable `id`, `node_type`, `label`, `country`, `platform`, `source_id`,
`source_url`, `timestamp`, `provenance`, and the computed `degree` / `in_degree` /
`out_degree` / `weighted_degree`.

Node types: `content`, `account`, `actor`, `theme`.

Edges carry: stable `id`, `source`, `target`, `relation`, `weight`, `directed`,
`evidence`, `evidence_quote`, `country`, `platform`, `source_id`, `provenance`.

Relations derived from the record itself:

| relation | meaning |
|---|---|
| `posted` | account → content |
| `mentions` | content → actor |
| `associated_with` | content → theme |
| `supports` / `opposes` / other | **only** from an already-extracted pipeline edge |

`supports` and `opposes` are never inferred here. A stance is exactly the kind of
claim this layer must not fabricate, so it appears only when a prior stage
extracted it with a quote.

## Identity

Stable and reproducible across runs and independent of row order. Content identity
follows the repository rule already used by `roihu_rdf.document_node`: the canonical
`video_id` wins, and the row index is only a fallback for legacy rows with no
identifier. No random or per-run ids.

Actor identity prefers the canonical entity id from `ep24_entities` when the
normalization layer resolved the mention, read from the per-mention
`ep24_entity_resolution_json` records. Two things this deliberately does **not** do:

- it does not zip the flat `ep24_entity_ids` and `ep24_entity_canonical_names`
  columns, which are written independently and are not positionally aligned — doing
  so mispaired one person with another's id;
- it does not fall back to a positional guess when no resolution record exists.

A reconciliation pass merges `actor:<hash>` nodes into `entity:<id>` nodes on the
folded canonical name, and only when **exactly one** canonical name matches. Two
people who share a surname are never merged, and the self-loops a merge would create
are dropped rather than invented.

## Metrics

Only straightforward, interpretable values: degree, in-degree, out-degree, weighted
degree, weakly-connected components, and density. A test asserts that community
detection and centrality measures are absent, so scope creep fails loudly.

## Outputs

Written under `sna/`:

- `sna_nodes_<language>.csv` — `id`, `node_type`, `label`, `country`, `platform`, `source_id`, `source_url`, `timestamp`, `provenance`, degree family
- `sna_edges_<language>.csv` — `id`, `source`, `target`, `relation`, `weight`, `directed`, `evidence`, `evidence_quote`, `country`, `platform`, `source_id`, `provenance`
- `sna_network_<language>.json` — node/edge tables plus the summary and run metadata
- `sna_report_<language>.md` — the human-readable report

The CSV names and the `id` / `label` / `source` / `target` columns are the contract
`roihu_rdf.maybe_emit_network` reads, so **Step 9 consumes these tables directly with
no ad-hoc translation**. Verified end-to-end: the tables project into Step 9 as nodes
and edges, and the resulting Turtle parses with RDFLib.

Per-row fields are appended to the cumulative CSV, additively: `sna_analysis_markdown`,
`sna_edges_json`, `sna_node_ids`, `sna_edge_ids`, `sna_summary_json`,
`sna_report_markdown`. Every incoming column is preserved; the write goes through
`ep24_pipeline.write_cumulative_csv`, which fails loudly if a column was dropped or
mutated.

## Report

The report follows the documented structure — Dataset, Network construction, Nodes,
Edges, Basic network statistics, Most connected nodes, Main observed relationships,
Castellsian interpretation, Caveats and uncertainty, Provenance / run metadata — and
explains the graph in prose rather than only dumping metrics.

The interpretation section is explicitly marked as theory-informed interpretation
rather than measurement. On small graphs it states that network-making power and
programmer/switcher roles are not identifiable, rather than claiming them.

## Running

```bash
# one language, no model calls
python3 step_8_roihu_social_network_analysis.py \
    --input /path/to/ep24_<country>.csv --language fi --country Finland \
    --output-dir sna --no-extract --no-interpret

# full run through Slurm
sbatch scripts/roihu/step_8_roihu_social_network_analysis.sbatch
```

Useful flags: `--input`, `--output-dir`, `--language`, `--country`, `--limit`,
`--no-extract` (skip the LLM edge extraction), `--no-interpret` (skip the Castells
LLM call and use the deterministic structural reading instead).

The stage never depends on an LLM being reachable: with no model the report falls
back to a deterministic structural reading that only restates measured counts.

## Tests

`tests/test_roihu_sna.py` (39 tests) and `tests/test_ep24_sna_rdf_handoff.py` (8).
Synthetic fixtures only — no private research data appears in tests or docs.
