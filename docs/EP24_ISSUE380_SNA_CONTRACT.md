# Issue #380: empirical SNA and cross-stage graph identity

## Method

Reference: Borgatti, Everett & Johnson, *Analyzing Social Networks*, 2nd ed.,
chapters 1–5, 7, 9–11, 13–15. The privately supplied book is not redistributed.
Step 8 describes observable, quoted and source-attributable interaction events.
A shared theme, stance or discourse opposition never generates a social edge.
Communication mentions are directional references, not reciprocal contacts.
Institutional affiliation is a separate relation family, never mixed into
actor-to-actor communication metrics by default.

An EP24 record/video is the observation unit; the collection/country/period and
inclusion criteria define the population boundary. A one-video fragment is not
a whole-network sample. Event records retain video ID, date, platform, evidence,
relation family, direction and confidence. Parallel source events are preserved.

## Activation

The ordinary numbered Roihu pipeline includes Steps 1–9. Step 0 remains an
import/bootstrap operation. Step 8 makes one evidence-extraction model call per
processed non-empty row; deterministic code validates relations and computes
measurements. Step 9 is a deterministic CPU RDF export. No batch reprocessing
is automatically triggered by this source change.

Historical Castells interpretation fields are retained for compatibility but
cleared in Step 8's newly processed rows. Step 6 and Step 7 methodologies are
independent and unchanged.

## Identity and RDF

Every video/document is the shared provenance anchor, even if no real actor can
be matched. A resolved `entity:<id>` from upstream entity resolution maps to
one `/actor/<digest(id)>` RDF identity across DNA and SNA. Local actor labels,
roles and provisional identifiers keep their former step-specific URI. Legacy
DNA and SNA URI aliases link to canonical identities only for resolved IDs.

Step 6 abstract 'us'/'them' codings are NOT automatically people or organizations;
only explicit identified actors may be joined with the other stages. DNA
actor–concept agreement/opposition is a discourse coding, not a social tie.

## Aggregation and visualization boundaries

A downstream visualization service should read typed persisted event and DNA
statement records. Deduplicate events by event ID; group repeated directed ties
by relation family and source-item IDs, and expose the temporal selection.
Offer distinct Step 6 coding, Step 7 actor–concept, Step 8 actor–actor and
combined multilayer views. The combined heterogeneous graph is not itself an
ordinary one-mode SNA network. Recompute metrics after filtering, and retain
provenance drilldown to the originating record and source quotation. Export
GraphML/GEXF/CSV/JSON/RDF with explicit layer fields. Avoid publishing data or
cross-country identity matches without evidence and the relevant data policy.

**Implementation status:** The existing analysis repository has a Step 8 typed
pair CSV aggregator and Step 9 RDF projection. Full filtered interactive graphs
belong in the separate Data Visualization application; this file defines the
interoperability boundary, not a claim that its UI is already delivered.

## Read-only cross-video event views

`ep24_sna_aggregate.aggregate_social_events` accepts cumulative persisted
records and returns a deterministic, relation-specific event view, weighted
projection, filtered metrics and source-item/evidence drilldown. Resolved
`entity:` IDs join across videos; all other IDs are scoped to their document.
Repeated copies of an event ID count once; conflicting copies fail closed.
Missing evidence and malformed direction/confidence are rejected and counted.
This validates the persisted evidence field, not the truth of its quotation.

The CSV adapter never connects to MongoDB or submits jobs:

```bash
python scripts/ep24/export_social_network.py /private/fi.csv /private/pl.csv \
  --relation mention --country finland --start 2026-01-01 --end 2026-12-31 \
  --output /private/network_snapshots/mentions_001
```

Install `networkx>=3,<4` for GraphML/GEXF export. The destination must be new;
files are private (0600), with a 0700 output directory and SHA-256 manifest.
JSON contains the complete boundary, metrics, events, nodes and projection;
CSV tables and GraphML/GEXF event graphs support downstream visualization.
GraphML/GEXF use a directed container with the original `directed` event
attribute preserved. The JSON projection expands undirected events to two arcs.
Weights count distinct events; self events are retained but excluded from the
simple loopless density. Isolates absent from selected events are not inferred.
Dates are inclusive ISO timestamps; an end date alone means midnight. Missing
or invalid dates are excluded when a time filter is active. Actor filtering
selects incident events, suitable for an ego view. Confidence defaults to zero
when absent, and is excluded by a positive minimum-confidence filter.

This delivers the Step 8 aggregation/export interface. Step 6/7 multilayer
aggregation, interactive application filters and RDF aggregation remain separate
acceptance work; the existing Step 9 exporter still supplies per-record RDF.
