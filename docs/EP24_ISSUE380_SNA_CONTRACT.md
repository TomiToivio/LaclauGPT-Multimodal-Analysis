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
