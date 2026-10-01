# EP24 RDF projection (issue #5)

This additive stage follows the historical sequence: preprocess → frame → summary
→ postprocess → populism. DNA and SNA, when implemented, precede RDF. The frozen
legacy reference is `e011c34274c41923e801c32c824c5afa7468e1a1`.

Design reference: LaclauGPT-Data-Analysis commit
`bf80f435c0390a1d96b3cec21bda09b6c83e7ca0`,
`src/laclaugpt_data_analysis/rdf.py` and `knowledge_graph.py`.
Adapt stable project-scoped identities, PROV activities and explicit assertion
provenance to CSV/SQLite; do not import the AI26 canonical record or service stack.

RDF is a projection, never the authoritative research dataset. Keep every original
CSV field as a literal, including human edits, unknown columns and empty strings.
Interpret only explicitly structured fields. Never split legacy comma-separated
entities into presumed canonical actors or interpret sentiment as DNA agreement.

Implementation and synthetic validation are in progress on the associated PR.
A real EP24 run on CSC Roihu remains required before issue #5 can close.
