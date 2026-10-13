# EP24 Step 6: evidence-first Laclau/Palonen discourse analysis

Step 6 is the central theory-critical analysis stage of the EP24 reprocessing pipeline.
Its canonical implementation is `roihu_populism.py`, wrapped by
`step_6_roihu_discourse_analysis.py`.

## Theory contract

The stage follows the generic Laclau / Mouffe / Palonen contract in the LaclauGPT
`THEORY.md` rather than assuming that every political video contains populism.

Document-level output is provisional and evidence-linked:

- an Us requires a politically meaningful collective subject, not merely a plural pronoun;
- criticism, disagreement, negative sentiment and opponent mentions do not by themselves
  establish an antagonistic frontier;
- affect is affective investment/expression, not sentiment polarity;
- co-occurrence is not a chain of equivalence;
- nodal, floating and empty signifiers are candidates requiring evidence;
- hegemony and bipolar/hegemonic polarisation are corpus-level claims;
- Palonen fringe/mainstream/competing dynamics are provisional evidence, never party labels.

The schema explicitly permits empty lists, uncertainty, counter-evidence and abstention.
`laclau_formula_conditions_met` is true only when the current source supports both a
politically meaningful collective Us and an `antagonistic_frontier`. It is not a
populism score and does not assign a permanent identity to an actor or party.

## Evidence classes

The effective prompt is split into provenance classes:

1. recorded current-source metadata;
2. source-derived representations such as ASR/OCR;
3. human researcher annotations;
4. derived prior-stage model analysis;
5. codebook context;
6. researcher memory;
7. retrieved corpus/RAG context.

Classes 4-7 are explicitly marked as context, not direct source evidence. Researcher
`entities` and `themes` remain authoritative canonical seeds and are never overwritten.

The complete effective context is bounded by `LACLAUGPT_STEP6_MAX_CONTEXT_CHARS`
(default 28000 chars) and fingerprinted into `laclau_context_sha256`.

## Persistence

When MongoDB is enabled, Step 6 uses the shared `ep24_context.py` and
`ep24_db.py` stack:

- country codebooks and researcher memory are bootstrapped into MongoDB;
- bounded memory/RAG/codebook context is retrieved before inference;
- the cumulative row is persisted with Mongo `patch_documents`, never replacement,
  so fields produced by Steps 1-5 cannot be erased;
- successful Step 6 output is added to RAG for later stages/corpus work.

Redis coordination is optional. When `LACLAUGPT_REDIS_URL` is absent, record locking and
status calls degrade to no-ops and the stage still runs.

The old `formula_of_populism.db` SQLite cache is no longer a source of truth.

## Outputs

Legacy compatibility columns remain available:

- `formula_of_populism_analysis`
- `formula_of_populism_us`
- `formula_of_populism_frontier`
- `laclau_summary_md`

They are deterministically derived from the richer structured result.

Modern fields include:

- `laclau_structured_json`
- `laclau_formula_conditions_met`
- `laclau_abstention_reason`
- `laclau_prompt_version`
- `laclau_model_metadata_json`
- `laclau_generated_at`
- `laclau_context_sha256`
- `laclau_context_truncated`
- `laclau_codebook_fingerprint`
- `laclau_codebook_context_json`
- `laclau_memory_context_json`
- `laclau_rag_context_json`
- `laclau_raw_response`
- `laclau_status`
- `laclau_error`
- `laclau_runtime_seconds`
- `laclau_persistence_status`

Invalid structured output records the raw model response in `laclau_raw_response` and
the validation error in `laclau_error` for reproducible debugging.

## Execution

The numbered wrapper keeps the shared CLI contract:

```bash
python step_6_roihu_discourse_analysis.py --country finland --limit 10
python step_6_roihu_discourse_analysis.py --country poland --limit 10
python step_6_roihu_discourse_analysis.py --country portugal --limit 10
```

The same environment-based input/output contract remains usable from sbatch and cron.
CSV checkpointing defaults to every 10 processed records and can be changed with
`LACLAUGPT_STEP6_CHECKPOINT_EVERY`.
