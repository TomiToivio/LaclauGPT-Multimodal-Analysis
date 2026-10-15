# EP24 Steps 4–8: 30-source manual quality review (#241)

**Status: NOT YET PERFORMED.** This is a reproducible review form, **not** evidence of a successful benchmark. Do not close #241 or authorize force/full reruns based solely on this document.

## Before reviewing

- Freeze **exactly 10 FI, 10 PL and 10 PT distinct source IDs**, including contradictory, non-political and multilingual examples where present. Store the selection in a separate, versioned manifest of source IDs only. Never fabricate IDs to fill a country.
- Record the source dataset snapshot, selection method and random seed, commit SHA, model identifiers, prompt versions, generation options, environment and review date.
- Use an isolated dataset or a selected-ID path. Keep cumulative CSV, MongoDB source fields, researcher annotations and previous model results unchanged.
- Run Steps 4–8 for these IDs only and compare old and new outputs for the **same** inputs. Inspect primary-language ASR, original video/keyframes, OCR and provenance, not just earlier model summaries.
- Record *N/A* when a stage or source feature is genuinely absent, *not reviewed* when evidence was not checked, and *failure* for a failed execution. Do not mistake those states for a passing score.

## One row per source and stage

Use `docs/reviews/ep24-241-review.csv` or a secure local equivalent. Recommended headers:

```csv
country,source_id,step,reviewer,reviewed_at,source_snapshot,commit_sha,model_id,prompt_version,options_fingerprint,old_output_ref,new_output_ref,execution_status,original_language_fidelity,unsupported_claims,unsupported_quotes,ocr_asr_omissions,dna_actor_accuracy,dna_stance_accuracy,dna_time_accuracy,sna_false_edges,sna_missing_edges,notes,evidence_ref,pass_fail
```

Each country must have 10 **unique** source IDs. Each selected source should have a row for each applicable step (4, 6, 7 and 8; Step 5 can be annotated separately if relevant). Fields `original_language_fidelity`, `dna_actor_accuracy`, `dna_stance_accuracy`, `dna_time_accuracy` use **0 = demonstrably incorrect, 1 = partly correct, 2 = supported, N/A = not applicable**, with written examples for 0/1. Counts (`unsupported_claims`, `unsupported_quotes`, `ocr_asr_omissions`, `sna_false_edges`, `sna_missing_edges`) are non-negative integers, or N/A if not applicable. `execution_status` must distinguish success, failure and not-run. `pass_fail` is a manual judgment with an explanation; do not silently turn unknown into pass.

Source/old/new/evidence references should be **stable opaque identifiers or paths**, not private user content or raw personal data committed to Git. Any examples reproduced publicly must be anonymized and cleared for publication. Preserve the full restricted review record outside the public repository where required.

## Sign-off gate

Before ticking the issue acceptance criteria, document all of:

1. Manifest of **10 distinct FI + 10 PL + 10 PT source IDs** and selection rationale; totals and exclusions.
2. Reproducible Step 4–8 configurations and commands, execution successes/failures, context-budget/truncation telemetry, and hashes that tie results to prompt/model/options.
3. Side-by-side source/old/new evidence judgments in every applicable stage, including false statements, mistaken attribution and spurious DNA/SNA links.
4. Aggregate counts by country and stage and a qualitative error summary with concrete, traceable examples.
5. A named human reviewer and decision: **approve targeted expansion**, **fix and repeat benchmark**, or **hold**. A success threshold must be chosen and written **before** scoring, not optimized after looking at the outcomes.
6. Green CI on the candidate commit and confirmation that no source records or prior research interpretations were overwritten.

Until signed off, **no bulk force rerun, source-data reset or issue closure**. See [EP24_PROMPT_EVIDENCE_RUNBOOK.md](../EP24_PROMPT_EVIDENCE_RUNBOOK.md) for the controlling policy.
