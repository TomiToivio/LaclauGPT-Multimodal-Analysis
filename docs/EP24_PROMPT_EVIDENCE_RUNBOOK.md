# EP24 Step 4–8 evidence, prompt and rerun gates (#241)

The cumulative CSV/MongoDB record is the **source of truth**, never a prompt.
The per-stage packet is a derived, bounded model input; it does not rewrite
researcher annotations, alter the locked Laclau/Palonen theory, alter the
Leifeld DNA schema, or change graph invariants.

| Stage | Default model-visible budget | Highest-priority evidence |
| --- | ---: | --- |
| 4 | 20,000 characters | video narrative, ASR, keyframes, OCR, metadata |
| 6 | 28,000 characters | original speech, translation, OCR, prior fusion, canonical annotations |
| 7 | 20,000 characters | direct utterance, time/source, codebook and canonical concept candidates |
| 8 | 12,000 characters | explicit actor-relation quotes and DNA statements |

Memory and corpus RAG are not proof of statements or ties. For Steps 6–8
the packet omits them entirely. Stream labels distinguish original ASR from
model-derived summaries and researcher normalization. Caps ensure a large
historical field does not crowd out primary evidence. A log line gives
model-visible characters, token estimate (chars/4) and truncation flag.
Step 4 already has its bounded packet implementation (PR #244).

## Benchmark before reruns

1. Select and save **10 source IDs each** for Finland, Poland and Portugal
   (30 total), keeping source CSVs and old outputs unchanged.
2. Run Steps 4–8 under bounded limits, only on selected records or an isolated
   copy. Record prompt version, model ID/options, counts and individual errors.
3. Manually score original-language evidence fidelity, unsupported quotes,
   omissions from OCR/ASR, DNA stance/actor/time accuracy and SNA false edges.
4. Inspect source and generated output side by side; compare old vs new on the
   **same source IDs**, including contradictory and non-political videos.
5. Only after review, consider wider runs. Do not mass force-reset MongoDB or
   overwrite historical data to make metrics appear green.

Tests are not a substitute for this manual sample review. Current changes
deliberately leave this runtime validation gate open.
