# From a video to an analytical claim — a researcher's guide

This page answers one question: **if I want to inspect or challenge an analytical decision, where do I
look?** It is written for someone who reads code occasionally and social science constantly. The
generated [`REPO_MAP.md`](REPO_MAP.md) lists every module; this page explains the pipeline the modules
form.

The pipeline turns collected media (videos, posts, pages) into a cumulative **dataframe** — one row per
item, one column group per stage — and then into a coding you can analyse. Every stage is **additive**:
source metadata and upstream columns are carried forward unchanged, and a stage only appends its own
columns. That is the rule `ep24_pipeline.py` enforces, and it is why you can always see what was known
before a stage made its judgement.

## The ten stages

| # | stage | code | what it decides | GPU |
| ---: | --- | --- | --- | --- |
| 0 | import | `step_0_roihu_import.py` | seeds the Mongo dataframe from private canonical CSVs; never overwrites an existing row | no |
| 1 | preprocess | `step_1_roihu_preprocess.py` → `roihu_preprocess.py` | stages media from Allas, takes **one** keyframe at t = 1.0 s, OCRs it, runs speech recognition over the whole video | yes |
| 2 | frame | `step_2_roihu_frame.py` → `roihu_frame.py` | multimodal reading of that one keyframe, with the accumulated row passed as context | yes |
| 3 | video | `step_3_roihu_video.py` | whole-video analysis with a video-language model (wraps `experiments/vllm_video_test.py`) | yes |
| 4 | summary | `step_4_roihu_summary.py` → `roihu_summary.py` | social-semiotic summary and fusion of the row's multimodal material | yes |
| 5 | postprocess | `step_5_roihu_postprocess.py` → `roihu_postprocess.py` | structured entities, themes, sentiment — stored **separately** from the human-coded columns, followed by the quality gate | yes |
| 6 | discourse analysis | `step_6_roihu_discourse_analysis.py` → `roihu_populism.py` | Laclau/Mouffe/Palonen coding: articulation, us/them, frontier, formula of populism, evidence-linked JSON, abstention | yes |
| 7 | discourse network | `step_7_roihu_discourse_network_analysis.py` | Leifeld-style DNA statements: one actor–concept statement with an explicit support/oppose qualifier **where evidenced** | yes |
| 8 | social network | `step_8_roihu_social_network_analysis.py` | evidence-supported actor-to-actor ties; DNA congruence is not treated as a social tie | yes |
| 9 | RDF export | `step_9_roihu_rdf.py` → `roihu_rdf.py` | deterministic, CPU-only export to RDF; emits no dataframe columns | no |

`ep24_stage_contract.py` holds this table in machine-readable form — module, entry point, sbatch script,
the columns each stage appends, and a note on what it does. **It is the source of truth**; if this page
and the contract disagree, the contract is right and this page is a bug.

## What is a model doing, and what is a rule?

Three different kinds of judgement end up in the same dataframe, and the column names tell you which:

| kind | how to recognise it | who decides | can it be wrong in the interpretive sense? |
| --- | --- | --- | --- |
| **human-coded** | `political_preference`, `entities`, `themes`, `researcher_note` | the researcher, outside the pipeline | no — it is the reference |
| **model-generated** | `frame_analysis_*`, `summary_*`, `vllm_video_*`, `formula_of_populism_*`, `laclau_*`, `dna_*`, `sna_*` | an LLM, per stage, from a prompt you can read in that stage's module | yes |
| **deterministic** | `*_quality_status`, `*_quality_reason`, `*_context_sha256`, `rdf_*`, aggregates | code — thresholds and rules you can test | no, but the **threshold choice** is a methodological one |

The quality gate (`laclaugpt_quality.py`) is the clearest example of the third row: it returns
`OK` / `REPROCESS` / `DELETE` for steps 2–5 by rules, and those rules decide which rows survive into the
next stage. If a finding looks like an artefact, the gate is one of the first places to check.

## Provenance: how to trace a claim back

Three kinds of trace exist, and they answer different questions:

1. **Which context did the model see?** Context hashes are appended per stage
   (`frame_analysis_context_sha256`, `dna_context_sha256`, and the prompt contract in
   `ep24_prompt_contract.py`). Equal hashes mean equal context; a changed hash between two runs means the
   model saw something different, not that it reasoned differently.
2. **Which prompt and codebook produced a coding?** The prompt text lives inside the stage module today
   (see the *Prompt sites* section of [`REPO_MAP.md`](REPO_MAP.md)); the codebook is loaded from the
   private registry (`roihu_codebooks.py`, `roihu_codebook_sources.py`) and its identity resolution is in
   `roihu_identity.py`.
3. **Which evidence supports a statement?** Step 6 and 7 outputs are evidence-linked JSON
   (`laclau_structured_json`, `dna_statements_json`): each coding carries the span it rests on, and
   abstention is a recorded outcome (`laclau_formula_conditions_met`, `uncertainty`) rather than an
   empty cell.

If a claim cannot be traced through all three, treat it as unverified rather than as a finding. That is
the same rule the audit tools in the sibling repositories apply.

## Running, inspecting, resuming

```bash
python step_6_roihu_discourse_analysis.py --help     # every numbered step shares one CLI contract
python step_6_roihu_discourse_analysis.py --limit 5  # bounded run over five eligible rows
```

- **Running a stage** always goes through `ep24_cli.configure_step_cli`, which resolves the stage number,
  the private settings, and the cumulative dataframe paths. Two wrappers that behave differently are a
  bug, not a preference.
- **Which rows a stage touches** is decided by `ep24_stage_orchestrator.py` from database state
  (`ep24_stage_state.py`, `ep24_eligibility.py`) — not by the CSV order. A row skipped by an eligibility
  rule is `not eligible`, which is different from failed.
- **Resuming** is safe by construction: stages are additive and idempotent over their own columns;
  `ep24_backups.py` writes atomic checkpoints for the cumulative dataframe.
- **Debugging a single row** means reading its columns before and after the stage, plus the log line
  `ep24_logging.py` writes for that stage. If the two disagree about what the stage did, the log is the
  one to trust about the run, the columns about the result.

## Glossary

| term | meaning in this pipeline |
| --- | --- |
| **articulation** | the linking of demands/signifiers into a discourse; coded per row, not per corpus |
| **nodal point** | a signifier that anchors a discourse and organises the others around it |
| **empty / floating signifier** | a signifier whose meaning is under contest; "floating" when it is claimed by competing discourses |
| **antagonism** | a relation in which an identity is constituted by blocking another; **not** any conflict or obstacle the model notices |
| **formation** | a discourse formation: the pattern of articulations the coder identifies in the material |
| **imaginary** | the horizon that makes a formation seem natural or desirable |
| **formula of populism** | Laclau's criteria for a populist articulation (equivalence chain, frontier, unfulfilled demand) — recorded as conditions met, not as a verdict |
| **DNA statement** | a discourse-network unit: one actor, one concept, one support/oppose qualifier, one evidence span |
| **arena** | the sampled stratum a source belongs to (elites, media, public, …); a sampling fact, never an ideology label |
| **codebook** | the private, bilingual set of category definitions and aliases the coding is validated against |
| **evidence span** | the quoted fragment a coding rests on, located in the source text |
| **abstention** | the model declining to code because the evidence does not support it — a recorded outcome, not a missing value |
| **quality gate** | the deterministic `OK`/`REPROCESS`/`DELETE` decision between stages 2 and 5 |
| **cumulative dataframe** | the single CSV/Mongo table where each stage appends its columns to the same rows |

## Where to challenge a decision

| question | where to look |
| --- | --- |
| Why does this row have a coding I disagree with? | the row's `*_structured_json` and its evidence spans; then the stage prompt in that module |
| Why was this row dropped? | `*_quality_status` / `*_quality_reason`, then the gate rules in `laclaugpt_quality.py` |
| Why did this row never run? | stage state and eligibility (`ep24_stage_state.py`, `ep24_eligibility.py`) |
| Did the model see the same thing last run? | the `*_context_sha256` columns |
| Is this an artefact of the sample? | the arena/eligibility columns, and the coverage audits under `scripts/ep24/` |
| Where is the code for a stage? | [`REPO_MAP.md`](REPO_MAP.md), or `ep24_stage_contract.py` |

The goal of the current refactor (#411) is that every answer in that table stays readable from the
documentation and the code, without a maintainer in the loop.
