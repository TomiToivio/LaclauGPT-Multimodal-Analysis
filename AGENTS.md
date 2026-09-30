# Agent rules

## Repository role

This repository has two intentionally different branches:

- `legacy`: frozen historical documentation of the original CSC Puhti / EP24 pipeline.
- `main`: active, incremental CSC Roihu adaptation.

The older phase-branch policy does not govern this repository's current migration.
Do not synchronize `main` with a phase branch unless the human author explicitly asks.

## Legacy branch is immutable

Never modify, merge into, rebase, force-push, clean up, reformat, modernize or
backport changes to `legacy`.

If historical code contains a bug or obsolete assumption, document it on
`main`; do not repair the historical record.

## Human code is authoritative

Never replace human-written code merely because a rewrite appears cleaner.

Before changing existing code:

1. understand the current behavior;
2. identify the smallest required change;
3. preserve scientific and output behavior unless explicitly asked to change it;
4. make a narrow reviewable commit;
5. validate before the next change.

Do not perform opportunistic refactors, broad formatting, renames, schema
changes, prompt rewrites, or methodology changes while doing infrastructure work.

When uncertain, preserve the human implementation.

## Public/private boundary

Public open-source code belongs here.

Private material belongs in the private `TomiToivio/LaclauGPT-Private`
repository or private CSC storage, including:

- real research data;
- researcher notes;
- private codebooks;
- private settings;
- credentials and secrets;
- restricted/unpublished research material;
- machine-specific private configuration.

Public code may define environment-variable/path contracts and dummy fixtures,
but must never copy private content into this repository to make a job or test work.

## Roihu migration sequence

Work in this order:

1. preserve and document the Puhti baseline;
2. make the existing pipeline runnable on CSC Roihu;
3. test improved Gemma4 models through configurable Ollama model selection;
4. validate every historical stage and output contract;
5. only then inspect LaclauGPT-Data-Analysis for candidate improvements;
6. port improvements one at a time only when justified.

Do not copy the newer Data-Analysis pipeline wholesale.

## Scientific boundaries

Infrastructure modernization is not permission to redesign the research method.

Do not silently alter:

- frame or summary prompts;
- Laclau / Palonen analysis logic;
- schemas or CSV field meanings;
- codebooks;
- researcher annotations;
- stage order;
- interpretation rules.

Such changes require an explicit human task.

## Development principle

**Preserve history. Preserve human code. Change one thing at a time. Validate it. Then continue.**
