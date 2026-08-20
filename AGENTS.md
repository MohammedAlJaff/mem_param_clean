# Agent Guide

Canonical instructions for agents and contributors working in this repo.

## Session Start

- Read this root `AGENTS.md` before making repository changes. Its root-level
  location makes it the canonical automatically discovered instruction file.
- Check `git status` before editing so existing user changes are preserved.
- Read `TODO.txt` and the latest relevant `MIGRATION_JOURNAL.md` entry for
  context. Do not begin a TODO item unless the current user request selects it.
- Treat the current user request as the scope for the session. Historical
  journal entries describe past decisions; they do not authorize new work.
- Within repository documentation, use `mem_param_src/config.py` for executable
  defaults, `docs/config_reference.md` for field meanings, `TODO.txt` for open
  work, and `MIGRATION_JOURNAL.md` for completed decisions and checks.

## Project Framing

Paper: "Equivalent in Principle, Unequal in Practice"

Current repo question:
- Under fixed architecture, data, optimizer, sampler, and evaluation protocol,
  how do target parameterizations (`x`, `eps`, `v`) change memorization
  dynamics in Flowmatching?

Conference framing:
- Target parameterization changes the route from generation to memorization.

Key measurements:
- `f_mem`
- nearest-neighbor ratio and distance statistics
- checkpoint-level memorization trajectories over training

## Communication Rules

- The user is a researcher. Use simple wording and precise research
  terminology.
- Do not use slang, hype wording, playful labels, or informal phrases.
- Prefer direct descriptions of methods, assumptions, metrics, results,
  blockers, and next actions.
- Keep responses short by default. Answer the question first, usually in one
  to three sentences. Add detail only when the next action would otherwise be
  ambiguous.
- Prefer short bullets over long explanations.

Examples:
- Avoid: "This is a nice little smoke test."
  Use: "This is a short runtime test that checks whether the path executes
  without errors."
- Avoid: "This metric is doing oracle shit."
  Use: "This metric uses the known synthetic data-generating distribution."
- Avoid: "The runner wires everything together."
  Use: "The runner constructs the config, launches training, and optionally
  runs checkpoint evaluation."

## Engineering Discipline

- Think before coding. State assumptions, expose ambiguity, and ask when the
  requirement is materially unclear.
- Prefer the simplest implementation that solves the stated problem. Do not add
  speculative features, single-use abstractions, unrequested configurability, or
  broad error handling for scenarios that cannot occur in the current design.
- Make surgical changes. Touch only lines that directly serve the request,
  match the existing style, and do not refactor adjacent code unless asked.
- Clean up only your own changes. Remove imports, variables, helpers, or files
  made unused by the current edit. Mention unrelated dead code instead of
  deleting it.
- Preserve user changes in a dirty worktree. Never revert changes you did not
  make unless explicitly asked.
- Define success criteria for non-trivial tasks. Use a short plan with a
  verification step for each change, then run targeted checks or state why they
  could not be run.
- Record verification with the environment, command or check, outcome, and any
  remaining limitation. Distinguish syntax checks, mocked tests, runtime tests,
  and full integrations rather than describing them all as verified.
- Push only when explicitly asked.


## Data and Artifact Rules

- Persist the exact training and reference tensors for every run.
- Persist `data_manifest.json` with source sizes, catalog hashes, and stable
  sample identifiers aligned with the saved tensors.
- Persist the exact `z_init` used for checkpoint replay.
- Post-hoc evaluation must operate on saved artifacts, not rebuilt datasets.
- Keep outputs inside the caller's chosen run directory.
- Use `torch.utils.data.Dataset` and batched loading for real training paths.
- Avoid training-time augmentation in canonical memorization studies unless the
  study is explicitly about augmentation.
- When W&B is enabled and no explicit W&B flag is passed, derive project, run
  name, and group from the resolved config. Project is dataset plus model. Run
  name includes dataset, model, `N`, batch cap, prediction/loss spaces,
  training steps, and seed. Group is dataset plus model plus training steps so
  `x`, `eps`, `v`, batch-size, seed, and `N` sweeps stay comparable.

## Python Style

- This section is the current canonical Python style guide. Do not depend on a
  separate style file unless one is later reviewed and added to the repository.
- Use `mem_param_src/model.py`, `mem_param_src/sampling.py`, and adjacent tested
  modules as the current style and notation examples.
- Prefer math-first, teaching-first comments and docstrings.
- Use paper notation directly when the paper defines the quantity: `rho`, `r`,
  `d1`, `d2`, `N`, `tau`, and `t`.
- Use canonical metric keys in return dictionaries, JSON, and W&B logs. Do not
  add duplicate aliases such as `nn_ratio`, `ratios`, or `ratio_threshold` in
  memorization metric records.
- Inline comments for named quantities should state the formula or definition,
  for example `r = d1 / d2` or `delta_f_mem = f_mem(gen) - f_mem(held-out)`.
- Canonical first-party Python should not use type hints or
  `from __future__ import annotations`.

## Notebook Style

- Use ASD-STE100 Simplified Technical English, Issue 9, for all notebook
  comments, descriptions, instructions, titles, and Markdown text.
- Use a maximum of 20 words in an instruction. Use a maximum of 25 words in a
  descriptive sentence.
- Write one topic in each sentence. Use active voice. Use the imperative form
  for instructions.
- Use one word for one meaning. Use the same term for the same object or
  operation. Avoid idioms, ambiguous words, unnecessary synonyms, and phrasal
  verbs.
- Use technical nouns and technical verbs when exact research, mathematics,
  dataset, tensor, library, API, or code terms are necessary. Do not change an
  exact identifier to satisfy the general vocabulary.
- Start every code cell with a short top comment that states the cell's purpose
  in plain research language.
- Add concise inline comments beside or immediately above most non-obvious
  lines so the reader can understand the operation, scientific assumption,
  tensor meaning, or expected result without relying on a separate Markdown
  cell.
- Markdown cells may still explain equations, interpretation, and experimental
  context, but they do not replace the top note and inline comments in code
  cells.
- Comment the reason for transforms, split selection, seeds, tensor shapes,
  metric settings, and plotting choices. Avoid comments that merely repeat the
  Python syntax.
- Keep each code cell focused on one conceptual step and keep comments concise
  enough that the code remains easy to scan.

## Environment

- For local work on this machine, use the `memcompgen` Conda environment. Prefer
  explicit commands such as `conda run -n memcompgen python -m pytest`.
- If the clean checkout has not been installed in editable mode, run commands
  from the repository root with `PYTHONPATH=.`.
- On the Berzelius cluster, load Mambaforge and activate `mem_param_env` before
  running project-dependent commands.
- Do not assume that `memcompgen` and `mem_param_env` contain identical
  dependency versions; record which environment produced each check.
- The bare system Python is not the project environment.
- Berzelius/NSC nodes may require staging datasets to local scratch.
- Pre-download weights before job submission; compute nodes may have no
  internet.

## Do Not

- Do not recreate a live dependency from `mem_param_src/` inside archived
  material.
- Do not create vague catch-all modules such as `utils.py` unless there is a
  very clear, narrow purpose.
- Do not edit vendored code under `baselines/*/vendor` or `third_party/`
  unless explicitly asked.
- Do not mutate unrelated files, generated artifacts, notebooks, job scripts, or
  experiment outputs unless explicitly requested.
- Do not push to remote or amend past commits without explicit instruction.
