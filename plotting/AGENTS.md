# Shared plotting workspace

Read `figure_catalog.json` and `../docs/local_plotting_handoff.md` before edits.
All agents use this plotting directory and `figures.py` as the shared entrypoint.

- Current paper drafts go to `../outputs/paper_figures/`. Leave the earlier
  `../outputs/figures/` collection intact. Work on one requested paper item
  or an explicitly selected small group; do not run `--all` for routine edits.
  Consult `../docs/paper_figure_plan.txt` and the new folder's `README.txt`.
  Generate and check the selected item, then return it for focused review.
- Use `paper_style.py` for typography and parameter labels: DejaVu Sans text,
  STIX math, and bold italic `\boldsymbol{x}`, `\boldsymbol{\epsilon}`, and
  `\boldsymbol{v}`. Import `TARGET_LABELS` and `TARGET_PAIR_LABELS`; do not
  redefine them in a renderer. Use `LATEX_PREAMBLE` for standalone LaTeX
  figures. Templates may set font sizes, but keep the shared font families.
- M4 / Table 2 is the researcher's explicit exception: retain the supplied
  original LaTeX styling and inherit manuscript fonts and named table colors.
  Do not impose the shared font families or plot palette on this table.
- M5 is now an explicitly requested reproduction of workshop Figure 4.
  Preserve the vector artwork and its historical rho=0.5, first-1% onset,
  SE, and final-checkpoint substitution; explain these in the caption.
  Its source asset is under references/. Do not relabel it as current data.
- Use the researcher's paper names in conversation, commands, and files:
  M1 setup, M2 diagonal results, M3 matrix, M4 robustness table, M5 timescales,
  M6 theory; A1 trajectories, A2 representation agreement, A3 extended timescales, A4 weighting,
  A5 threshold sensitivity. These replace the former type-based IDs.
  Table 1's conversions are a companion to M1, named M1_CONVERSIONS.
- Give a distinct layout, dataset, metric, or uncertainty alternative a
  descriptive suffix within its paper group, such as A3_DIT. Keep the
  `variant` descriptive and short. Do not introduce another naming system.
  Routine adjustments to an existing design retain its ID and create a new build.
- Record data sources, metrics, rho, seed coverage, uncertainty, and checkpoint
  selection. Existing renderers have fixed scientific contracts; changing a
  metadata label cannot implement a new metric. Add the supported renderer first.
- Keep main/appendix/unassigned placement separate from draft/approved/rejected
  review status. Record approval only after the user explicitly approves an
  exact build. Do not infer approval from a successful render or a style request.
- Use only `plot_data/` for newly computed numerical input. M5 preserves an
  explicitly selected historical graphic without reconstructing its values.
  Preserve metrics, exported CSVs,
  cluster provenance, checkpoints, and evaluation records. Adapt external code
  as a visual reference; do not assume its statistics or CSVs match this study.
- Keep variants within a paper group consistent. Apply timescale style edits to
  both A3_UNET and A3_DIT when that group is selected. Preserve data-dependent
  uncertainty and censoring.
- Build with `conda run -n memcompgen python plotting/figures.py ID` from the
  repository root. Inspect the PDF and its build record. Use `--show ID` for
  the owning renderer. Use commands such as `figures.py M2` or `figures.py M5`.
- Preserve both output folders' `versions/` archives. Build records hash inputs and all plot
  code and store a source archive. Any plot-code edit conservatively makes
  older receipts stale; stale approval appears as needs-review. Never silently
  update an approved build ID or modify the archived PDF/source.

For quick edits, change the existing template; avoid independent copied scripts
that bypass the catalog. Existing figures start draft/unassigned until reviewed.
