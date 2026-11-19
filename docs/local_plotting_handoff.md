# Local plotting handoff

## Rebuttal handoff: 2026-11-19

The current code and notebooks are saved on `codex/portable-plot-data`.
New rebuttal work uses `codex/rebuttal`. The fixed paper-file archive is
`paper_snapshots/2026-11-19/paper_figures.zip`. Its manifest lists file hashes.
This archive preserves the local workspace. It does not identify the
submitted manuscript or approve additional figure builds.

Use a separate Berzelius checkout at
`/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_param_rebuttal`.
Inspect the existing checkout and jobs first. Keep its files and results.
If the new destination already exists, inspect it before making changes.

From the new checkout, load Mambaforge and activate `mem_param_env`.
Confirm `git rev-parse HEAD` matches `origin/codex/rebuttal`.
Check imports for `torch`, `numpy`, `matplotlib`, `pytest`, `nbformat`,
`ipykernel`, `jupyterlab`, and `mem_param_src`. Record package versions.
Check that `mem_param_src.__file__` points into the new checkout.
Use `PYTHONPATH=.` from this checkout to avoid an older editable install.

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_best_fid_summary.py tests/test_figure_catalog.py \
  tests/test_joint_trajectory_grid.py tests/test_memorization_sensitivity.py \
  tests/test_robustness_table.py
```

Expected result: 28 tests pass. This check passed locally in `memcompgen`.
It has not yet run on Berzelius. SSH authentication currently fails.
Read existing data from its original paths. Save new results under the
new checkout's `outputs/` directory. Select a reviewer question before
creating or running a new experiment.

## Current paper naming and workflow

Use the names from the researcher's figure plan in conversation, commands,
PDF names, and the index. These replace the former type-based figure IDs.

| ID | Purpose | Current generator |
| --- | --- | --- |
| M1 | Setup schematic | Done; handmade by the authors |
| M2 | Diagonal dynamics and best-quality summaries | `plot_diagonal_summary.py` |
| M3 | Prediction/loss matrices | `celeba32_3x3/plot_best_fid_matrices.py` |
| M4 | Robustness table: raw pixels and calibrated features | `plot_robustness_table.py` |
| M5 | Original workshop timescales, revised caption | `plot_workshop_timescales.py` |
| M6 | Linear/RFN random-initialization gaps | `notebooks/theory/theory_notebook_from_claude.ipynb`, combined M6 cell; not registered in catalog |
| A1 | Joint FID/memorization grids: A1 U-Net and A1_DIT | `celeba32_3x3/plot_f_mem_grid.py` |
| A2 | Representation comparison at best FID | `plot_memorization_sensitivity.py` |
| A3 | Extended timescales and sensitivity | `plot_diagonal_timescales.py`; `A3_UNET` and `A3_DIT` |
| A4 | Diagonal targets and equivalent velocity-loss weights, following F07 | `celeba32_3x3/compare_weighted_velocity.py`, diagonal mode |
| A5 | Raw-pixel sensitivity at best FID; A5_TRAJECTORIES over training | `plot_memorization_sensitivity.py` |

The reference PDF's Table 1 is M1's companion, `M1_CONVERSIONS`. Its Figure 4
is the accepted main-text layout for M5. M4 means the robustness table.
M6 is the planned theory addition. A3 holds extended appendix timescales. Descriptive
suffixes identify variants inside these groups, for example `A3_DIT`.
M5 now reproduces the selected workshop artwork with a revised caption.
The crop and source provenance are in `plotting/references/`; its receipt
hashes and archives both. Original conventions remain rho=0.5, first 1%
onset, SE over three seeds, and final-checkpoint substitution for unobserved
onsets. The caption explains the resulting plateaus and capped gaps.
The legacy `F12` and `TS01` commands resolve to this selected M5.
The preceding current-data M5 is archived at `versions/M5/1632be7bf6a4/`.
Its renderer is now available as `A3_UNET`, alongside `A3_DIT`: three panels
on a 6.75-by-1.95-inch canvas, rho=0.3, sustained 5% onset, SD and explicit
lower bounds. M5's reproduced artwork is an explicit exception to these rules.
A2 and A5 are first CelebA32 U-Net drafts, each with one panel per training
size and mean +/- sample SD. A2 compares calibrated pixels, SSCD, LPIPS,
and DINOv2 at alpha=0.01. A5 varies fixed pixel rho=0.3/0.5/0.7. Both keep
the same best-FID checkpoint per seed. They compare aggregate rates, not
per-image flags, and do not test sensitivity of the onset definition.
`A5_TRAJECTORIES` adds the workshop's 3x3 trajectory grid: target rows and
rho columns, using all 72 current checkpoints (71 positive steps displayed),
mean/sample SD, and four training sizes. Zero means remain in aggregation
and are omitted on log axes; SD bands are clipped to display limits. Small
markers keep isolated positive observations visible. Target colors and N
shades/line styles follow M2. The existing A5 best-FID summary is retained.
The historical stacked F12/F13 layouts described below remain archived.
`A1` and `A1_DIT` now reproduce the workshop's joint-axis convention in
full prediction/loss grids. All four N values share each panel, distinguished
by shades. FID is dotted on the left; raw-pixel F_mem at rho=0.3 is solid
on the right. Neither metric uses markers. `GRID_SIZE_COLORS` in the grid
renderer owns the four shades
per target, also shown in the legend (darkest N=512, lightest N=4096).
Axes match across both models: log training steps, linear
FID 0-600, and linear F_mem 0-1.03. U-Net has three-seed mean/sample SD;
DiT has seed 0 and no uncertainty bands. Both use 71 positive checkpoints
and write PDF/SVG plus plain-text/LaTeX captions, without numerical exports.
`A4` follows the researcher's latest F07 selection: x/x, eps/eps, v/v,
and explicit native-v/v weights (1-t)^2 and t^2. Columns use N=512/2048;
FID appears above raw-pixel f_mem at rho=0.3. Means and sample SD use three
seeds, with repeated unit controls counted once. All 71 positive steps are
shown. The shared two-column legend sits above the panels and names both
equivalent weights. PDF/SVG and text/LaTeX captions share the A4 stem.
The earlier matched-row drafts remain in A4's version archive. A4_DIAGONAL
retains the original comparison and legacy aliases; prior F07 exports are intact.

```bash
# Build only the F07-style weighting comparison.
conda run -n memcompgen python plotting/figures.py A4
```

The command writes `outputs/paper_figures/A4.pdf`, `.svg`, caption files,
and a build receipt. The saved data hashes and exported settings match.
The diagonal comparison tests whether matching the induced loss weights also
matches observed quality and memorization; it does not equate network outputs.

The specification is [paper_figure_plan.txt](paper_figure_plan.txt).
M1 is complete. The researcher approved M2 build `43d9a790863d`.
The researcher also approved M4 build `515a0307ff37`: the compact robustness
table with a metric-description column and means only.
M3's current draft adapts Figure 3 on page 7 of the supplied paper PDF:
wide cells, colored target labels and diagonal borders, 8% diagonal color
tints, and neutral off-diagonals. It reports raw-pixel percentages at rho=0.3
and sample SD, with gray paired FID beneath. Its 8.8-inch width matches M2;
height is 2.5 inches. Build it with `python plotting/figures.py M3`.
Two requested variants reuse that renderer: `M3-SSCD` reports U-Net SSCD at
held-out-calibrated alpha=0.01 with three-seed sample SD; `M3-DIT` reports
Small-DiT raw pixels at rho=0.3, seed 0 only, without uncertainty estimates.
Both use N=512/2048 and keep the same layout and width. They remain drafts
with manuscript placement unassigned. The headline M3 remains raw-pixel U-Net.
Newly computed numerical inputs come only from `plot_data/`; M5 is the
explicitly requested historical artwork reproduction. Keep source CSVs and evaluation
records intact. Main results use raw-pixel memorization at rho=0.3, per-seed
best-quality checkpoint selection, and mean +/- sample SD. DiT has one seed.

New drafts go to `outputs/paper_figures/`. The earlier `outputs/figures/`
collection remains intact. Work on one selected item or a selected small group.

```bash
# Generate M2 only.
conda run -n memcompgen python plotting/figures.py M2

# Inspect its settings and generating code.
conda run -n memcompgen python plotting/figures.py --show M2

# Generate M5: the main timescale figure.
conda run -n memcompgen python plotting/figures.py M5

# Build current-data appendix timescales with the shared template.
conda run -n memcompgen python plotting/figures.py A3_UNET A3_DIT

# Build the matched representation and fixed-rho comparisons.
conda run -n memcompgen python plotting/figures.py A2 A5

# Build the full trajectory threshold-sensitivity companion only.
conda run -n memcompgen python plotting/figures.py A5_TRAJECTORIES

# Build the paired full-grid joint trajectories, U-Net and Small-DiT.
conda run -n memcompgen python plotting/figures.py A1 A1_DIT

# Generate a selected pair.
conda run -n memcompgen python plotting/figures.py M2 M3

# Generate only the requested M3 variants: PDF/SVG, captions, and build receipts.
conda run -n memcompgen python plotting/figures.py M3-SSCD M3-DIT

# Generate M4 / Table 2 only: PDF preview, manuscript LaTeX, captions, receipt.
conda run -n memcompgen python plotting/figures.py M4
```

Outputs use the same names: `M2.pdf`, `M2.svg`, `M2.caption.txt`, and
`M2.build.json`. The new folder's `README.txt` records the review list;
`INDEX.md` links generated items. Older IDs are accepted only as command
aliases and do not create additional alias PDFs. Do not use `--all` during
this focused revision.

The tracked `plotting/figure_catalog.json` owns IDs and scientific settings.
`plotting/figures.py` calls existing generators and preserves exact builds under
`outputs/paper_figures/versions/`. Build records include input and code hashes
and a source archive. Keep archived builds intact. A change in code, input,
or the reviewed PDF makes an earlier build require review. Only explicit user
approval of an exact build permits an approved status.

For M3 variants, `summarize_f_mem_at_best_fid.py` selects each seed's own
lowest-FID saved checkpoint, then pairs the selected memorization metric.
`plot_f_mem_grid.py` reads the exported fixed-rho or calibrated-alpha rows.
`plot_best_fid_matrices.py` formats percentages and FID in the shared cells.
SSCD changes both representation and threshold rule relative to headline
raw pixels; a controlled representation comparison also needs calibrated pixels.

M4 retains Table 2's four dataset/model columns at N=2048 and adds a left-hand
metric-description column. Its first row uses
raw-pixel f_mem at fixed rho=0.3 everywhere. A second row uses held-out-calibrated
feature f_mem at alpha=0.01: SSCD copy-detection descriptors for CelebA, and
a frozen encoder trained to reconstruct digit images and classify their labels
for MNIST. A gray quality row shows Inception FID or MNIST feature-space Frechet
distance. The caption defines these representations, thresholds, and seed counts.
The digit encoder is documented in [image_evaluation.md](image_evaluation.md);
its joint training objective is in `jobs/berzelius/mnist_jointae32/train.py`.
The external SSCD definition comes from its
[official repository](https://github.com/facebookresearch/sscd-copy-detection).

`plot_robustness_table.py` reuses `plot_fid_by_training_size.py:load_records`
to validate the accepted quality protocols. It selects each seed's lowest
saved quality value (earliest exact tie), joins both memorization metrics at
that step, and computes means/sample SD. The compact table displays means only,
matching the supplied original; Small-DiT remains seed 0. The previous SD table
is archived under `outputs/paper_figures/versions/M4/fa972b2246c8/`.
`M4.tex` is a table fragment, usable with `\input{outputs/paper_figures/M4.tex}`.
It needs amsmath, xcolor, booktabs, makecell, and colortbl. At the researcher's
request, M4 inherits manuscript fonts and its existing `tablehead`, `tablerow`,
and `qgray` colors instead of the shared plotting preamble. It keeps the original
`\small`, `\arraystretch{1.55}`, default column padding, and plain line breaks.
Slash spacing also matches the original. The left-hand column names raw pixels,
features, and best quality beside the four dataset columns. Each stacks its
three triplets. The original's pale header/body backgrounds remain.
Native small text and tabular* fit the manuscript's 6.75-inch line width without
whole-table scaling. The preview uses that text width plus 0.12-inch margins.
Its manuscript number is automatic; only the separate preview sets Table 2.
The generator compiles that exact fragment with pdflatex and records both
the source and PDF. It does not overwrite any source or derived metric CSVs.
Numbers are formatted directly from the exports: two DiT SSCD values display
as 98.5 and 91.1, rather than the old PDF's 98.6 and 91.2, because the stored
floats fall just below the respective one-decimal rounding boundaries.

`plotting/paper_style.py` owns DejaVu Sans text, STIX math, and bold italic
parameter labels: `\boldsymbol{x}`, `\boldsymbol{\epsilon}`, and
`\boldsymbol{v}`. Import its `TARGET_LABELS` and `TARGET_PAIR_LABELS` rather
than defining local labels. Templates retain their own sizes and layout.
Standalone LaTeX uses the matching `LATEX_PREAMBLE`, except M4 as specified above.

## Historical renderer details

The following notes preserve earlier F-number references and commands for
tracing existing work. Use the M/A names and workflow above for current drafts.
The old PDF links refer to the earlier output collection.

## Paired diagonal trajectories: F10

F10 updates the supplied historical figure using the current CelebA32 U-Net
export. Its columns follow that reference: eps/eps in blue, v/v in green,
and x/x in red. The upper row shows log Inception FID on a pale blue background.
The lower row shows raw-pixel f_mem at fixed rho=0.3 on a pale warm background.
The researcher explicitly selected raw pixels for this figure.

Four line styles identify N=512, 1024, 2048, and 4096. For this template,
larger N also uses a lighter target color, matching the reference. N=8192 in
the historical figure is absent from the current bundle. Each line is the
arithmetic mean of three seeds; bands show sample SD. The figure uses all
71 positive saved checkpoints without smoothing. Step zero remains in the
CSV tables but is omitted from the logarithmic time axis, tau=step/1,000,000.

```bash
# Rebuild F10 using its reusable DT template.
conda run -n memcompgen python plotting/figures.py F10

# Rebuild the figure and its per-seed and mean/SD numerical tables.
conda run -n memcompgen python plotting/plot_diagonal_trajectories.py
```

The second command reads only `plot_data/celeba32_unet_3x3/`. It joins metrics
by run_id, checks the 36 diagonal runs and matching checkpoint schedules,
and uses the common seed-summary function. Outputs in the U-Net figure folder
have basename `diagonal__fid__fmem_raw_pixels__rho_0p3__seed_mean_sd`:

- `.pdf` and `.svg`: current figure.
- `__per_seed.csv`: all 5,184 values, including run IDs, thresholds, tau,
  and whether the checkpoint is displayed.
- `__mean_sd.csv`: all 1,728 summaries, with seed counts, means, and sample SD.
- `.caption.txt`: metric, uncertainty, plotting, and source-data definitions.

Edit `COLUMN_TARGETS`, `SIZE_WHITE_MIX`, and `ROW_BACKGROUNDS` in the new
renderer for this template's order, shade progression, and backgrounds.
The base target colors and two-inch panel widths still come from the shared
style owner. Its canvas is about 7.11 by 3.65 inches, with two-inch-wide axes.

## Trajectories with a summary row: F11

F11 adds two wide comparison panels below F10's two trajectory rows: best FID
and raw-pixel f_mem at that best-FID checkpoint. The
new renderer calls F10's `draw_trajectories` function so both figures use the
same curves and styles. Summary panels retain x/x red, eps/eps blue, and v/v
green. All numerical inputs come from the same 36 runs under `plot_data/`.

Best FID is selected within each seed, with the earlier checkpoint breaking
exact ties. The paired f_mem comes from that same checkpoint. The bottom
panels show means and sample SD across the three seeds. Both best-FID summaries
match the existing best-FID tables exactly. Values are recomputed for raw
pixels at rho=0.3; calibrated SSCD values from the reference are not reused.

The onset panel and its threshold annotation have moved to the separate
F12 timescale figure. The remaining best-FID panels share the full bottom row.

```bash
# Rebuild only F11; retain all numerical CSVs.
conda run -n memcompgen python plotting/figures.py F11

# Rebuild the figure and its 36 per-seed / 12 mean-SD summary records.
conda run -n memcompgen python plotting/plot_diagonal_summary.py
```

Both commands were executed in local memcompgen. Artifacts in the U-Net
figure folder use basename
`diagonal__trajectories_and_summaries__raw_pixels__rho_0p3`: PDF, SVG,
caption text, `__per_seed.csv`, and `__mean_sd.csv`. Per-seed records retain
run IDs, selected checkpoints, onset cutoff/rule, censoring, and the number
of supporting saved checkpoints. Existing F10 tables retain all trajectory
values. The existing F11 tables retain their earlier onset fields unchanged;
F12 exports the combined timescale records. F11's canvas is 8.8 by 4.35
inches, a moderate reduction that preserves its earlier
9.9239-by-4.9-inch aspect ratio. Its trajectory axes are about 2.39 inches
wide; its summary axes are about 3.77 inches wide. Edit `FIGURE_WIDTH_INCHES`
and `FIGURE_ASPECT_RATIO` in its renderer to control the full canvas.
Tick labels, panel letters, legends, and footer remain eight points in the
export. Bold panel letters (a)-(h) sit above the upper-left corner of each panel.
Reduced y-tick density and adjusted spacing keep the smaller layout legible.
The FID and f_mem rows have equal height, each 40% taller than the bottom
summary row. The legend sits below the bottom panels' axis labels with a
small gap; the title follows below the legend.
The compact caption area below the panels groups the N line-style key and
the parameter color/marker key. Dataset, metric, and uncertainty form the
bottommost title line. The separate top header and legend are removed.
F11/F12 use bold serif math for parameter pairs. `TARGET_PAIR_LABELS` in
`plot_diagonal_summary.py` owns their shared LaTeX-style labels.
F10 remains the compact two-row version.

The [official AISTATS 2027 template](https://aistats.org/aistats2027/AISTATS2027PaperPack.zip)
sets the two-column width to 6.75 inches and places numbered captions below
figures. Use the PDF at full text width and let LaTeX assign its paper number;
F11 is only the stable file code. Scaling the current export to 6.75 inches
makes its eight-point labels about 6.1 points in the paper. The plot text and bold panel
letters are our readability convention, not numerical requirements in the
template. The separate `.caption.txt` retains the full metric definitions.

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/F11.pdf}
  \caption{Generation and Memorization on CelebA32. (a--c) FID trajectories.
    (d--f) Raw-pixel $f_{\mathrm{mem}}$ at $\rho=0.3$.
    (g) Best FID selected separately per seed. (h) Memorization at that checkpoint.
    Lines show three-seed means; bands and error bars show sample standard deviations.}
  \label{fig:celeba32-trajectories}
\end{figure*}
```

## Separate timescales and window: F12

F12 stacks three panels vertically for the same 36 CelebA32 U-Net diagonal runs:

- tau_gen: first saved step with FID <= 1.10 * the run's minimum saved FID.
- tau_mem: first saved step from which f_mem >= 0.05 at every remaining
  saved checkpoint through one million steps.
- delta_tau = tau_mem - tau_gen, computed within each seed before aggregation.

The plan originally applied the sustained 5% cutoff to calibrated SSCD at
alpha=0.01. F12 uses the selected raw-pixel metric at rho=0.3. The crossing
rule matches, but the resulting times can differ. The 5% cutoff is separate
from rho and has no calibrated false-positive interpretation here.
`QUALITY_TOLERANCE` in `plot_diagonal_timescales.py` owns the 10% tolerance;
`ONSET_CUTOFF` in `plot_diagonal_summary.py` owns the shared onset cutoff.
The new renderer reuses that module's per-seed onset calculation.

tau_gen measures attainment of near-best within-run quality. It need not equal
the best-FID checkpoint, and quality is not required to stay within the band.
Negative delta_tau means memorization onset precedes near-best quality. The
window panel retains negative values on a linear scale with a zero line.
The first two panels use log scales. All y axes use training steps/1,000,000.

Onset is unobserved in all three seeds for eps/eps at N=2048 and N=4096,
and v/v at N=4096. For a censored seed, tau_mem > T and delta_tau > T - tau_gen,
where T is its last evaluated step. F12 averages these bounds to obtain a
lower bound on the three-seed mean, shown by an open marker and upward arrow.
These bounds are explicitly separate from estimated means and have no SD.
Lines stop at missing estimates. Observed conditions show arithmetic means
and sample SD; gap SD is computed from the paired seed differences.

```bash
# Rebuild both figures without rewriting numerical CSVs.
conda run -n memcompgen python plotting/figures.py F11 F12

# Rebuild F12 and write its new per-seed and summary tables.
conda run -n memcompgen python plotting/plot_diagonal_timescales.py
```

Both commands were executed in local memcompgen. Outputs use basename
`diagonal__timescales__raw_pixels__rho_0p3` in the U-Net figure folder:
PDF, SVG, caption text, `__per_seed.csv` (36 rows), and `__mean_sd.csv`
(12 rows). Records include run IDs, the FID threshold, best checkpoint,
tau_gen, tau_mem, paired gaps, and separate censoring/lower-bound fields.
The CSV values use unscaled training steps. The canvas is about 3.25 by
4.8 inches with three 2.55-inch-wide panels, one shared training-size axis,
and one shared units label. The compact title remains at the bottom.
`MARKER_SIZE=5` sets the observed markers and legend markers in points;
censored markers are half a point larger. The first panel uses a tighter
logarithmic range while retaining every displayed mean and SD bar. The
on-figure caption is shortened; the accompanying caption file retains the
full definitions and uncertainty rules.

## DiT timescales: F13

F13 applies F12's definitions and stacked TS template to the 12 diagonal
CelebA32 Small-DiT runs in `plot_data/celeba32_dit_3x3/`. Each of the four
training sizes has seed 0 only. The figure explicitly labels this coverage,
draws no SD bars, and leaves SD fields blank in the summary table. All 12
runs have an observed sustained 5% raw-pixel onset; every signed gap is negative.

The paired-metric loader now accepts the experiment name and checks that
FID and raw-pixel f_mem have matching checkpoint grids. The common per-seed
selector computes best FID and sustained onset for either architecture.
The TS renderer handles uncertainty according to the actual seed count.
F13 uses tighter y limits fitted to its observed values, with the same log
time / linear gap scales as F12. These limits differ between the two figures.
The canvas, larger markers, canonical colors, and bold math labels match F12.
F12 and F13 are a paired figure family: apply visual style changes to both.
Both use faint dotted vertical gridlines at each training-size tick in all
three panels. Seed uncertainty, censoring, and fitted y limits remain tied
to each figure's data. Use `python plotting/figures.py --type TS` to rebuild
the pair. The common renderer writes a manuscript-ready `.caption.tex` beside
each PDF, as well as the detailed `.caption.txt` with input provenance.

```bash
# Rebuild only the DiT counterpart, preserving its numerical tables.
conda run -n memcompgen python plotting/figures.py F13

# Rebuild it and write its 12 per-seed / 12 condition-summary records.
conda run -n memcompgen python plotting/plot_diagonal_timescales.py --experiment celeba32_dit_3x3
```

Both commands were executed in local memcompgen. PDF/SVG, the full caption,
`__per_seed.csv`, and `__mean_sd.csv` use the same descriptive basename as
F12, under `outputs/figures/celeba32_dit_3x3/`. In the single-seed summary,
mean fields equal the seed-zero values and SD is undefined. Use
`python plotting/figures.py --type TS` to rebuild both timescale figures.
The existing U-Net records remain identical. Rebuilding the paired figures
updates their graphics and captions while preserving the numerical tables.

### Reading F12 and F13 in the current export

The quality time is relative to each run's own minimum saved FID. It does not
mark the first useful generated samples or compare runs at one absolute FID.
The onset time uses the sustained 5% raw-pixel criterion at rho=0.3. The signed
gap subtracts these two saved-checkpoint times. A negative gap means onset
precedes near-best quality; a positive gap does not establish that quality
stays within its band throughout the interval.

For U-Net, 25 of the 27 observed seed gaps are negative. The other two are
x/x at N=4096; its third seed is negative, so its positive mean is not a
consistent seed-level ordering. All three eps/eps seeds at N=2048 and 4096,
and all three v/v seeds at N=4096, remain below 5% at every saved checkpoint
through one million steps. Their arrows show lower bounds, not onsets at one
million. The sharp drop in v/v quality time from N=2048 to 4096 also uses a
different target: mean best FID changes from 7.00 to 10.84. This drop alone
does not demonstrate faster attainment of the same sample quality.

For DiT, all 12 seed-zero gaps are negative. At N=4096:

| Pair | tau_mem (steps) | tau_gen (steps) | f_mem at tau_gen |
| --- | ---: | ---: | ---: |
| x/x | 95,000 | 695,000 | 91.61% |
| eps/eps | 210,000 | 900,000 | 44.37% |
| v/v | 75,000 | 695,000 | 79.08% |

Here eps/eps delays onset most, but all three conditions are well beyond the
5% threshold by near-best FID. Quality improves after memorization begins;
these records do not show that memorization causes that improvement. Larger
training sets delay onset in all three DiT conditions, while near-best FID
can move much later still. This explains large negative gaps without implying
that the onset itself moved earlier.

The seed-zero U-Net/DiT exports share data-bundle hashes, image format, learning
rate, effective batch size, training horizon, and evaluation sample/checkpoint
counts. Microbatch sizes differ (256 versus 64). These fields do not establish
matched model capacity or complete training/evaluation configuration equality.
Only one DiT seed is available. Capacity, inductive bias, and optimization
dynamics are possible explanations, not isolated causes in these figures.
The two figures also use different y limits, so compare values, not curve height.
These observations were checked against `plot_data/` on 2026-10-06.

## Panel dimensions

The default subplot axes box is 2 inches wide. Labels, gaps, and legends add to the
total figure width. Heights vary with the layout: about 2.36 inches for the
three diagonal panels, 1.67 for the quality 3x3 grids, 1.76 for the dual-axis
3x3 grid, 2.31 for the weighting comparison, and 2.01 for the Python table.
F10's two-row trajectory layout uses panels about 1.05 inches high.
F11 uses 2.39-inch trajectory panels and two 3.77-inch summary panels on an
8.8-by-4.35-inch canvas. F12 stacks three 2.55-inch panels on a 3.25-by-4.8-inch canvas.
F13 uses the same dimensions as F12.

`PANEL_WIDTH_INCHES` and `set_panel_width` in
`plotting/celeba32_3x3/plot_f_mem_grid.py` set the common width. Most renderers
calls `set_panel_width(figure, axes[0, 0])` after choosing its layout. The
helper adjusts the canvas width while retaining the chosen height.
F12/F13 pass their local widths through the helper's `width` argument.
F11 sets its canvas width and aspect ratio directly.
The same module's `matplotlib.rcParams` sets common font sizes, including
8-point tick labels. Individual templates override title or table sizes where
their content needs a different setting.

The best-FID table places the paired FID in parentheses on the next line
within each entry. This keeps all digits readable in a 2-inch cell. Its LaTeX
alternative uses 2-inch-wide, 2.4-inch-high minipages with the same two-line
entries. Both renderers take their width from `PANEL_WIDTH_INCHES`.
These dimensions describe the exported PDFs; scaling a PDF in a manuscript
also scales its panels and text.

## Shared visual convention

The canonical prediction-target colors are:

| Diagonal condition | Prediction target | Color | Hex |
| --- | --- | --- | --- |
| eps/eps | eps | Blue | `#1F77B4` |
| x/x | x | Red | `#D62728` |
| v/v | v | Green | `#2CA02C` |

Off-diagonal cells keep their prediction row's color family. With one training
size, use the canonical color. With two sizes, use a lighter shade for the
smaller size and the canonical, darker shade for the larger size. For the
current N=512/2048 comparisons, circles denote 512 and triangles denote 2048.
These means use solid lines, except where line style already identifies a
metric or a velocity weight. Lighter shades mix 35% white into the canonical
color: x `#E47373`, eps `#6DA7CE`, and v `#76C176`.

Four-size single-metric trajectories distinguish N=512/1024/2048/4096
with solid/dashed/dash-dot/dotted lines. The dual-axis FID/f_mem grid reserves
dotted lines for FID and solid lines for f_mem, so its training sizes use
circle/square/triangle/diamond markers instead.

In the CelebA32 weighting comparison, use distinct colors for the three
velocity conditions: unit weighting is solid green, x_equivalent is dashed
purple (`#9467BD`), and eps_equivalent is dotted orange (`#FF7F0E`). The
lighter/darker size rule and markers apply to all five conditions. Their
labels state both x-equivalent/epsilon-equivalent and the weighting formulas.
The best-FID table colors values by prediction
target and keeps the stacked training-size labels neutral.

`TARGET_COLORS`, `target_color`, `size_color`, `N_TRAIN_LINESTYLES`, and `N_TRAIN_MARKERS` in
`plotting/celeba32_3x3/plot_f_mem_grid.py` own these shared definitions.
The other portable plotting commands import them. Apply this convention
to future figures. `CONDITIONS` in `compare_weighted_velocity.py` adds the
two weighting-variant colors without changing the canonical diagonal colors.

## Pull and environment

From the existing local clone:

```bash
git fetch origin
git switch codex/portable-plot-data
git pull --ff-only origin codex/portable-plot-data
```

The current CelebA32 plotting command needs Python and Matplotlib. It does not
need PyTorch, CUDA, the datasets, or any project checkpoint:

```bash
python -m pip install matplotlib
python plotting/celeba32_3x3/plot_f_mem_grid.py
```

Expected outputs:

```text
outputs/figures/celeba32_unet_prediction_loss_3x3/
  all_n_train__fid__fmem_raw_pixels__rho_0p3__seed_mean_sd.pdf
  all_n_train__fid__fmem_raw_pixels__rho_0p3__seed_mean_sd.csv
```

The figure is a 3x3 prediction-target by loss-target grid. Every panel overlays
`N_train={512,1024,2048,4096}`. Lines are three-seed means. Bands are plus or
minus one sample standard deviation. Dotted curves show Inception FID on the
left axis. Solid curves show raw-pixel `f_mem` at fixed `rho=0.3` on the right
axis. Both metrics use the prediction-target color and training-size markers,
with all 72 saved
checkpoints. Each metric has a common linear scale across the nine panels.
The plotted table identifies each metric and retains its individual seed values,
mean, and sample standard deviation. FID rows have no `rho` threshold.

## Memorization at best FID

From the repository root:

```bash
python plotting/celeba32_3x3/summarize_f_mem_at_best_fid.py
```

For each training size, prediction/loss pair, and seed, select the lowest
Inception FID among all 72 saved checkpoints. Use the earlier checkpoint for
an exact tie. Read raw-pixel `f_mem` at `rho=0.3` from that same checkpoint,
then compute the three-seed mean and sample standard deviation (`ddof=1`).
Selected steps and FID levels can differ across seeds and conditions.

The command prints each cell as `f_mem mean ± SD (best FID mean ± SD)`.
Both metrics use the same selected checkpoint for each seed. It writes two
tables under the same figure directory:

- `fmem_at_best_fid__rho_0p3__per_seed.csv`: 108 run IDs, selected steps, FIDs,
  and paired `f_mem` values, with condition and seed labels.
- `fmem_at_best_fid__rho_0p3__mean_sd.csv`: 36 condition summaries with the
  seed count, mean, and sample standard deviation for both metrics.

Both tables derive only from `plot_data/`. No evaluation is rerun.

Render the same summary as a 3x3 grid:

```bash
python plotting/celeba32_3x3/plot_f_mem_at_best_fid.py
```

Rows are prediction targets. Columns are loss targets. Each panel stacks
N=512,1024,2048,4096 with neutral size labels and prediction-colored values.
Each entry shows f_mem on the first line and the paired FID in parentheses
on the next line. The renderer calls
the same summary function and reads numerical inputs only from `plot_data/`.
It writes vector PDF and SVG files with basename
`fmem_at_best_fid__rho_0p3__grid` in the same figure directory. The standalone
LaTeX source uses basename `fmem_at_best_fid__rho_0p3__grid__latex` so its
compiled PDF does not overwrite the Python-rendered PDF.

If LaTeX is installed, compile the table with:

```bash
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=outputs/figures/celeba32_unet_prediction_loss_3x3 \
  outputs/figures/celeba32_unet_prediction_loss_3x3/fmem_at_best_fid__rho_0p3__grid__latex.tex
```

This creates the alternative PDF table beside the Python-rendered figures.

## Diagonal targets and weighted velocity controls

Compare the three diagonal targets with the two equivalent velocity weights:

```bash
python plotting/celeba32_3x3/compare_weighted_velocity.py
```

This combines records from `celeba32_unet_3x3` and `celeba32_vv_weighting`
at their common training sizes, N=512 and N=2048. Each of the five conditions
has seeds 0, 1, and 2:

| Condition | Network output | Loss |
| --- | --- | --- |
| x/x | x | Unweighted x MSE |
| eps/eps | eps | Unweighted eps MSE |
| v/v | v | Unweighted velocity MSE |
| v/v, x_equivalent | v | Velocity MSE multiplied by `(1-t)^2` |
| v/v, eps_equivalent | v | Velocity MSE multiplied by `t^2` |

For the path `z_t = t*x + (1-t)*eps`, a velocity prediction implies
`x_hat = z_t + (1-t)*v_hat` and `eps_hat = z_t - t*v_hat`. The weighted
velocity errors therefore equal the x and eps errors of these implied
predictions. These are also the objectives of the matrix's v/x and v/eps
cells. This algebra does not make a directly trained x or eps network
identical to a velocity network. The comparison asks whether matching the
loss weighting also matches the observed trajectories. Weights remain
unnormalized, preserving the objective equality.

The exported records match on saved data-bundle hashes for each N and seed,
architecture, learning rate, batch size, training length, and evaluation
coverage. The declared runner configurations also match on data, network,
optimizer, time sampling, sampler, precision, and evaluation settings.
All six unweighted velocity controls reproduce the corresponding matrix
v/v FID and f_mem trajectories exactly. The script retains only the matrix
v/v controls; the repeated runs do not provide six independent seeds.
The portable records do not contain the complete frozen evaluation manifests,
so this check does not independently audit upstream feature caches or
checkpoint loading.

Outputs are saved under `outputs/figures/celeba32_weighting_comparison/`:

- `diagonal_vs_weighted_velocity.pdf` and `.svg`: FID above f_mem, with
  one column per training size. Curves show three-seed means and sample-SD
  bands. Dashed purple and dotted orange curves identify the weighted velocity controls.
- `trajectories.csv`: 2,160 paired checkpoint records from 30 runs, including
  run IDs, conditions, and seeds.
- `best_fid_per_seed.csv`: 30 per-seed minimum-FID checkpoints and paired
  raw-pixel f_mem values at rho=0.3; exact ties select the earlier step.
- `best_fid_mean_sd.csv`: ten condition summaries for both metrics.

All numerical inputs come from `plot_data/`. The script verifies matching
exported settings and the repeated v/v trajectories before combining them.
Different conditions can have different best FID levels; their memorization
at best FID is not a comparison at one fixed generation quality.

## Quality trajectories by training size: DiT, MNIST, and CelebA64

Run all four figure families from the repository root:

```bash
conda run -n memcompgen python plotting/plot_fid_by_training_size.py
```

| Experiment | Panels | Training sizes | Seeds | Checkpoints per run | Quality metric |
| --- | --- | --- | --- | ---: | --- |
| `celeba32_dit_3x3` | 3x3 prediction/loss matrix | 512, 1024, 2048, 4096 | 0 only | 72 | Inception FID |
| `mnist32_diagonal` | x/x, eps/eps, v/v | 512, 2048 | 0, 1, 2 | 20 | JointAE FD |
| `mnist64_diagonal` | x/x, eps/eps, v/v | 512, 2048 | 0, 1, 2 | 20 | JointAE FD |
| `celeba64_rgb_diagonal` | x/x, eps/eps, v/v | 512, 2048 | 0, 1, 2 | 75 | Inception FID |

Each panel overlays training sizes within its prediction-target color family.
Two-size comparisons use lighter circles and darker triangles; four-size
comparisons use distinct line styles. Quality uses a
logarithmic axis, shared across panels within each figure. The step axis is
linear near zero and logarithmic above 1,000. Lines connect saved checkpoints;
the script does not add checkpoints or smooth values. All evaluations use
10,000 generated samples and cover step zero through one million updates.

Three-seed figures show arithmetic means, sample-SD bands, and faint individual
seed curves. Only the displayed lower band is clipped at the positive log-axis
floor when needed; the exported SD is unchanged. DiT has seed 0 only in this
bundle: no uncertainty band is drawn and its CSV `std` fields are blank.

MNIST uses `mnist_fd_jointae32_v1`, the primary quality metric described in
`docs/image_evaluation.md`. It is not Inception FID and its values should not
be compared numerically with CelebA FID. CelebA64 uses the legacy dataset key
`celeba32_grey`; the run fields confirm 64x64 images and three RGB channels.

The script reads `runs.csv` and `evaluation_metrics.csv` only from `plot_data/`.
It checks model, image format, sizes, seeds, targets, sample counts, and complete
checkpoint coverage before plotting. It reuses target colors and size styles from the
existing CelebA32 plotting module. Edit the `EXPERIMENTS` table in the new
script to review the explicit coverage and metric choices.

For each experiment, outputs under `outputs/figures/<experiment>/` have
basename `all_n_train__<metric>__log_scale`:

- `.pdf` and `.svg`: vector versions of the figure.
- `__per_seed.csv`: run IDs, conditions, metric names, steps, and source values.
- `__mean_sd.csv`: seed counts, means, and sample SDs for every plotted point.

## Portable data layout

```text
plot_data/
  INDEX.json
  README.md
  celeba32_unet_3x3/
  celeba32_dit_3x3/
  celeba32_vv_weighting/
  celeba64_rgb_diagonal/
  mnist32_diagonal/
  mnist64_diagonal/
```

Every experiment directory has the same files:

- `runs.csv`: one row per accepted run and all experimental conditions.
- `training_metrics.csv`: training loss and saved monitoring trajectories.
- `evaluation_metrics.csv`: accepted checkpoint-level scientific metrics.
- `evaluation_diagnostics.csv`: raw-range, runtime, and threshold diagnostics.
- `manifest.json`: row counts, byte sizes, and SHA-256 hashes.
- `README.md`: the local schema reminder.

Join any metric table to `runs.csv` with `run_id`. Do not infer experimental
conditions by parsing filenames when the corresponding column exists in
`runs.csv`.

The `source_run_path` and `evaluation_result_path` columns are provenance
records from the cluster. They are not expected to exist on the laptop and are
not required for plotting.

## Current coverage

The committed bundle contains 216 accepted evaluations:

| Experiment | Accepted runs | Evaluation rows |
| --- | ---: | ---: |
| CelebA32 U-Net 3x3 | 108 | 186,624 |
| CelebA32 Small-DiT 3x3 | 36 | 62,208 |
| CelebA32 v/v weighting | 18 | 31,104 |
| CelebA64 RGB diagonal | 18 | 32,400 |
| MNIST32 diagonal | 18 | 5,760 |
| MNIST64 diagonal | 18 | 5,760 |

The final 18 seed-1 Small-DiT evaluations were still running when this bundle
was exported. Therefore `celeba32_dit_3x3` currently contains the 36 accepted
seed-0 runs only. Do not treat the missing seed-1 cells as zeroes and do not
silently mix incomplete seed coverage into aggregate plots.

## Rules for local Codex

1. Read `plot_data/README.md`, `plot_data/INDEX.json`, and the relevant
   experiment's `runs.csv` before writing a plot.
2. Use only `plot_data/` for numerical inputs. Do not require
   `outputs/evals/`, checkpoints, tensors, models, or cluster paths.
3. Preserve individual seeds until the requested aggregation is explicit.
   When showing a three-seed mean, use sample standard deviation.
4. Keep prediction space, loss space, training size, model, dataset, metric
   space, threshold kind, `alpha`, and `rho` explicit in filtering code.
5. Write vector PDFs and their compact plotted CSV tables under
   `outputs/figures/<experiment>/`.
6. Do not edit exported CSVs manually. The exporter writes deterministic rows,
   so regenerated scientific-data changes remain reviewable in normal Git
   diffs.
7. Keep one plotting entrypoint focused on one figure family. A style edit
   should take seconds and must not trigger evaluation work.

## Exporter scope

`plotting/export_plot_data.py` is the cluster-side regeneration command. Its
input roots intentionally point at retained accepted evaluation outputs. A
laptop without those outputs should consume the committed `plot_data/`
bundle, not rerun the exporter.

The exporter includes only accepted, publication-ready runs. It exports numeric
training history, monitoring records, evaluation metrics, and scalar
diagnostics. It excludes checkpoints and all large reproducible tensors.

## Suggested first prompt for local Codex

> Read `docs/local_plotting_handoff.md` and `plot_data/README.md`. Work only
> from the committed `plot_data/` CSVs. First reproduce the existing CelebA32
> U-Net 3x3 f_mem figure. Then help me iterate on one requested figure at a time.
> Preserve seed-level values, show exactly which metric and threshold are used,
> save figures under `outputs/figures/`, and never depend on cluster outputs
> or checkpoints.

## Validation performed before push

- Exported all six experiment directories successfully.
- Verified 216 accepted run rows and the recorded per-experiment row counts.
- Verified every exported file is below 40 MB.
- Reran the exporter and obtained byte-identical hashes for all 38 files.
- Passed Ruff and Python compilation for both plotting scripts.
- Regenerated the 3x3 figure from only `plot_data/` in 1.84 seconds.
- Passed Git whitespace checks.
