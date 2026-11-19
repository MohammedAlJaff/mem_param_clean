# Repository Map

STANDING COLLABORATION RULE
Use plain English, short sentences, active verbs, precise terms, and a natural
voice. Help the researcher understand and own the work. For substantial changes,
show the relevant files, how the parts connect, and a small usage example with
expected outputs. Use real code or clearly labelled pseudocode. State what was
tested. Link to the implementation, settings, and results. Keep reusable examples
in the relevant docs and the layout in docs/repo_map.md. Involve the researcher
in meaningful choices without routine approval requests. Apply this rule to
future notes and documentation. Full guidance: root AGENTS.md, Collaboration
and Code Understanding.

Current skeleton:

```text
mem_param_src/     reusable first-party package
external/          baseline and third-party code, if retained
notebooks/         reviewed notebooks, probes, and paper analysis
jobs/berzelius/    Slurm launchers for Berzelius, if retained
plot_data/        committed scalar records for local analysis
plotting/         local figures and summaries from the portable records
paper_snapshots/   fixed paper-file archives with file lists and hashes
docs/              migration notes and project intent
TODO.txt           single operational backlog of unfinished work
RUN_JOURNAL.md    submitted compute runs and chronological status checks
```

This tree is intentionally small. Add a new folder only when a reviewed file
class needs it.

Use `TODO.txt` for unfinished work. Use `MIGRATION_JOURNAL.md` for decisions,
checks, and completed work. Use `RUN_JOURNAL.md` for every clean-repository
compute submission and its append-only status history. These files must not
become competing backlogs.
Use `docs/config_reference.md` for every active configuration field, its
meaning, current default, implementation status, and legacy runner guidance.

The user-requested `CRITICAL_EPS_EPS_SAMPLING_AUDIT_2026-09-09.txt` preserves
the current sampling audit and Berzelius handoff, including artifact hashes
and an unexecuted fixed-checkpoint GPU command. It records an open diagnosis;
`TODO.txt` remains the backlog and `MIGRATION_JOURNAL.md` the completed-work log.

Known file classes from the old repo:

- model and sampler code
- losses and metrics
- synthetic data loaders
- image data loaders
- trainers
- checkpoint replay and evaluation code
- artifact, logging, and tracking code
- visualization and figure-generation code
- Berzelius job files
- experiment sweep definitions
- tests
- notebooks
- baseline and third-party code
- generated outputs, which stay untracked

Probable destinations:

- Library code goes under `mem_param_src/`.
- Dataset definitions and loader logic live under `mem_param_src/datasets/`.
- Generic metric math stays directly under `mem_param_src/metrics/`.
- Image preprocessing and metric adapters live under
  `mem_param_src/metrics/image/`.
- Dataset-specific synthetic metrics live under
  `mem_param_src/metrics/synthetic/`.
- Berzelius launch files go under `jobs/berzelius/`.
- `RUN_JOURNAL.md` records every submitted job and each later status check.
- Reviewed notebooks go under `notebooks/`.
- `notebooks/theory/` holds the supplied linear/RFN notebook, its existing
  figures, and proposed main/appendix LaTeX snippets. Start with
  `theory_audit_2026-10-07.txt` for verified claims and outstanding figure work.
  The notebook's M6 plotting cell combines the selected random-initialization
  panels from retained simulation arrays, without additional training.
- Baselines and third-party code go under `external/` only if retained.
- Generated outputs do not enter the clean repo by default.
- `paper_snapshots/2026-11-19/` is the approved exception. It preserves the
  local paper workspace for the rebuttal. Its archive and manifest retain
  figures, captions, Overleaf files, and prior build records. New results
  still go under `outputs/`; do not replace this snapshot.

Boundary rules:

- `docs/paper_figure_plan.txt` records the retained manuscript Figure 1-4 and
  Table 1-2 layouts, raw-pixel rho=0.3/sample-SD requirements, planned theory
  addition, and appendix scope. Its reference PDF hash fixes the visual baseline.
  It specifies paper contents separately from catalog IDs and build approval.

- Portable agent handoffs go under `outputs/handoffs/`. The
  `mem_param_plotting_handoff_2026-10-05` folder and matching ZIP contain
  copied scalar inputs, derived tables, current figures, six local plotting
  scripts, setup instructions, and checksums. Its README is the standalone
  entrypoint; it requires no clone, checkpoints, or cluster paths.

- `outputs/paper_figures/` is the fresh paper revision workspace. Its
  `README.txt` maps the six retained paper items to generators; `INDEX.md`
  lists only generated drafts. Build one selected item or a small selected
  group at a time. The earlier `outputs/figures/` collection stays intact.
  `results_figures.tex` and `appendix_figures.tex` supply Overleaf float
  blocks, captions, and labels using `figures/` paths. The accompanying
  `overleaf_figures.zip` includes those fragments and their PDF assets.
  `results_figures_reconciled.tex` is a separate alternative that preserves
  the supplied manuscript's float order and labels, with M6 appended.

- `plotting/figure_catalog.json` registers the paper's M1-M6 and A1-A5 names
  where generators exist, with descriptive suffixes for variants. It records
  scientific settings and exact-build review status. `plotting/figures.py`
  generates named outputs such as M2.pdf and M5.pdf. Older IDs work only as
  command aliases; current files and the index use M/A names. Build receipts
  and `outputs/paper_figures/versions/` preserve inputs, code, and artifacts.
  `plotting/AGENTS.md` gives the shared workflow. The catalog does not rewrite
  numerical CSV tables. See `docs/paper_figure_plan.txt` for planned items.

- `plotting/paper_style.py` owns DejaVu Sans text, STIX mathematics, and
  shared bold mathematical x/epsilon/v labels, including diagonal pairs.
  The common plotting module applies its Matplotlib settings; the standalone
  LaTeX renderer uses its matching preamble. Templates retain their own sizes.

- `plotting/celeba32_3x3/plot_best_fid_matrices.py` owns M3's two-matrix
  reference layout. It uses the established per-seed best-FID summary to show
  memorization percentages and paired FID at N=512 and 2048. M3 uses raw pixels;
  M3-SSCD uses calibrated SSCD alpha=0.01; both use U-Net and sample SD.
  M3-DIT uses raw pixels for Small-DiT with seed 0 and no uncertainty estimate.
  See `docs/local_plotting_handoff.md` for the ID/type map and commands.

- `plotting/plot_robustness_table.py` owns M4 / manuscript Table 2. It reuses
  the accepted quality loader, selects each seed's best quality at N=2048,
  and joins raw-pixel and calibrated feature memorization at the same step.
  The catalog builds a manuscript LaTeX fragment, its PDF preview, and captions
  under `outputs/paper_figures/`; the caption explains the dataset-specific
  encoders. The compact table displays means only and adds a metric-description
  column; the earlier SD version remains archived. See the plotting handoff
  for the one-command build and manuscript inclusion example.

- `plotting/plot_diagonal_trajectories.py` owns the M2_TRAJECTORIES variant.
  It reuses the quality-data loader and seed-summary function, joins raw-pixel
  f_mem at rho=0.3 to the same diagonal runs, and plots FID above f_mem.
  The supplied reference sets eps/v/x column order, target-color shade ramps,
  cool/warm backgrounds, and normalized log training time. PDF/SVG, a caption,
  per-seed values, and mean/SD tables go into the CelebA32 U-Net figure folder.

- `plotting/plot_diagonal_summary.py` owns M2 and its 3+3+2 template. It
  reuses F10's trajectory rows and adds two wide panels: best FID and raw-pixel
  f_mem at each seed's best FID. Its shared per-seed summary still computes
  sustained onset for F12 and preserves the existing numerical table schema.
  Its local FIGURE_WIDTH_INCHES is 8.8; FIGURE_ASPECT_RATIO preserves the
  previous wide canvas shape. The bottom panels span half the row each.
  The caption area groups the N line-style and parameter color/marker keys.
  PDF/SVG, a caption, and per-seed/summary tables use a separate output stem.

- `plotting/plot_diagonal_timescales.py` owns A3_UNET and A3_DIT (current-data appendix timescales),
  and their shared TS 1x3 template.
  Apply visual edits to both figures. It writes shared-style PDFs/SVGs,
  detailed text captions, and manuscript-ready `.caption.tex` files.
  It reuses the portable loader and per-seed onset calculation, then computes
  tau_gen at 10% of best FID and the paired gap tau_mem - tau_gen. Negative
  gaps remain visible. Censored groups show lower bounds on seed means with
  open arrows and no SD; observed groups show mean and sample SD. Separate
  per-seed/summary CSVs retain raw steps and explicit bound fields. The panels
  run horizontally on a 6.75-by-1.95-inch canvas, with linear time axes and
  2.8-point markers. Definitions and color keys appear in the caption.
  A3_DIT loads the 12 DiT diagonal runs through the shared paired-metric loader.
  Its single-seed values have no SD bars; its gap limits fit the observed range.
  Newly computed numerical inputs come only from `plot_data/`.

- `plotting/plot_workshop_timescales.py` owns M5: an explicitly requested
  reproduction of workshop Figure 4. It copies the vector artwork preserved
  under `plotting/references/` and writes captions explaining rho=0.5, first
  1% onset, SE, and capped unobserved onsets. The catalog archives and hashes
  the artwork and its provenance note; it does not reconstruct old values.

- `plotting/plot_memorization_sensitivity.py` owns A2, A5, and A5_TRAJECTORIES. It reuses the
  accepted quality loader and pairs seven readouts with each diagonal run's
  best-FID checkpoint. A2 compares calibrated pixels/SSCD/LPIPS/DINOv2 at
  alpha=0.01; A5 compares fixed pixel rho=0.3/0.5/0.7. Both show four N panels
  with mean/sample SD. Run `python plotting/figures.py A2 A5` for PDF/SVG,
  captions, and build receipts. The exports support aggregate-rate comparisons;
  per-image overlap and the onset-threshold sensitivity remain separate work.
  `load_trajectories` reuses the fixed-rho loader for all 72 checkpoints of
  the 36 U-Net diagonal runs. `render_trajectories` makes A5_TRAJECTORIES:
  a target-by-rho 3x3 grid on log axes, with mean/SD, shared N styles and
  canonical target colors. Its numerical inputs and output folder are shared
  with A5; it writes no numerical exports. Zero means break plotted lines
  without removing zero-valued seeds from the summaries.

- Portable CelebA32 U-Net analysis: `plotting/celeba32_3x3/plot_f_mem_grid.py`
  reads only `plot_data/` and plots FID and raw-pixel f_mem trajectories.
  Its `render` entrypoint also owns the current A1 and A1_DIT joint-axis
  appendix grids. The same loader validates all nine cells and 72 checkpoints;
  the renderer overlays four sizes and pairs dotted FID with solid F_mem.
  `python plotting/figures.py A1 A1_DIT` writes PDF/SVG and text/LaTeX captions
  under `outputs/paper_figures/`, with SD only for the three-seed U-Net data.
  It also owns the shared target palette (eps blue, x red, v green),
  lighter/darker shades for two sizes, training-size line styles, and
  training-size markers for portable figures. Its PANEL_WIDTH_INCHES and
  set_panel_width helper default to two-inch axes; F12/F13 pass wider settings.
  F11 sets its full canvas size directly, retaining its original aspect ratio.
  `summarize_f_mem_at_best_fid.py` reuses its record selection, chooses each
  seed's lowest-FID checkpoint, and summarizes the paired f_mem values.
  `plot_f_mem_at_best_fid.py` renders that summary as a 3x3 table with the
  training sizes stacked inside each cell, plus a standalone LaTeX version.
  `compare_weighted_velocity.py` supplies A4: the F07 comparison of diagonal
  targets with two explicit velocity weights, using one v/v baseline per seed.
  A4_DIAGONAL retains its legacy aliases. The optional matched mode compares
  the weights with v/x and v/eps; its previous A4 drafts remain archived.
  Both modes use N=512 and N=2048. Catalog exports and captions go to
  `outputs/paper_figures/`.
  Vector PDFs and scratch tables go under `outputs/figures/`. The local plotting
  handoff documents the commands and checkpoint selection rule.

- `plotting/plot_fid_by_training_size.py` plots the DiT 3x3 matrix and the
  MNIST32, MNIST64, and CelebA64 diagonal quality trajectories by training size.
  It reads only `plot_data/`, uses JointAE FD for MNIST and Inception FID for
  CelebA, and labels DiT's single-seed coverage explicitly. Vector PDFs and seed
  tables go under `outputs/figures/<experiment>/`.

- Network backbones: `mem_param_src/networks/` owns the shared `forward(z, t)`
  implementations. `model_setup.py` resolves config layouts and constructs the
  selected MLP, UNet, or DiT. `networks/dit.py` adapts the official SiT Small
  backbone as an unconditional raw-prediction model. Its retained license is
  `networks/LICENSE_SiT.txt`. GDFM continues to own prediction-space and loss
  conversions. `notebooks/sanity/dit_backbone_sanity.ipynb` demonstrates only
  factory construction, shape inspection, loss, and existing sampling calls.

- Berzelius smoke jobs: `jobs/berzelius/eps_recovery_smoke/submit.sh` submits
  rows from `variants.tsv` through `run_array.sbatch`. The shared experiment
  config is `configs/runners/celeba32_eps_recovery_smoke.py`. Scripts call the
  existing image runner; they do not own model, loss, or sampler logic.
  `inspect_results.py` reads saved raw tensors and hashes without model calls.

- One-million-step checkpoint evaluation:
  `jobs/berzelius/eps_recovery_1m/eval/submit.sh` launches six epsilon-only
  tasks at the shared 100k, 390k, and 695k checkpoints. Each source uses its
  as-launched sampler. `prepare_fid_reference.py` first persists one canonical
  10,000-image CelebA validation tensor and verifies its saved 1,024-image
  training-artifact prefix. `evaluate_checkpoints.py` writes append-only metric
  rows under `outputs/evals/`; `combine_results.py` makes one comparison CSV
  and plot. The evaluator reads source checkpoints and the shared frozen FID
  reference without modifying source runs. `submit_references.sh` adds the saved
  `v/v` and `x/x` references to the same three-point 10k/10k comparison.
  See `jobs/berzelius/eps_recovery_1m/eval/README.md` for the data flow.

- Feature-space memorization validation:
  `jobs/berzelius/eps_recovery_1m/feature_f_mem/` stages and verifies encoder
  assets, builds a frozen source manifest, prepares common validation features,
  replays three checkpoints per trajectory, and combines raw-first results.
  Its public encoder adapters live in `metrics/image/features.py`; all neighbor
  math remains in `metrics/nearest_neighbors.py`. Primary artifact paths are
  `monitor/random_projected_pixels`, `eval/full_pixels`, and
  `eval/features/{sscd,lpips,dinov2}`. Generated tensors, feature matrices,
  distances, ratios, and two neighbor identities remain under the untracked
  evaluation output. `feature_f_mem/README.md` documents preprocessing,
  resolutions, threshold calibration, artifact counts, and launch order.
  `notebooks/sanity/feature_f_mem_encoder_sanity.ipynb` is a visual small-scale
  study on saved CelebA tensors. It shows A/B galleries, held-out threshold
  calibration, interpolation/noise heatmaps, known-copy recovery, nearest-A
  examples, a final audit table, and LPIPS equivalence.

- Matched multi-seed training:
  `jobs/berzelius/eps_recovery_multiseed/` launches seeds 1 and 2 for matched
  `eps/eps`, `v/v`, and `x/x` training. It reuses the one-million-update config
  and changes only seed and parameterization. Its six-task array is deliberately
  unthrottled. The researcher submits it after the seed-0 feature runtime check;
  no feature encoder runs inside training.

- Recoverable long training: `runners/recovery.py` owns the stateful shuffled
  batch sampler, signal-to-boundary stop request, advisory run lock, complete
  RNG capture, and two-slot atomic recovery rotation. `runners/training.py`
  saves model, optimizer, sampler, history, elapsed time, and W&B identity at
  recoverable optimizer boundaries. `runners/image_train.py --resume-run-dir`
  reloads the saved config and rejects scientific overrides. Recoverable CUDA
  studies enable `runtime.deterministic`; the base default stays off.

- Prediction/loss and data-format gates:
  `jobs/berzelius/celeba32_prediction_loss_3x3/` launches the nine CelebA32
  gray prediction/loss cells on thin A100s. `jobs/berzelius/eps_eps_data_image_format/`
  launches seven EPS/EPS dataset, resolution, color, and capacity rows on fat
  A100s. Both are unthrottled, target one million updates, stop first at 200k,
  freeze their source inside the batch directory, and resume only through a
  researcher-selected array. Their READMEs define the evaluation and audited
  continuation protocol.

- Small-DiT diagonal pilot:
  `jobs/berzelius/celeba32_dit_diagonal_100k_viax/` launches seed-zero
  `x/x`, `eps/eps`, and `v/v` on CelebA32 gray at N=256 for 100k updates.
  Its shared config is
  `configs/runners/celeba32_dit_diagonal_100k_viax.py`. The wrapper freezes
  source and settings, uses physical/effective batch 64/256, and provides exact
  recovery-state resume without duplicating training or sampling code.

- Small-DiT production matrix:
  `jobs/berzelius/celeba32_dit_3x3_nscale_1m_viax/` launches the full
  prediction/loss 3x3 at CelebA32-gray training sizes 512, 1024, 2048, and
  4096 for seed 0. Its shared config is
  `configs/runners/celeba32_dit_3x3_nscale_1m_viax.py`. Each run targets one
  million updates. The unthrottled initial array uses 24-hour allocations and
  signal-triggered exact recovery. The manual resume wrapper reuses the frozen
  source, saved run directory, and W&B identity.


- Small-DiT seed-1 architecture robustness matrix:
  `jobs/berzelius/celeba32_dit_robustness_s1_1m_viax/` launches the full
  prediction/loss 3x3 at CelebA32-gray N={512,2048} for seed 1. Its shared
  config is `configs/runners/celeba32_dit_robustness_s1_1m_viax.py`. The
  isolated 18-task package uses the seed-0 scientific settings, fresh run and
  W&B identities, prefix-nested data selections, and manual exact recovery.

- EPS/EPS 64x64 N=512 capacity gate:
  `jobs/berzelius/eps_eps_64_n512/` freezes and validates the 24-row matrix,
  probes six UNet/attention profiles on their real thin or fat GPU class, and
  selects physical batch size under explicit memory limits. Six online-W&B
  smokes and one exact online resume check guard the main launch. The main
  arrays use full named A100 GRES, stop at 400k, and split through matching
  `aftercorr` tasks only when measured runtime requires it. A dependent
  verifier writes `GATE_400K_COMPLETE` after checking every saved run.
  `configs/runners/eps_eps_64_n512_400k.py` owns the shared scientific settings
  and four named checkpoint-sampling profiles.

- Sampling: `mem_param_src/sampling.py` owns direct epsilon-to-velocity
  conversion and Heun's optional final Euler step. `mem_param_src/model.py`
  retains training losses/conversions and forwards `model.sample` arguments.
  `sampling.eps_to_v` and `sampling.heun_final_euler` select the two behaviors.
  Monitoring and saved-checkpoint evaluation pass these controls explicitly.
  Older saved evaluations retain historical settings unless overridden.
  See the runnable API shape and comparison controls in the README's
  "Direct epsilon conversion and final Euler" section.

- Image monitoring: `runners/image_train.py` reads `--f-mem-monitor` and
  `--f-mem-samples`. `runners/training.py` calls `runners/image_monitoring.py`
  at preview checkpoints. It samples once for grids and optional f_mem, using
  `metrics/image/projection.py` and shared `metrics/nearest_neighbors.py`.
  JSON records and the fixed projection stay in the run's `monitoring/` folder.
  Optional full generated tensors go to `monitoring/generated_samples/`.
  Post-training evaluation stores its optional checkpoint tensors in
  `evaluations/<name>/generated_samples/`.
  `runners/image_plotting.py` draws fixed training examples above generated
  samples in `monitoring/comparison_grids/`. It reads saved loss and f_mem
  records to refresh `monitoring/plots/` during training.
  `runners/tracking.py` sends scalar records and grids to W&B when enabled.

- `mem_param_src/` is library code only.
- Job files can call live entrypoints, but must not own study logic.
- Notebooks are for reviewed analysis and probes, not core implementation.
- Plot code must be separated from metric computation.
- Image metric modules must reuse the generic nearest-neighbor implementation;
  they must not carry a second copy of `d1`, `d2`, `r`, or `f_mem` math.
- Checkpoint evaluation must read saved artifacts, not rebuild datasets.

- Dataset-aware image evaluation: `metrics/image/mnist_jointae.py` owns the
  frozen MNIST32 encoder, canonicalization, raw-feature Fréchet math, and
  training-bank feature geometry. Its hashed weights, provenance manifest, and
  compact test statistics live under `metrics/image/assets/` as package data.
  `runners/image_protocol.py` selects MNIST JointAE FD or natural-image
  Inception FID from the saved dataset configuration. It also calibrates
  JointAE, SSCD, DINOv2, or LPIPS memorization. It samples once per checkpoint
  and caches features, float64 statistics, and top-two neighbors by source and
  encoder hashes. Natural-image encoders come from one verified schema-v3 asset
  manifest. `runners/image_eval.py` exposes this path through protocol flags and
  retains the earlier all-checkpoint replay mode. `docs/image_evaluation.md`
  defines the public API, metric names, command, and artifacts.

- Fresh MNIST diagonal production matrix:
  `jobs/berzelius/mnist_diagonal_1m/` launches 36 new one-million-update
  diagonal trajectories. MNIST32 uses U-Net-S on thin A100-40GB nodes.
  MNIST64 uses U-Net-L on fat A100-80GB nodes with five exact-resume stages.
  `configs/runners/mnist_diagonal_1m.py` owns the shared settings. The launcher
  freezes every input, uses W&B project `mnist-unet-diagonal-1m`, and checks
  all 36 final artifact sets.

- Fresh CelebA32 U-Net-S production matrix:
  `jobs/berzelius/celeba32_unet_3x3_1m/` owns the frozen launch, exact manual
  resume, and strict completion workflow for 108 full-grid trajectories.
  `configs/runners/celeba32_unet_3x3_1m.py` owns the shared scientific settings.
  Runs use thin A100-40GB nodes, W&B project `celeba32-unet-3x3-1m`, paired
  target-cell identities, prefix-nested training sizes, and 72 checkpoints.

- CelebA32 native-v/v weighting controls:
  `jobs/berzelius/celeba32_unet_vv_weighting_1m/` owns the frozen launch,
  exact manual resume, and strict completion workflow for nine N=512 runs.
  `jobs/berzelius/celeba32_unet_vv_weighting_n2048_1m/` owns the matching nine
  N=2048 runs. Each has a same-named config under `configs/runners/`. Both
  manifests cross seeds 0-2 with raw `x_equivalent`, `unit`, and
  `eps_equivalent` velocity-loss weights. Within each seed and size, the three
  runs pair source IDs, initialization, random streams, and monitor noise. Each
  N=512 source selection is the prefix of its matching N=2048 selection.

- Fresh CelebA64 RGB diagonal production matrix:
  `jobs/berzelius/celeba64_rgb_diagonal_1m/` owns five staged fat-GPU
  arrays, exact resume, and strict completion checks for 18 fresh trajectories.
  `configs/runners/celeba64_rgb_diagonal_1m.py` owns the successful
  24,386,627-parameter attention U-Net settings. The launcher uses paired
  target-cell identities, prefix-nested N={512,2048} subsets, 72 Bonnaire
  checkpoints, and W&B project `celeba64-rgb-unet-diagonal-1m`.
