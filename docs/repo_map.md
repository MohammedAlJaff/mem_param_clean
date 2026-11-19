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
plotting/          fast CSV-only research figure scripts
plot_data/         tracked, deterministic scalar tables grouped by experiment
jobs/berzelius/    Slurm launchers for Berzelius, if retained
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
- Baselines and third-party code go under `external/` only if retained.
- Generated outputs do not enter the clean repo by default. The narrow exception is
  reviewed scalar-only exports under `plot_data/`; checkpoints, tensors, and rendered
  figures remain untracked.

Boundary rules:

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

- AISTATS experimental-design source:
  `docs/aistats_experimental_design_appendix.md` is the self-contained,
  long-form description of the target parameterizations, common training and
  evaluation protocol, all five production blocks, supporting pilots, and
  rules that separate scientific evidence from engineering validation. Future
  paper text can shorten this source without recovering settings from launch
  scripts.

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
  defines the public API, metric names, command, and artifacts. The minimal
  `mnist_checkpoint_metrics.ipynb` and `celeba_checkpoint_metrics.ipynb`
  notebooks each select one checkpoint, generate eight samples, display a grid,
  and report dataset-aware quality and memorization metrics. The reusable
  `runners/sampling_benchmark.py` measures sampling without metrics.
  `runners/celeba_test_reference.py` builds shape-matched clean CelebA test-10k
  banks from official filenames and declared saved training bundles. It records
  raw-JPEG and transformed-tensor collision checks and full provenance.
  `runners/checkpoint_metrics_benchmark.py` adds one interactive CelebA32 gate
  with raw-pixel and SSCD `f_mem`, Inception FID-10k, reusable features,
  top-two neighbors, and verified generated-bank reuse.
  `runners/figure_bundle.py` verifies completed checkpoint gates and exports
  portable CSV, JSON, NPZ, and PNG figure inputs plus one Mac-ready archive.
  It does no sampling, encoding, neighbor search, or FID computation.
  `runners/evaluation_noise.py` builds and verifies persistent, shape-specific
  seed-2003 evaluation-noise banks. `runners/full_suite_plots.py` writes the
  development trajectory, grid, diagnostic, and nearest-neighbor plots.
  `runners/weighting_trajectory_plots.py` validates the complete 18-run
  loss-weighting result matrix and writes seed-mean trajectories with sample
  standard-deviation bands. It keeps N=512 and N=2048 separate and places all
  three weighting schemes in every metric panel.
  `plotting/weighting/` splits the accepted aggregate once into one CSV per
  metric. Its small commands then render only one requested figure from those
  tables, without scanning checkpoints or regenerating unrelated plots.
  `docs/fast_plotting.md` defines the reusable CSV-first plotting workflow,
  its aggregate schema, one-figure command pattern, and three-second runtime
  target. `docs/full_evaluation_readiness.md` records the final evaluator
  coverage audit, shape-specific reference artifacts, experiment-block state,
  storage constraint, and rolling-batch launch gates.
  `runners/full_suite_retention.py` creates an identity-checked deletion
  manifest after paper-tier acceptance. It preserves metrics, plots,
  statistics, neighbor evidence, and selected sample tensors while removing
  only reproducible sample and feature caches.
  `jobs/berzelius/all_1m_paper_eval/` validates the exact completed-run matrix,
  freezes source and input hashes, and submits the 198 paper evaluations as a
  dependency chain. One MNIST32 task gates the later arrays. Each task repeats
  the evaluator for cache-reuse validation, applies strict acceptance, and then
  performs exact-path retention. The launcher supports task-specific restart
  from the same frozen batch.
  `jobs/berzelius/celeba32_streaming_paper_eval/` replaces the CelebA32
  portion with independent arrays and one evaluator pass. It validates and
  removes reproducible caches after each checkpoint. Its `--dit-seed1`
  selection submits only the 18 newly completed seed-1 DiT runs, with all
  72 checkpoints and 10,000 samples each. Workers import only the frozen
  evaluator source. See that directory's README for submit and restart commands.


- Historical storage audit:
  `runners/old_storage_cleanup.py` creates immutable keep and delete manifests
  for selected `old_mem_par` output roots. It is dry-run by default. It merges
  resumed attempts, accounts for hard links, keeps every tenth checkpoint plus
  the first and final checkpoints, protects documented checkpoints, and hashes
  documented temporary copies before classifying them. Applying a reviewed
  manifest requires the exact old root and removes only explicit regular-file
  paths. The audit also snapshots the obsolete intrinsic/ambient prototype
  source before it lists those live source files in a separate deletion group.
  See `docs/old_storage_cleanup.md` for the command and safety checks.
  `runners/full_suite_validate.py` enforces checkpoint counts, finite values,
  f_mem ranges, sample shapes, cache reuse, and readable plots.
  `notebooks/sanity/full_metric_suite_smoke.ipynb` reviews all three smoke
  outputs and has a disabled one-checkpoint rerun cell.
  `jobs/berzelius/full_metric_suite_smoke/` freezes and launches either the
  all-checkpoint 512-sample trajectory smoke or the final-checkpoint
  10,000-sample replica smoke. Both profiles use isolated development output
  roots.
  `jobs/berzelius/celeba32_vv_weighting_paper_eval/` freezes and launches all
  18 paper-tier CelebA32 `v/v` loss-weighting runs. The matrix contains
  `x_equivalent`, `unit`, and `eps_equivalent` at `N={512,2048}` and seeds
  `{0,1,2}`. It evaluates 10,000 fixed-noise samples at all 72 checkpoints,
  verifies all frozen inputs before and after each task, and requires a
  cache-only second evaluator invocation.
  `checkpoint_sampling_runtime.ipynb` exposes sampling interactively.
  `jobs/berzelius/checkpoint_sampling_runtime/` contains the short sampling
  job and both direct interactive commands.
  `docs/unified_checkpoint_evaluation_plan.md` defines the later run-level
  production workflow, storage policy, restart boundaries, and historical
  10,000-sample A100 runtime evidence.

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
