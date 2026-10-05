# Three-point epsilon configuration check

This is a small decision run for choosing the next `eps/eps` setup. It does
not evaluate the full checkpoint trajectory. A second two-task job adds the
`v/v` and `x/x` references to the same comparison.

All training runs use one seed: `training.seed=0`. The evaluation noise and real
reference are also fixed across runs. This supports a controlled configuration
choice, but it does not measure variation across training seeds.

## Comparison

The evaluator uses three real, nonzero checkpoints shared by all six epsilon
runs: **100,000**, **390,000**, and **695,000** optimizer updates.

| Task | Existing run | Sampler used for evaluation |
| --- | --- | --- |
| 0 | `eps_old_sampling` | old via-x conversion, safe division, full Heun |
| 1 | `eps_direct_full_heun` | direct conversion, safe division, full Heun |
| 2 | `eps_direct_final_euler` | direct conversion, safe division, final Euler |
| 3 | `eps_exact_full_heun` | direct conversion, exact division, full Heun |
| 4 | `eps_exact_final_euler` | direct conversion, exact division, final Euler |
| 5 | `eps_higher_lr` | direct conversion, safe division, final Euler; LR `1e-3` |

Each checkpoint is evaluated with the configuration attached to that existing
run. The launcher submits the six tasks together with no throttle.

For each point it records:

- FID from 10,000 generated samples against 10,000 CelebA validation images;
- pixel-space `f_mem`, including rho 0.3, 0.5, and 0.7;
- held-out loss at random times and fixed times 0.5 and 0.75;
- raw finite/range diagnostics before FID clipping.

## Reference protocol

The launcher first submits one reference-preparation job. That job uses the
exact training preprocessing and deterministic validation selection to save a
frozen 10,000-image artifact plus a manifest containing its source IDs,
indices, tensor hash, artifact hash, preprocessing, and selection seed. It
must verify that the first 1,024 images, labels, indices, and sample IDs exactly
match the reference prefix saved during training.

Only after that verification succeeds can the six evaluation tasks start.
Every task reads the same frozen reference artifact. The original 1,024-image
`data_bundle.pt` reference remains the source for held-out loss and the
pixel-space `f_mem` baseline. FID uses vendored `torch-fidelity` 0.4.0-beta
with explicit `-1_1` normalization. The `fid_10k` field is populated only when
both the generated and reference counts equal 10,000.

## Run it

Preview only:

```bash
bash jobs/berzelius/eps_recovery_1m/eval/submit.sh --dry-run
```

Expected summary:

```text
tasks=6 checkpoints=3 first=100000 last=695000
Samples/FID reference/loss: 10000/10000/1024
```

Submit:

```bash
bash jobs/berzelius/eps_recovery_1m/eval/submit.sh
```

The dependency chain is reference preparation, unthrottled array `0-5`, then
the combiner. Each task appends one JSONL result immediately after a checkpoint
finishes. The dependent combine job writes:

- `combined/eps_config_checkpoint_metrics.csv`
- `combined/eps_config_comparison.png`
- `combined/summary.json`

Feature-space `f_mem` remains outside this quick comparison. The separate
`../feature_f_mem/` workflow replays matched eps/v/x checkpoints, retains raw
SSCD/LPIPS/DINOv2 arrays, and calibrates learned-space rho from held-out
features. See its README before comparing these pixel results with learned ones.

## Add the v/v and x/x references

After the corrected epsilon evaluation exists, submit both saved reference
parameterizations together:

```bash
bash jobs/berzelius/eps_recovery_1m/eval/submit_references.sh
```

This reuses the same frozen 10,000-image CelebA reference, feature cache, and
evaluation noise. It evaluates sources 6 (`v/v`) and 7 (`x/x`) with their saved
samplers at the same three checkpoints. The dependent combiner adds:

- `combined/reference_checkpoint_metrics.csv`
- `combined/reference_comparison.png`
- `combined/all_parameterization_checkpoint_metrics.csv`
- `combined/all_parameterization_comparison.png`
- `combined/reference_summary.json`
