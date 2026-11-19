# Checkpoint sampling runtime benchmark

This job measures sampling only. It does not encode samples or compute metrics.

The default request uses the completed CelebA32 N=2048, seed-0, x/x run. It
samples step 1,000,000 with the saved 50-step full-Heun settings. It measures
10,000 images on one thin A100-40GB node.

The worker saves `batches.csv`, `progress.json`, and `summary.json`. It does not
save the generated 10,000-image bank. The summary reports:

- measured sampling time and throughput;
- measured or projected time for 10,000 samples at one checkpoint;
- projected sampling time for every checkpoint in the selected run;
- setup, checkpoint-load, warm-up, GPU-memory, and raw-range diagnostics.

The full-run value is a sampling-only projection. Metric encoders, nearest
neighbors, artifact writes, and scheduler wait time are outside this estimate.

## Preview and submit

```bash
bash jobs/berzelius/checkpoint_sampling_runtime/submit.sh --dry-run
bash jobs/berzelius/checkpoint_sampling_runtime/submit.sh
```

The submission requests the permitted `devel` QoS under project
`berzelius-2026-269`, one A100-40GB GPU, and 30 minutes. The worker stops before
the wall limit after a complete sample batch. Short wall time can improve
scheduling, but it does not guarantee an immediate start.

Use another saved image run without changing the benchmark code:

```bash
bash jobs/berzelius/checkpoint_sampling_runtime/submit.sh \
  --run-dir RUN_DIR \
  --checkpoint-step 1000000 \
  --measure-count 10000 \
  --target-count 10000 \
  --batch-size 64
```

Set `--measure-count 512` for a faster projection. Keep `--target-count 10000`
so the report still estimates the paper-scale sample bank.

## Interactive notebook

Request a short thin-GPU allocation:

```bash
salloc -A berzelius-2026-269 -t 00:30:00 \
  --constraint=thin --gres=gpu:A100-SXM4-40GB:1 \
  --cpus-per-task=4 --mem=32G
srun --pty bash
module load Mambaforge
mamba activate mem_param_env
jupyter lab --no-browser
```

Open `notebooks/sanity/checkpoint_sampling_runtime.ipynb`. Change one settings
cell. The notebook calls the same benchmark function as this batch job.

## Fixed clean CelebA test reference

Build the reference once on a CPU node:

```bash
python -m mem_param_src.runners.celeba_test_reference \
  --celeba-root ../old_mem_par/data \
  --source-run-root outputs/job_sets/celeba32_unet_3x3_1m/20260927T191224Z_2530324 \
  --source-run-root outputs/job_sets/celeba32_dit_3x3_nscale_1m_viax/20260926T104429Z_1787399 \
  --source-run-root outputs/job_sets/celeba32_dit_diagonal_100k_viax/20260925T172747Z_3410286 \
  --source-run-root outputs/job_sets/celeba32_dit_robustness_s1_1m_viax/20260929T124425Z_2240178 \
  --source-run-root outputs/job_sets/celeba32_unet_vv_weighting_1m/20260928T144550Z_222375 \
  --source-run-root outputs/job_sets/celeba32_unet_vv_weighting_n2048_1m/20260928T151300Z_373224 \
  --source-run-root outputs/job_sets/celeba64_rgb_diagonal_1m/20260928T131627Z_3579761 \
  --output-artifact outputs/reference_banks/celeba_test_seed1_clean_n10000.pt
```

The builder maps every saved train index through the official train catalog.
It rejects raw JPEG and transformed-tensor collisions. It walks the seed-1
test permutation until it accepts 10,000 unique images. The artifact and JSON
manifest record the identities, labels, preprocessing, rejection counts,
partition hash, source runs, source tensor hashes, and complete artifact hashes.
The builder refuses to overwrite an existing artifact.

## Full interactive checkpoint evaluation

Run this command inside an existing GPU allocation:

```bash
cd /proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_param_clean
bash jobs/berzelius/checkpoint_sampling_runtime/run_interactive_metrics.sh
```

The default command evaluates step 1,000,000 from the same run as the completed
2026-10-03 gate. It reuses and verifies this generated bank without sampling:

```text
outputs/interactive_checkpoint_metrics/20261003T000727Z/generated_samples.pt
```

The evaluator checks the checkpoint hash, sampler settings, noise seed, sample
count, and generated tensor hash. It then:

- checks the clean test bank before any sampling or metric computation;
- computes raw-pixel `f_mem` at fixed `rho={0.3,0.5,0.7}`;
- computes test-calibrated raw-pixel and SSCD `f_mem` curves;
- computes Inception FID from 10,000 generated and 10,000 test images;
- saves features and top-two neighbor evidence for later threshold changes;
- prints progress, stage times, metric values, and the 72-checkpoint estimate.

The positional arguments are checkpoint step, sampling batch size, output
directory, and generated-bank path:

```bash
bash jobs/berzelius/checkpoint_sampling_runtime/run_interactive_metrics.sh \
  1000000 64 outputs/my_clean_test_metrics \
  outputs/interactive_checkpoint_metrics/20261003T000727Z/generated_samples.pt
```

The default output is a new timestamped directory under
`outputs/interactive_checkpoint_metrics/`. The completed validation result in
`20261003T000727Z/` remains unchanged. The final scalar record is
`summary.json`.

## Completed clean-test measurement

The completed output is
`outputs/interactive_checkpoint_metrics/20261003T084819Z/`. It reused the
verified generated bank and reported:

- FID-10k: `34.8159`;
- raw-pixel f_mem at fixed rho `0.3`: `0.5197`;
- calibrated raw-pixel f_mem at alpha `0.01`: `0.9450`;
- calibrated SSCD f_mem at alpha `0.01`: `0.6235`.

The metric-only reuse pass took `31.992` seconds. The saved sampling kernel took
`154.401` seconds. Their full 72-checkpoint projection is `3.469` GPU-hours on
the measured A100-80GB setup.

Real Slurm replay job `17693255` requested one thin A100, normal QoS, and 30
minutes. Slurm initially estimated the next day, but the job started after 5
minutes 44 seconds and completed in 1 minute 4 seconds. Its output is
`outputs/interactive_checkpoint_metrics/20261003T090549Z/`. It reproduced all
four metrics above to displayed precision. Treat start times as estimates,
not reservations.

## Export the figure bundle

Run the exporter on a CPU after the metric job is complete. Do not run this
command inside the GPU job. The clean-gate bundle below already exists. The
command records its exact provenance:

```bash
cd /proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_param_clean
module load Mambaforge
conda run -n mem_param_env python -m mem_param_src.runners.figure_bundle \
  --evaluation-root outputs/interactive_checkpoint_metrics/20261003T090549Z \
  --output-dir outputs/figure_bundles/celeba32_checkpoint_gate_test_v1 \
  --study-id celeba32-checkpoint-gate-test-v1
```

The exporter refuses to replace an existing bundle. Use a new output name for
a new export. The result is a directory and this transfer archive:

```text
outputs/figure_bundles/celeba32_checkpoint_gate_test_v1.tar.gz
```

The archive is 909 KiB. Its SHA-256 value is
`ce50051f9396e7773a904b26ed5a1fd1a600cb957d2a7af16ff11cbc9fbccf36`.
It contains no checkpoint, model, PyTorch file, or feature matrix.
