# Feature-space `f_mem`

This workflow is post-hoc. It does not add SSCD, LPIPS, DINOv2, or 10,000-sample
nearest-neighbor work to training.

## The three metric paths

| Path | Purpose | Input | Distance | Count |
| --- | --- | --- | --- | ---: |
| `monitor/random_projected_pixels` | Cheap checkpoint monitor during training | clamp/normalize, 32 to 16, fixed Gaussian projection to 256 dimensions | Euclidean | 128 |
| `eval/full_pixels` | Canonical post-hoc pixel result | raw float32 `1x32x32`, flattened to 1,024; no clamp or resize | Euclidean | 10,000 |
| `eval/features/sscd` | Copy-oriented post-hoc result | gray to RGB, bilinear 320, ImageNet normalization | Euclidean on normalized 512-D descriptors | 10,000 |
| `eval/features/lpips` | Low/mid-level perceptual post-hoc result | gray to RGB, native 32 | Euclidean in the weighted five-layer AlexNet embedding; squared distance equals LPIPS v0.1 | 10,000 |
| `eval/features/dinov2` | General visual representation result | gray to RGB, bicubic 224, ImageNet normalization | Euclidean on normalized 384-D final CLS | 10,000 |

Do not compare their absolute ratios or `f_mem` values as if they had the same
scale. The random projection is a fast trend monitor. Full pixels are the raw
native-resolution result. Learned encoders intentionally change the geometry.

The seed-0 checkpoint-390k sensitivity records use SSCD 288/320, LPIPS 32/64,
and DINOv2 224/518. The backbone for DINOv2 is constructed at its native
518-pixel positional grid; the adapter interpolates positions when its input is
224. Primary multi-seed results always use 320, 32, and 224 respectively.

## Visual feature-space sanity notebook

`notebooks/sanity/feature_f_mem_encoder_sanity.ipynb` is a visual, small-scale
study using saved CelebA tensors and staged local weights. It shows disjoint
reference A and held-out B galleries, calibrates one held-out threshold per
space, sweeps interpolation and noise, recovers known noisy-copy fractions,
and places queries beside their nearest A image. The plots cover SSCD, LPIPS,
and DINOv2; nearest neighbors always come from A.

The committed default is `PROFILE = "quick"` and `DEVICE = "cpu"`. It uses 32
reference images, 128 calibration images, 128 validation images, and primary
resolutions only. Its rates are illustrative, not experiment estimates. The
final compact audit table checks dimensions, finite values, repeatability,
exact-copy retrieval, and descriptor norms. A separate LPIPS scatter checks
that squared embedding distance matches LPIPS v0.1.

For the larger researcher-run GPU study, change only:

```python
PROFILE = "full"
DEVICE = "cuda"
RUN_RESOLUTION_CHECK = True
```

Then launch an allocated GPU notebook kernel from the repository root:

```bash
module load Mambaforge
conda run -n mem_param_env jupyter lab notebooks/sanity/feature_f_mem_encoder_sanity.ipynb
```

The full profile uses the declared larger banks and enables SSCD 288/320,
LPIPS 32/64, and DINOv2 224/518 comparisons. It has not been run or submitted
by this implementation session.

## Calculation, in pseudocode

```python
train = saved_training_images()       # [256, 1, 32, 32], exact per training seed
valid = frozen_valid_images()         # [10000, 1, 32, 32], common to every run
samples = replay_checkpoint(           # [10000, 1, 32, 32], exact float32
    common_noise_seed_2003,
    full_heun_steps=50,
)

for space in [full_pixels, sscd, lpips, dinov2]:
    train_f = encode_and_hash(train, space)
    valid_f = encode_and_hash(valid, space)
    sample_f = encode_and_hash(samples, space)

    calibration = nearest_two(valid_f[:1024], train_f)
    validation = nearest_two(valid_f[1024:], train_f)
    generated = nearest_two(sample_f, train_f)

    save(generated.d1, generated.d2)
    save(generated.d1 / generated.d2)
    save(generated.nearest_index, generated.second_nearest_index)

    for q in [0.001, 0.005, 0.01, 0.02, 0.05]:
        rho = quantile(calibration.ratio, q)  # fixed without looking at samples
        report(q, rho,
               mean(validation.ratio < rho),
               mean(generated.ratio < rho))
```

Full pixels additionally retain canonical `rho=0.3`, `0.5`, and `0.7`. Learned
spaces use held-out percentile curves. Choosing another percentile later does
not require regeneration or re-encoding because all feature and neighbor
arrays are retained.

## Encoders and offline assets

`stage_encoder_assets.py` pins DINOv2 source commit
`7764ea0f912e53c92e82eb78a2a1631e92725fc8` and stages official SSCD, LPIPS
AlexNet/calibration, DINOv2 ViT-S/14, and torch-fidelity Inception weights. It
writes SHA-256 values to `outputs/model_assets/feature_f_mem/manifest.json`.
Compute workers call `verify` and use local paths only.

The assets are already staged in this workspace. Verify them with:

```bash
module load Mambaforge
conda run -n mem_param_env python \
  jobs/berzelius/eps_recovery_1m/feature_f_mem/stage_encoder_assets.py verify \
  --manifest outputs/model_assets/feature_f_mem/manifest.json
```

To rebuild the cache on another workspace, replace `verify` with:

```bash
stage --output-root outputs/model_assets/feature_f_mem
```

## Run order

First preview seed 0. This writes nothing and submits nothing:

```bash
bash jobs/berzelius/eps_recovery_1m/feature_f_mem/submit.sh \
  --dry-run --scope seed0
```

The researcher can then submit seed-0 replay:

```bash
bash jobs/berzelius/eps_recovery_1m/feature_f_mem/submit.sh --scope seed0
```

After its runtime and memory use look acceptable, train seeds 1 and 2 with the
six-task launcher documented in `jobs/berzelius/eps_recovery_multiseed/README.md`.
Then evaluate all nine trajectories:

```bash
bash jobs/berzelius/eps_recovery_1m/feature_f_mem/submit.sh \
  --scope all \
  --multiseed-root outputs/job_sets/eps_recovery_multiseed/<batch>/runs
```

The dependency chain is reference features, one unthrottled trajectory array,
then the combiner. No job has been submitted by this implementation session.

## Artifacts and checks

Each task writes exact samples under `generated/`, matrices under `features/`,
raw `d1`, `d2`, `r`, and both indices under `nn_stats/`, and provenance under
`manifests/`. `combined/` contains raw records, percentile curves, seed means
and sample standard deviations, paired parameterization differences,
per-image Spearman correlations, memorized-set Jaccard overlap, ordering tables,
and the primary 1% curve plot.

For all nine trajectories the combiner requires 27 sample banks, 81 primary
learned-space records, 27 full-pixel records, and nine seed-0 sensitivity
records. It rejects nonfinite samples, wrong counts, mixed validation hashes,
or incomplete percentile curves. Low `f_mem` is not interpreted without finite
sample-range diagnostics and acceptable FID-10k.
