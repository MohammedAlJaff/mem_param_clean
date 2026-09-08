# Open problem: `eps/eps` succeeds on CelebA32 but fails on Flowers102

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

## Status

**OPEN AND UNSOLVED.**

The current evidence does not support a general claim that `eps/eps` fails.
The old CelebA32 study produced good `eps/eps` results under a different protocol.
The clean Flowers102 sanity run produced noise under its tested protocol.

The research question is:

> Which protocol difference changes `eps/eps` from a useful generator into a
> failed generator?

Do not describe the Flowers102 result as a general parameterization failure.
Do not describe safe division as the cause until a matched control establishes it.

## Meaning of `eps/eps`

The network predicts the path noise `eps`.
The loss compares that prediction directly with the sampled noise target.
Therefore, diagonal `eps/eps` training does not use a target-space division.

For the linear Flow Matching path,

```text
z_t = t x + (1 - t) eps
```

sampling still requires a velocity field.
Exact conversion gives

```text
x_hat = (z_t - (1 - t) eps_hat) / t
v_hat = x_hat - eps_hat
      = (z_t - eps_hat) / t
```

For fixed `z_t`, epsilon error enters the velocity with magnitude `1 / t`.
This conditioning is a sampling issue, even when the training loss is finite.

## Historical CelebA32 evidence

The old checkout is `../mem_par` relative to the clean repository root.
Its saved evaluation table contains direct quantitative evidence:

```text
outputs/evals/mem_boundary_20260427_all_done_quality_20260501/
  all_checkpoint_metrics.csv
```

The table contains:

- 21 completed `eps/eps` runs.
- Seven training-set sizes from 256 through 16,384.
- Three seeds for each training-set size.
- 72 checkpoint rows per run.
- 1,512 `eps/eps` checkpoint rows in total.
- 10,000 generated samples per checkpoint.
- Heun sampling with 50 steps.
- FID from `inception-v3-compat`, feature layer 2048.

The following table summarizes the best and final FID values across three seeds.
Each cell gives `minimum / median / maximum`.

| `N_train` | Best FID | Final FID | Median `f_mem` at best FID |
| ---: | ---: | ---: | ---: |
| 256 | 19.54 / 19.62 / 19.85 | 27.44 / 27.55 / 29.04 | 0.8631 |
| 512 | 11.73 / 11.84 / 11.86 | 14.46 / 15.07 / 15.99 | 0.6706 |
| 1,024 | 9.70 / 10.73 / 11.79 | 10.04 / 10.95 / 11.86 | 0.5025 |
| 2,048 | 16.20 / 16.48 / 17.17 | 16.90 / 17.17 / 18.11 | 0.0702 |
| 4,096 | 13.15 / 13.20 / 13.81 | 15.64 / 16.69 / 17.73 | 0.0002 |
| 8,192 | 8.71 / 8.87 / 9.23 | 10.52 / 11.31 / 12.19 | 0.0000 |
| 16,384 | 6.10 / 6.76 / 7.37 | 9.11 / 11.38 / 12.30 | 0.0000 |

Across all 21 runs, the best-FID median is `11.84`.
The minimum best FID is `6.10`.
The old canonical `f_mem` column uses `rho = 0.5`.

These values show that the historical `eps/eps` models produced useful samples.
They also show the expected memorization change across training-set size.
The old project summary reports `eps/eps` as the latest and least memorizing diagonal.

The user also reports visually good CelebA32 grayscale samples.
The current machine does not contain the old checkpoint tree referenced by the CSV.
Therefore, this pass did not replay or visually inspect those old checkpoints.

Additional old evidence appears in these files:

- `README.md`: CelebA32 parameterization ordering and `F_mem@FID_best`.
- `PAPER_OBJECTIVE.md`: dataset, path, and checkpoint-level measurements.
- `CODING_JOURNAL.md`: 72 checkpoints through one million updates.
- `jobs/job_sets/mem_gen_boundary_20260427/submit.sh`: the exact sweep grid.

## Clean Flowers102 failure

The clean run directory is:

```text
outputs/sanity/flowers102_runner_eps_eps_exact_64_n8_10k_20260820T191129Z
```

The run used:

- Eight fixed Flowers102 training images.
- RGB images with shape `3 x 64 x 64`.
- One seed, `7`.
- A 669,251-parameter UNet.
- Adam with learning rate `1e-3`.
- 10,000 optimizer updates.
- Uniform training times from `0.01` through `0.99`.
- Exact `eps`-to-velocity division.
- Heun sampling with 50 steps.
- A fixed bank of 64 initial noise tensors.
- Final output `x_1`.

The recorded loss decreased from `0.993115` to `0.007049`.
This optimization result did not produce useful samples.
The final grid contained colored noise.

The 50-step final result had:

- Sample range `[-14.723, 16.013]`.
- Fraction inside `[-1, 1]` equal to `0.3156`.
- Mean pixel-space `d1` equal to `270.293`.
- Mean nearest-neighbor ratio `r` equal to `0.988833`.
- `f_mem = 0` at `rho = 0.3`, `0.5`, and `0.7`.

The fixed checkpoint replay gave:

| Heun steps | Sample range | Fraction in `[-1, 1]` | Mean `d1` |
| ---: | ---: | ---: | ---: |
| 50 | `[-14.723, 16.013]` | 0.3156 | 270.293 |
| 200 | `[-14.627, 15.847]` | 0.3175 | 268.527 |
| 500 | `[-14.618, 15.833]` | 0.3177 | 268.379 |

The sample RMSE was `0.029318` between 50 and 200 steps.
It was `0.002376` between 200 and 500 steps.
The 200-step and 500-step outputs converged to the same failed result.

On exact path states at `t = 0.01`, epsilon RMSE was `0.027713`.
The induced velocity RMSE was `2.771305`.
The implementation matched the direct formula within `2.25e-5` maximum error.

These checks exclude two narrow explanations for this saved model:

- A 50-step Heun grid is not the main failure cause.
- An algebraic error in the tested conversion is not the main failure cause.

They do not identify the actual cause.

## Confirmed protocol differences

This comparison is not controlled.
Many variables changed together.

| Variable | Historical CelebA32 | Clean Flowers102 sanity |
| --- | --- | --- |
| Dataset | CelebA faces | Oxford Flowers102 |
| Tensor shape | `1 x 32 x 32` | `3 x 64 x 64` |
| Values per image | 1,024 | 12,288 |
| Training-set size | 256 to 16,384 | 8 |
| Seeds | 0, 1, and 2 | 7 |
| Model parameters | 3,917,409 | 669,251 |
| UNet profile | Multipliers `(1,2,4)` | Multipliers `(1,2)` |
| Attention | Resolutions 16 and 8 | None |
| Residual blocks | Two per level | One per level |
| Dropout | 0.1 | 0.0 |
| Optimizer | Adam | Adam |
| Learning rate | `1e-4` | `1e-3` |
| Optimizer updates | 1,000,000 | 10,000 |
| Batch cap | 256 | 8 |
| Training interval | `[0.001, 0.999]` | `[0.01, 0.99]` |
| Time distribution | Uniform | Uniform |
| Sampling interval | `[0.001, 0.999]` | `[0.01, 0.99]` |
| Sampling division | Always safe, epsilon `0.05` | Exact |
| Sampler | Heun, 50 steps | Heun, 50 steps |
| Saved sample | Integrated `x_t_max` | Converted endpoint `x_1` |
| Runtime | CUDA with TF32 requested | MPS with FP32 |
| Quality evaluation | FID on 10,000 samples | Visual grid and pixel metrics on 64 samples |

Both studies use the same linear path and `[-1, 1]` image normalization.
The CelebA transform is still present in the clean loader.

### Important safe-division difference

The old code always applied denominator protection during conversion.
It used `safe_div_denom_eps = 0.05` and started sampling at `t = 0.001`.

For early `eps` conversion, the old implementation used:

```text
d = max(t, 0.05)
x_hat = (z_t - (1 - t) eps_hat) / d
v_hat = x_hat - eps_hat
```

Its epsilon sensitivity is `(1 - t) / d + 1`.
This value is about `21` at the early clamped times.

The clean exact run starts at `t = 0.01`.
Its epsilon sensitivity is `1 / t = 100` at the first time.

The old 50-step grid clamps the first three grid locations.
This changes the learned ODE during the most sensitive part of sampling.
It is a strong candidate explanation, but it remains untested as a matched control.

The earlier clean safe replay used epsilon `0.01`.
That threshold equals the clean start time, so it did not change the equation.
It does not test the historical `0.05` policy.

### Explicit dimensionality hypothesis

The historical CelebA tensor has `1 x 32 x 32 = 1,024` scalar values.
The failed Flowers tensor has `3 x 64 x 64 = 12,288` scalar values.
The Flowers state is therefore 12 times larger.

If per-coordinate error stays fixed, its total L2 norm grows with `sqrt(D)`.
The expected norm change is `sqrt(12)`, or approximately `3.46`.
This scaling does not prove that dimension causes the failed trajectory.
It makes raw cross-resolution distance comparisons invalid without normalization.

Higher dimension can interact with several finite-model effects:

- The model has less effective capacity per image coordinate.
- High-resolution spatial frequencies can be harder to predict.
- RGB prediction must model color structure absent from grayscale data.
- Small per-coordinate endpoint errors can accumulate into a large state error.
- A fixed optimizer budget can fit low-resolution data sooner.

Test dimension while keeping the underlying Flowers images fixed.
Use the same source indices and deterministic preprocessing for each condition:

1. RGB at `64 x 64`, with `D = 12,288`.
2. RGB at `32 x 32`, with `D = 3,072`.
3. Grayscale at `32 x 32`, with `D = 1,024`.

Keep the training schedule, batch, optimizer, time distribution, and sampler fixed.
Record the exact parameter count for each network shape.
Use the same random seed, but save a separate `z_init` for each tensor shape.

Report per-coordinate RMSE and L2 distance divided by `sqrt(D)`.
Do not compare raw pixel `d1` across these shapes as if its scale were fixed.

Interpret the controls as follows:

- Improvement from RGB64 to RGB32 supports a resolution or dimension effect.
- Improvement only after grayscale conversion implicates color or channel count.
- Failure on Flowers gray32 but success on CelebA gray32 rejects raw dimension alone.
- Success from the historical clamp at every shape implicates endpoint conditioning.

These outcomes can still contain interactions.
Confirm any dimensionality result with more than one seed.

## What we know

1. Historical `eps/eps` succeeded on CelebA32 under the old protocol.
2. Clean `eps/eps` failed on the recorded Flowers102 sanity protocol.
3. Low diagonal epsilon loss does not guarantee a useful sampled field.
4. Increasing Heun resolution does not repair the fixed Flowers102 model.
5. Exact endpoint conversion amplifies early epsilon error by `1 / t`.
6. The old sampler modified this early conversion with a larger clamp.
7. The old and clean experiments are not matched well enough for attribution.

## What we do not know

We do not yet know whether the Flowers102 failure comes from:

- Exact division near the noise endpoint.
- The changed start and end times.
- The smaller model.
- The shorter training budget.
- The ten-times larger learning rate.
- The larger image dimension.
- Dataset structure.
- Off-path error accumulation.
- The approximate noise start at nonzero time.
- A code-path difference outside the checked conversion formula.
- An interaction between several of these variables.

We also do not know whether clean CelebA32 reproduces the old result.

## Ranked resolution plan

### 1. Replay the fixed Flowers102 model with the old sampling policy

Do not retrain for this test.
Use the same checkpoints and `z_init`.

Create separate named evaluations:

1. Keep `[0.01, 0.99]` and set safe epsilon to `0.05`.
2. Use `[0.001, 0.999]`, safe epsilon `0.05`, and `x_t_max`.
3. Use `[0.001, 0.999]`, exact division, and `x_t_max` as a diagnostic.

The first evaluation isolates the clamp.
The second reproduces the old sampling policy.
The third measures the unclamped endpoint instability.

### 2. Test dimensionality with the same Flowers images

Run the RGB64, RGB32, and grayscale32 conditions defined above.
Use the same selected image identities and the same resolved training protocol.

First use a short matched screen.
Then confirm any apparent transition with a larger training budget and multiple seeds.

### 3. Recover one historical checkpoint and its artifacts

Search Berzelius or the preserved run storage for one completed `eps/eps` run.
Prefer `N_train = 256`, seed `0`, and its FID-best checkpoint.

Replay that checkpoint with:

1. The old sampler.
2. The clean sampler in `x_t_max` mode.
3. The clean sampler in `x_1` mode.
4. Exact and safe division settings.

This is the strongest code-path comparison because it avoids retraining.

### 4. Reproduce one CelebA32 run through the clean runner

Use the exact old data transform, selected indices, network profile, optimizer,
time interval, and sampling policy.

Start with `N_train = 256` and seed `0`.
Save checkpoints at 10,000, 100,000, and 1,000,000 updates.
This ladder separates training-budget effects without changing the final target.

Use the historical FID extractor only for protocol comparison.
Also compute the clean memorization metrics from the same generated samples.

### 5. Build a matched dataset bridge

After reproduction, change one factor at a time.
At minimum, control these factors:

1. CelebA grayscale 32 versus Flowers grayscale 32.
2. Grayscale 32 versus RGB 64.
3. Historical UNet versus the small clean UNet.
4. Learning rate `1e-4` versus `1e-3`.
5. Update count 10,000 versus the historical ladder.
6. Exact division versus safe epsilon `0.05`.

Do not interpret a dataset effect while architecture and image shape also change.

### 6. Measure where trajectory error begins

For each matched run, record time-binned epsilon error on exact path states.
Also record prediction norms and velocity norms on generated trajectory states.

The exact-path test measures supervised fit.
The generated-path test measures off-path feedback.
Their divergence will locate the failure mechanism more precisely.

### 7. Compare the native velocity control

Compare `eps/eps` with `v/v` plus `eps_equivalent` weighting.
Keep the network, data, optimizer, time samples, and initial state fixed.

This control separates the weighted objective from the predicted representation.
Run it after the historical sampling difference is tested.

## Common interpretation errors

- Do not treat a low epsilon MSE as proof of good generation.
- Do not compare 10,000 updates with one million updates as equivalent training.
- Do not compare the old FID directly with the clean pixel distance.
- Do not compare a best-FID checkpoint with only a final checkpoint.
- Do not call epsilon `0.01` a safe-division test when `t_min` is also `0.01`.
- Do not change `z_init` between sampler controls.
- Do not rebuild the training subset during evaluation.
- Do not mix `x_t_max` and `x_1` outputs in one result table.
- Do not infer a dataset effect from different image shapes and model sizes.
- Do not infer a numerical cause from one seed.

## Resolution criteria

Close this problem only when all conditions hold:

1. A clean replay or clean training run reproduces useful CelebA32 `eps/eps` samples.
2. A matched experiment identifies which change or interaction creates the failure.
3. The result holds across more than one seed.
4. The report separates best-checkpoint quality from final-checkpoint quality.
5. The final sampling division and endpoint policy has a measured justification.
6. The exact configs, subsets, checkpoints, `z_init`, and metrics remain saved.

Until then, keep this problem open.
