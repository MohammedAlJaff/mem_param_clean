# Full evaluation readiness audit

Date: 2026-10-04

## Verdict

The full-suite evaluator is ready for every completed production model family.
It now supports grayscale 32×32, grayscale 64×64, and RGB 64×64 tensors.
It also supports the saved U-Net and Small-DiT checkpoints.

The 198 completed runs were submitted on 2026-10-04 as eight dependent arrays.
A one-run MNIST32 gate must pass before any downstream array can start. Each
accepted result is compacted before its array dependency releases the next
batch. This keeps the uncompressed 10,000-sample caches within project quota.

The frozen batch is
`outputs/evals/all_1m_full_10000/20261004T005702Z_3271331`.

## Experiment-block status

| Block | Runs | State | Checkpoints per run | Shape | Evaluation input status |
| --- | ---: | --- | ---: | --- | --- |
| CelebA32 U-Net 3×3 | 108 | trained | 72 | 1×32×32 | submitted |
| MNIST32 diagonal | 18 | trained | 20 | 1×32×32 | submitted after gate |
| MNIST64 diagonal | 18 | trained | 20 | 1×64×64 | submitted after gate |
| CelebA64 RGB diagonal | 18 | trained | 75 | 3×64×64 | submitted |
| CelebA32 Small-DiT seed 0 | 36 | trained | 72 | 1×32×32 | submitted |
| CelebA32 Small-DiT seed 1 | 18 | training | 65–66 saved now | 1×32×32 | wait for training |
| CelebA32 v/v weighting | 18 | trained and evaluated | 72 | 1×32×32 | accepted |

The submitted blocks contain 198 runs. The seed-1 DiT block adds 18 runs after
training completes. The weighting block already has accepted paper-tier results
for all 18 runs and 1,296 checkpoints.

## Code changes from this audit

- The protocol manifest records `image_channels` for every metric mode.
- The strict validator reads the expected channel count from the protocol.
- The clean CelebA reference validator reads image shape from its manifest.
- The clean CelebA reference builder accepts image size, channel count, and
  resize size. Its old 32×32 grayscale defaults remain unchanged.
- The mocked natural-image integration now uses an RGB 64×64 tensor.
- The MNIST fixture now follows the saved run size, so MNIST64 uses real
  1×64×64 inputs before JointAE canonicalization.
- The launchers copy the exact torch-fidelity package into the frozen batch.
  Workers no longer import live source from `old_mem_par`.


## Immutable reference artifacts

The common seed-2003 stream is shape-specific. All training seeds use the same
evaluation stream. The training monitor tensors remain unchanged.

- Grayscale 32×32 noise:
  `outputs/reference_banks/evaluation_noise/canonical_gaussian_seed2003_n10000.pt`.
- Grayscale 64×64 noise:
  `outputs/reference_banks/evaluation_noise/canonical_gaussian_seed2003_n10000_c1x64.pt`.
  Artifact SHA-256:
  `23b30f2f826937e754c6a60a14a4ece6a842d528feb6025a0968091ceb834256`.
- RGB 64×64 noise:
  `outputs/reference_banks/evaluation_noise/canonical_gaussian_seed2003_n10000_c3x64.pt`.
  Artifact SHA-256:
  `fb490741ba39db0e07211c65ac7535e5f9472e7d02a0ad21f5113424391210be`.
- Clean CelebA32 test bank:
  `outputs/reference_banks/celeba_test_seed1_clean_n10000.pt`.
- Clean RGB64 CelebA test bank:
  `outputs/reference_banks/celeba64_rgb_test_seed1_clean_n10000.pt`.
  Artifact SHA-256:
  `20ccce3201d7ef58b6870a1a1af5fbc2d7368dce7f5abab9e19ab22d923e0b35`.

The RGB64 builder audited all 18 saved training banks. It selected 10,000
official-test images after examining 10,001 candidates.

## Verification evidence

The evaluator audit passed 32 focused tests in `mem_param_env`. The final
submission gate passed 35 focused protocol, metric, figure, and retention tests.
Together they cover canonical noise, reference collision checks, MNIST32,
MNIST64, RGB64, full-suite resolution, held-out losses, cache identities,
restart reuse, natural-image features, and strict acceptance.

A separate CPU preflight used real artifacts. For one representative run from
each row below, it verified the data manifest, canonical noise, reference
compatibility, checkpoint count, final step, and strict checkpoint loading:

- CelebA32 U-Net: 72 checkpoints, final step 1,000,000.
- MNIST32 U-Net: 20 checkpoints, final step 1,000,000.
- MNIST64 U-Net: 20 checkpoints, final step 1,000,000.
- CelebA64 RGB U-Net: 75 checkpoints, final step 1,000,000.
- CelebA32 Small-DiT: 72 checkpoints, final step 1,000,000.

The encoder-asset manifest also passed its hash and source checks. Python
compilation, Ruff, and `git diff --check` passed for the focused changes.

## Existing weighting-result provenance note

The accepted weighting evaluation passed all numerical and cache checks. Its
original launcher froze the torch-fidelity Git commit but did not copy the
dirty package files into the batch. Those source files have modification times
from 2026-06-12, before the 2026-10-03 evaluation. Their current deterministic
tree hash is
`f8da0188bc476ac3c700b1d3773c6174ce74cec3d4a068e12f960b446285ca7a`.

This is strong reconstruction evidence, but it is not a submission-time hash.
Before publication, either archive this exact source with the accepted result
and record the limitation, or recompute the Inception features and FID under a
new frozen batch. The other metric encoders do not use torch-fidelity.

## Storage gate

Project usage is 13.0 TiB. The normal quota is 14.6 TiB. The accepted
18-run weighting evaluation uses 521 GiB, or about 29 GiB per run.
Retaining the same cache layout for 198 more runs would require about 5.6 TiB.

One representative run stores about 29 GiB under checkpoint caches. At its
final checkpoint, generated features use 366 MiB and samples use 40 MiB.
Most of this data is reproducible after acceptance.

`runners/full_suite_retention.py` now writes a dry-run deletion manifest by
default. Apply mode validates every path identity before it unlinks any file.
It keeps metrics, identities, completion records, neighbor evidence, plots,
statistics, and selected qualitative sample tensors. Two mocked before-and-
after tests pass. The submitted MNIST32 gate is the required real-output test;
downstream arrays cannot start unless that gate succeeds.

## Launch gates for each rolling batch

1. Require `state=trained`, `final_step=1000000`, and `error=null`.
2. Require the exact block-specific checkpoint count and final checkpoint.
3. Freeze source, environment, run paths, artifacts, encoder assets, the exact
   torch-fidelity package, and all hashes.
4. Select the canonical noise artifact by tensor shape.
5. Select the clean CelebA reference by preprocessing shape. MNIST uses its
   saved official-test bank and packaged 10,000-image quality statistics.
6. Run the complete suite with 10,000 samples and 512 held-out loss images.
7. Repeat the evaluator and require complete cache reuse.
8. Run strict paper-tier validation and build the block aggregate.
9. Preserve the selected figure inputs, then apply the reviewed retention
   manifest.
10. Check `nscquota` before launching the next batch.

## Recommended launch order

The submitted dependency graph implements the recommended order. Job `17696049`
is the one-run MNIST32 gate. Jobs `17696050` and `17696051` then evaluate the
remaining MNIST32 and all MNIST64 runs in parallel. Jobs `17696052` through
`17696056` evaluate CelebA64 RGB, Small-DiT seed 0, and three 36-run CelebA32
U-Net chunks in sequence.

The frozen launcher is `jobs/berzelius/all_1m_paper_eval/`. It supports exact
task restart from the first incomplete checkpoint and restart of an interrupted
retention manifest. Evaluate the seed-1 DiT block only after all 18 runs reach
one million steps.
