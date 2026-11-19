# Dataset-aware image evaluation

The saved-run evaluator selects a quality representation from the dataset. It
does not label every Fréchet distance as FID.

| Metric name | Representation | Dataset role |
| --- | --- | --- |
| `inception_fid` | Inception-v3 | Primary natural-image quality |
| `mnist_fd_jointae32_v1` | Raw JointAE 128D features | Primary MNIST32 and MNIST64 quality |
| `mnist_cafd_jointae32_v1` | Per-class raw JointAE features | MNIST class support diagnostic |
| `f_mem_mnist_jointae32_v1` | Standardized and normalized JointAE features | MNIST memorization |

Do not compare values from different rows as one metric. The feature spaces,
dimensions, preprocessing, and finite-sample floors differ.

## Frozen MNIST encoder

The metric asset was trained on the official MNIST training split. A seeded
55,000-image subset updated weights. The remaining 5,000 images selected the
checkpoint. The official 10,000-image test split supplied final metrics and
reference statistics only.

The public loader resolves the checkpoint through Python package data and
checks its SHA-256 value:

```python
import torch

from mem_param_src.metrics.image import load_mnist_jointae32

images = torch.randn(16, 1, 64, 64)  # One saved MNIST64 batch in [-1, 1].
encoder = load_mnist_jointae32(device="cuda")
features = encoder(images, image_range="-1_1")

assert features.shape == (16, 128)
```

The loader returns the raw linear projection. It does not apply the classifier's
LayerNorm or SiLU. It also does not apply feature standardization or L2
normalization.

MNIST canonicalization converts the declared range to `[0,1]`, accepts one
grayscale channel or three replicated grayscale channels, clamps, and resizes
to 32 by 32. Downsampling uses bilinear antialiasing. No alignment transform is
learned or estimated.

The bundled files are:

- `metrics/image/assets/mnist_jointae32_v1.pt`: trained weights;
- `metrics/image/assets/mnist_jointae32_v1.json`: provenance and hashes;
- `metrics/image/assets/mnist_jointae32_v1_reference.pt`: compact global and
  per-class official-test statistics.

## MNIST memorization geometry

MNIST FD uses raw features. MNIST feature-space memorization transforms the
same raw features before calculating neighbors:

1. Encode the exact saved training bank.
2. Compute each coordinate's mean and population standard deviation from that
   bank.
3. Standardize the training and query features with those values.
4. L2-normalize each feature row.
5. Calculate exact Euclidean top-two neighbors with chunked `torch.cdist`.

The saved neighbor artifact retains `d1`, `d2`, `r`, `nn_idx`, and
`second_nn_idx`. It never stores the full query-by-reference distance matrix.

Threshold calibration uses a frozen prefix of the saved held-out bank. The
remaining held-out examples validate the rates. The evaluator freezes one
`rho` for each alpha in `{0.005, 0.01, 0.02, 0.05}` and reuses these values for
all selected checkpoints.

## Run-folder command

The protocol mode selects checkpoints and writes restartable artifacts:

```bash
mem-param-image-eval \
  --run-dir RUN_DIR \
  --checkpoints 100000,390000,695000,1000000 \
  --metrics fid,f_mem,class_fd \
  --feature-spaces dataset_default \
  --sample-count 10000 \
  --protocol-id mnist_jointae32_fixed10k_v1 \
  --noise-path SAVED_NOISE_BANK.pt \
  --device cuda
```

For MNIST, `fid` resolves to `mnist_fd_jointae32_v1`. For a registered natural
image dataset, it resolves to `inception_fid`. The evaluator rejects the MNIST
encoder for natural images and rejects natural-image encoders for MNIST.

Natural-image evaluation uses the staged, hash-checked asset manifest. Inception
is the quality space. SSCD, DINOv2, and LPIPS are available for memorization:

```bash
mem-param-image-eval \
  --run-dir RUN_DIR \
  --checkpoints 100000,390000,695000 \
  --metrics fid,f_mem \
  --feature-spaces dataset_default,sscd,dinov2,lpips \
  --sample-count 10000 \
  --protocol-id natural_fixed10k_v1 \
  --noise-path SAVED_NOISE_BANK.pt \
  --assets-manifest outputs/model_assets/feature_f_mem/manifest.json \
  --device cuda
```

LPIPS remains a nearest-neighbor space. The protocol does not form its
5,760-dimensional covariance or report LPIPS FD.

The saved run must contain `config.json`, `run.json`, `data_bundle.pt`,
`data_manifest.json`, and the requested checkpoints. It must also provide enough
saved noise rows through `z_init.pt` or `--noise-path`. The evaluator never
rebuilds a data selection.

The protocol output has this form:

```text
evaluations/PROTOCOL_ID/
├── eval_manifest.json
├── reference/
│   ├── SPACE/training_features.pt
│   ├── SPACE/heldout_features.pt
│   ├── SPACE/calibration_neighbors.pt
│   ├── SPACE/validation_neighbors.pt
│   └── SPACE/calibration.json
├── checkpoints/step_XXXXXX/
│   ├── samples.pt
│   ├── sample_manifest.json
│   ├── features/SPACE.pt
│   ├── features/SPACE.COMPLETE.json
│   ├── nearest_neighbors/SPACE.pt
│   ├── fd.json
│   ├── metrics.json
│   └── COMPLETE.json
└── combined/
    ├── checkpoint_metrics.csv
    └── summary.json
```

Each cache records its source tensor hash, encoder hash, preprocessing, and
feature layer. A restart reuses only matching sample, feature, and neighbor
artifacts. A changed top-level protocol request requires a new protocol ID or
output directory.

## Full metric-suite smoke

The `full` metric preset runs every registered quality, memorization, held-out
loss, sample-diagnostic, and runtime branch for the saved dataset family. Use
the persistent evaluation-noise artifact when more rows are needed than the
run's monitoring-only `z_init.pt` file:

```bash
mem-param-image-eval \
  --run-dir RUN_DIR \
  --checkpoints all \
  --metrics full \
  --sample-count 512 \
  --loss-count 512 \
  --noise-artifact \
    outputs/reference_banks/evaluation_noise/canonical_gaussian_seed2003_n10000.pt \
  --reference-artifact \
    outputs/reference_banks/celeba_test_seed1_clean_n10000.pt \
  --assets-manifest outputs/model_assets/feature_f_mem/manifest.json \
  --output-dir outputs/smoke/full_metric_suite_512/RUN/unet32 \
  --write-plots \
  --device cuda
```

The canonical artifact contains `[10000, 1, 32, 32]` CPU Gaussian rows from
seed 2003. Its manifest records the generator state, PyTorch version, tensor
hash, and source-prefix hashes. The first 128 rows match the selected runs'
saved monitoring noise exactly. The evaluator verifies the artifact and takes
only the requested prefix. It does not modify `z_init.pt`. If
`--noise-artifact` is absent, the earlier per-run noise fallback remains
available.

`jobs/berzelius/full_metric_suite_smoke/` owns two development profiles. The
default profile evaluates every saved checkpoint with 512 samples. The
`final-replica10000` profile evaluates only step 1,000,000 with all 10,000
noise rows. Both write beneath `outputs/smoke/`, mark their manifests as
development-only, run the evaluator twice to prove cache reuse, and validate
all expected plots. Neither profile creates the final paper archive.

## Paper-tier full evaluation

Use `--publication-evaluation` only for a complete 10,000-sample full suite.
This mode requires `--metrics full`, `--sample-count 10000`, and
`--write-plots`. It writes `evaluation_tier=paper`, `development_only=false`,
and `publication_ready=true` into the evaluation manifest. The strict validator
must still write `ACCEPTED.json` before the result is accepted.

```bash
mem-param-image-eval \
  --run-dir RUN_DIR \
  --checkpoints all \
  --metrics full \
  --sample-count 10000 \
  --loss-count 512 \
  --noise-artifact \
    outputs/reference_banks/evaluation_noise/canonical_gaussian_seed2003_n10000.pt \
  --reference-artifact \
    outputs/reference_banks/celeba_test_seed1_clean_n10000.pt \
  --assets-manifest outputs/model_assets/feature_f_mem/manifest.json \
  --output-dir outputs/evals/STUDY/RUN \
  --write-plots \
  --publication-evaluation \
  --device cuda
```

`jobs/berzelius/celeba32_vv_weighting_paper_eval/` owns the frozen 18-task
launch for `N={512,2048}`, seeds `{0,1,2}`, and all three loss weights. Every
source run uses native `v/v` with `x_equivalent`, `unit`, or `eps_equivalent`
weighting and has 72 checkpoints. The launcher hashes every checkpoint and
shared input before submission. It keeps all generated banks and feature
caches until the paper figure contract is frozen.

### Seed-aggregated loss-weighting trajectories

After all 18 loss-weighting results pass strict acceptance, aggregate the exact
matrix across seeds. The command writes separate plot sets for both train
sizes:

```bash
/home/x_moalj/.conda/envs/mem_param_env/bin/python -m \
  mem_param_src.runners.weighting_trajectory_plots \
  --evaluation-root \
  outputs/evals/celeba32_vv_weighting_full_10000/BATCH
```

The command requires all three seeds for every weighting and train size. It
rejects an incomplete run, a changed checkpoint grid, a non-finite metric, or
an out-of-range `f_mem` value. Each panel contains `x_equivalent`, `unit`, and
`eps_equivalent`. The solid line is the arithmetic mean across seeds. The
shaded band is plus or minus one sample standard deviation. The plot applies
no smoothing and uses a symmetric-log step axis that includes step zero.

The primary overview uses calibrated `f_mem` at `alpha=0.01`. Supplementary
figures contain every calibrated alpha, the fixed raw-pixel rho values, quality
metrics, and held-out losses. The command writes PNG and PDF files, one exact
long-form aggregate CSV, and a manifest with all source and output hashes.

## Portable figure bundle

`mem_param_src.runners.figure_bundle` converts completed checkpoint gates to
one Mac-ready bundle. It reads saved metrics, sample banks, and top-two neighbor
files. It does not load a model, sample an image, encode an image, find a
neighbor, or calculate FID.

The exporter requires the official test reference by default. It rejects a
validation reference. The development-only `--allow-non-test-reference` flag
marks its output as not ready for publication.

The bundle contains:

- long CSV tables for run metadata, checkpoint metrics, calibration curves,
  distribution quantiles, and selected examples;
- NPZ files with `d1`, `d2`, `r`, `nn_idx`, and `second_nn_idx` for each
  checkpoint and metric space;
- shared calibration and validation neighbor arrays;
- exact float tensors for selected generated images and their two nearest
  training images;
- one PNG preview panel for each selected checkpoint;
- a JSON manifest, source hashes, row counts, and a SHA-256 list.

The frozen qualitative rule uses SSCD at alpha `0.01`. It selects the 16
smallest-r examples, the 8 unused examples nearest the calibrated boundary,
and 8 deterministic unused examples above the boundary. The exporter applies
this rule to each final checkpoint. Use `--example-key RUN_ID:STEP` to add a
disputed or planned figure checkpoint.

An ordinary non-figure checkpoint does not need its generated bank during
export. The exporter verifies its generated tensor hash against the saved
neighbor evidence. A final or selected checkpoint must retain its bank because
the exporter needs the exact image tensors.

Use one `--evaluation-root` argument for each completed checkpoint output. Run
the exporter on a CPU after all GPU evaluations are complete. This command
shows the required working directory:

```bash
cd /proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_param_clean
module load Mambaforge
conda run -n mem_param_env python -m mem_param_src.runners.figure_bundle \
  --evaluation-root outputs/interactive_checkpoint_metrics/RUN_A_STEP_1 \
  --evaluation-root outputs/interactive_checkpoint_metrics/RUN_A_STEP_2 \
  --evaluation-root outputs/interactive_checkpoint_metrics/RUN_B_STEP_1 \
  --output-dir outputs/figure_bundles/PAPER_STUDY_v1 \
  --study-id PAPER_STUDY_v1 \
  --example-key RUN_A:STEP
```

The command creates the output directory and a sibling
`PAPER_STUDY_v1.tar.gz`. It refuses to replace either output. Copy only the
archive to the Mac. Python, NumPy, and an image reader are enough to consume
the files. PyTorch and model code are not required.

## Minimal checkpoint notebooks

`notebooks/sanity/mnist_checkpoint_metrics.ipynb` selects one MNIST checkpoint,
generates eight samples, and reports JointAE FD, class-aware FD, calibrated
feature-space f_mem, raw sample diagnostics, and a sample grid.

`notebooks/sanity/celeba_checkpoint_metrics.ipynb` selects one CelebA checkpoint,
generates eight samples, and reports Inception FID, calibrated SSCD f_mem, raw
sample diagnostics, and a sample grid. It uses the verified schema-v3 encoder
manifest under `outputs/model_assets/feature_f_mem/`.

Both executed examples use the seed-zero, 512-example, x/x U-Net-S run at step
1,000,000. Change `RUN_DIR` and `CHECKPOINT_STEP` in the settings cell.
The eight-sample estimates are runtime checks, not paper-scale measurements.

`notebooks/sanity/checkpoint_sampling_runtime.ipynb` measures sampling without
metrics. It reports one-checkpoint time and projects the same 10,000-sample
workload across every saved checkpoint. The matching short Berzelius job is
under `jobs/berzelius/checkpoint_sampling_runtime/`.

`docs/unified_checkpoint_evaluation_plan.md` defines the production work unit,
metric matrix, held-out calibration, restart rules, and storage policy. The
selected CelebA32 gate uses one frozen, hash-checked 10,000-image official
test bank. Its builder rejects raw JPEG and transformed-tensor overlap with all
declared CelebA paper training banks. Other datasets and resolutions still need
equivalent frozen banks.

The completed paper-scale CelebA32 gate is under
`outputs/interactive_checkpoint_metrics/20261003T084819Z/`. It uses 10,000
generated images and the clean official-test bank. At step 1,000,000 for the
N=2048, seed-0, x/x U-Net-S run, it reports FID-10k 34.8159, raw-pixel f_mem
0.5197 at fixed rho 0.3, calibrated raw-pixel f_mem 0.9450 at alpha 0.01, and
calibrated SSCD f_mem 0.6235 at alpha 0.01. See
`docs/unified_checkpoint_evaluation_plan.md` for the measured runtime and
scope.

Natural-image protocol mode computes standard Inception FID and calibrated
SSCD, LPIPS, or DINOv2 nearest-neighbor memorization from the same generated
bank. The earlier `jobs/berzelius/eps_recovery_1m/feature_f_mem/` workflow
remains the frozen study-specific launcher and supplies the compatible staged
asset manifest.

## Reproducibility notebook

`notebooks/sanity/mnist_jointae_fd_sanity.ipynb` trains the final architecture
from scratch through the standalone trainer. It restores the qualified
validation checkpoint and shows reconstruction, classification, clean FD,
noise, translation, rotation, class collapse, and exact-copy memorization
controls. The notebook does not own runtime metric code.
