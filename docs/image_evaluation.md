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
