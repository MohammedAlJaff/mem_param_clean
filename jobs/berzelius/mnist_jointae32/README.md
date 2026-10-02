# MNIST JointAE-32 metric encoder

This job trains the shared 128-dimensional MNIST encoder proposed in
`notebooks/sanity/mnist_jointae_fd_sanity.ipynb`.

The official 60,000-image MNIST training split is deterministically shuffled
with seed 2003. The first 55,000 indices train the encoder and the remaining
5,000 choose the epoch. The official 10,000-image test split never changes
model weights or selects an epoch; it becomes the frozen real-reference bank.

The encoder feeds both a reconstruction decoder and a ten-class classifier.
The job selects the lowest reconstruction-loss epoch among epochs reaching
99% validation accuracy. If no epoch reaches that threshold, it preserves the
candidate artifacts, writes `FAILED_QUALIFICATION`, and exits nonzero.

Preview without writing or submitting:

```bash
bash jobs/berzelius/mnist_jointae32/submit.sh --dry-run
```

The researcher submits with:

```bash
bash jobs/berzelius/mnist_jointae32/submit.sh
```

Defaults are one thin A100-40GB, four CPUs, 32 GB host memory, fifteen epochs,
and one hour. Override the account, walltime, or epoch count with
`MP_MNIST_ENCODER_ACCOUNT`, `MP_MNIST_ENCODER_TIME`, and
`MP_MNIST_ENCODER_EPOCHS`.

Each frozen batch is written below `outputs/job_sets/mnist_jointae32/`. Its
`asset/` directory contains:

- `mnist_jointae32_v1.pt`: weights and fixed feature normalization;
- `mnist_test_reference_features.pt`: all test features, labels, logits, mean,
  and covariance;
- `split_indices.pt`: exact train, validation, and test indices;
- `history.json` and `manifest.json`: training and provenance records;
- `TRAINING_COMPLETE` only after qualification succeeds.

The intended metric names are `mnist_fd_jointae32_v1`,
`mnist_cafd_jointae32_v1`, and `f_mem_mnist_jointae32_v1`.
