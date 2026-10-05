# EPS/EPS data and image-format study

This study asks whether the useful CelebA32 EPS/EPS setting survives changes in
dataset, resolution, color, and capacity. The seven seed-0 rows are:

| Row | Variant | Intended comparison |
| --- | --- | --- |
| 0 | CelebA 32x32 gray | Fresh batch-rule control |
| 1 | MNIST 32x32 gray | Dataset change |
| 2 | CelebA 32x32 RGB | Color change |
| 3 | CelebA 64x64 gray | Resolution change |
| 4 | CelebA 64x64 RGB | Resolution plus color |
| 5 | Flowers 64x64 RGB | Dataset plus format change |
| 6 | Flowers 64x64 RGB, deeper UNet | Capacity probe |

All rows use N=256, seed 0, EPS prediction and EPS loss, Adam at 1e-4, literal
training division, deterministic CUDA kernels, and the same one-million-update
checkpoint schedule. Physical batch is 128 with two gradient-accumulation
microbatches, giving effective batch 256 on fat A100s. This common rule fits the
64x64 RGB and deeper model while preserving the effective batch of the old
CelebA32 result. Row 0 is necessary because its physical batch differs from the
old batch-256 anchor.

The first allocation stops at the scientific 200k checkpoint. Submit all seven
unthrottled tasks with:

```bash
bash jobs/berzelius/eps_eps_data_image_format/submit.sh
```

Set `MP_DATA_FORMAT_DEPENDENCY=afterok:JOB_ID` to guard the array behind a
validation job. The submitter freezes the source, config, and matrix inside the
new batch directory.

## Recovery and continuation

Complete recovery states rotate every 5,000 updates and include model, Adam,
all relevant RNG state, shuffled-batch cursor, history, elapsed time, and W&B
identity. Jobs request an early USR1 signal and disable automatic requeue.

Recover selected interrupted rows to 200k:

```bash
bash jobs/berzelius/eps_eps_data_image_format/resume.sh BATCH_DIR ARRAY_SPEC
```

After the 200k review, continue selected rows to one million:

```bash
bash jobs/berzelius/eps_eps_data_image_format/resume.sh BATCH_DIR ARRAY_SPEC 1000000
```

Resume uses the saved config and same W&B run. Audit the newest recovery state
and confirm that the row is not active before submitting it.

## 200k evaluation gate

For every row, replay the same fixed 10,000-noise bank with 50-step full Heun
over `[0.001, 0.999]`. Cross direct and legacy `via_x` epsilon conversion with
safe floor 0.05 and exact division. Use 10,000 held-out images from the matching
dataset for FID wherever the split permits, and calibrate each feature-space
f_mem contract against that dataset's frozen held-out set.

Report FID, full-pixel f_mem, SSCD, LPIPS, and DINOv2 f_mem. Treat the Flowers
deep row as a capacity diagnostic, not as another data-only comparison. Make
the continuation decision jointly across quality, memorization, stability, and
runtime rather than from training loss alone.
