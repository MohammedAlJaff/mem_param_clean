# CelebA32 prediction/loss 3x3

This study holds gray CelebA 32x32, N=256, seed 0, optimizer, UNet, and time
range fixed while crossing prediction space `{x, eps, v}` with loss space
`{x, eps, v}`. The nine rows are in `variants.tsv`. Off-diagonal target
conversion uses literal division during training (`training.use_safe_div=false`).

Every task uses physical and effective batch 256 on a thin A100, deterministic
CUDA kernels, 50-step full Heun monitoring, direct epsilon conversion, and a
safe sampling denominator floor of 0.05. The configured target is one million
optimizer updates, but the first allocation stops after the scientific 200k
checkpoint for a joint inspection. All nine tasks run without an array throttle.

Submit the first stage with:

```bash
bash jobs/berzelius/celeba32_prediction_loss_3x3/submit.sh
```

`submit.sh` freezes `mem_param_src`, the config, and the matrix inside the batch
directory. Set `MP_PRED_LOSS_DEPENDENCY=afterok:JOB_ID` to make a submission
eligible only after a validation job succeeds.

## Recovery and continuation

The trainer rotates two complete recovery states every 5,000 updates and at
checkpoints, signals, and planned stops. A state includes the model, Adam,
Python/NumPy/Torch CPU and CUDA RNGs, exact shuffled-batch cursor, history,
elapsed time, and W&B run ID. Slurm sends USR1 five minutes before the limit;
the process saves at the next optimizer boundary. Runs use `--no-requeue`, so a
researcher audits the artifact before resubmitting selected rows.

Recover interrupted rows to the 200k gate:

```bash
bash jobs/berzelius/celeba32_prediction_loss_3x3/resume.sh BATCH_DIR ARRAY_SPEC
```

After accepting the joint 200k review, continue selected rows to one million:

```bash
bash jobs/berzelius/celeba32_prediction_loss_3x3/resume.sh BATCH_DIR ARRAY_SPEC 1000000
```

Resume loads the frozen saved config and source snapshot and reuses the same
W&B run ID. Do not resume a row that is still running.

## 200k evaluation gate

Use 10,000 fixed-noise samples and the frozen 10,000-image CelebA validation
reference. Evaluate prediction `eps` rows with direct and legacy `via_x`
conversion, each under safe floor 0.05 and exact division. For prediction `x`
and `v`, conversion is irrelevant, so evaluate safe and exact division only.
All profiles use 50-step full Heun over `[0.001, 0.999]`.

Report FID-10k, full-pixel f_mem, and the frozen SSCD, LPIPS, and DINOv2
feature-space f_mem contracts. Compare the complete grid before approving any
continuation; raw training losses from different loss spaces are not directly
comparable.
