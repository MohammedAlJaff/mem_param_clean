# Epsilon generation: cluster code comparison and recovery checks

Date: 2026-09-09. Status: diagnosis open; historical artifacts recovered.

The strongest next check is to replay the successful historical weights through
both implementations under the same sampling policy. This can separate a code
regression from changed sampling and training without a new training run.

This report follows [the critical audit](../CRITICAL_EPS_EPS_SAMPLING_AUDIT_2026-09-09.txt)
and [the Flowers failure record](../protoypinging_and_trubbleshoting/eps_eps_sampling/OPEN_PROBLEM.md).
Those notes were written on the Mac. This review inspected both cluster folders.
No research training or trained-checkpoint inference ran during this review.

## New evidence from the cluster

The successful historical N=256, seed=0 checkpoint and its original config exist:

```text
../old_mem_par/outputs/job_sets/mem_gen_boundary_20260427/runs/
mem_gen_boundary_20260427_celeba32_N256_B256_eps_eps_steps1000000_s0-bd48e9dab3/
seeds/seed_0000/attempts/DONE_20260427_020458/
```

The lines above form one path. It contains:

- `checkpoints/step_025000.pt`: step, model, and optimizer state.
- `artifacts/config.json`: the actual saved historical settings.
- `artifacts/train_tensor.pt`: [256, 1, 32, 32], float32, finite, range [-1,1].
- `artifacts/reference_tensor.pt`: [1024, 1, 32, 32], float32, finite, range [-1,1].
- `artifacts/z_init.pt`: [1024, 1, 32, 32], finite float32 Gaussian noise.
- `figures/generation_monitor_samples_step_000750.png` and `_025000.png`.

All 228 model-state tensors loaded strictly into the clean GDFM after copying
the saved historical network/data settings. That model has 3,917,409 parameters.
This checks state compatibility, not forward-output or generation parity.

The stored step-750 grid shows mostly saturation and speckled noise, with one
clearer face. The step-25,000 grid shows eight recognizable faces. Both grids
use the same saved noise. The early historical failure resembles the recent
symptom; useful samples emerged later under the historical protocol.

The original CSV confirms:

| Update | FID on 10,000 generated samples |
| ---: | ---: |
| 750 | 257.793 |
| 1,250 | 230.806 |
| 10,000 | 172.636 |
| 15,000 | 51.061 |
| 20,000 | 24.211 |
| 25,000 | 19.622 |
| 1,000,000 | 27.551 |

Source: `../old_mem_par/outputs/evals/mem_boundary_20260427_all_done_quality_20260501/all_checkpoint_metrics.csv`.
The monitoring bank has 1,024 samples; the post-hoc quality evaluation used
10,000. A small replay does not reproduce the published CSV's FID protocol.

The clean `outputs/` directory is absent from this cluster checkout. Its recent
N=32 MPS checkpoint and samples remain unavailable here. Their loss and range
statistics below are from the critical audit, not fresh artifact inspection.

## Actual protocol differences

| Factor | Historical CelebA32 | Recent clean CelebA32 | Failed clean Flowers |
| --- | --- | --- | --- |
| Data shape | 1x32x32 | 1x32x32 | 3x64x64 |
| Training images / batch | 256 / 256 | 32 / 32 | 8 / 8 |
| Seed | 0 | 7 | 7 |
| Model parameters | 3,917,409 | 9,393,985 | 669,251 |
| UNet widths | 32/64/128 | 64/128/192 | Smaller diagnostic UNet |
| Time embedding | 128, derived from `None` | 64, inherited default | See saved experiment config |
| Attention | 16px encoder plus 8px middle | 8px middle | Different diagnostic profile |
| Dropout | 0.1 | 0 | 0 |
| Adam LR | 1e-4 | 1e-3 | 1e-3 |
| Training updates | Best at 25k; continued to 1M | 1k | 10k |
| Training times | Uniform [0.001,0.999] | Uniform [0.01,0.99] | Uniform [0.01,0.99] |
| Sampling | Heun50, [0.001,0.999] | Heun50, [0.01,0.99] | Heun50, [0.01,0.99] |
| Sampling division | Floor 0.05 | Exact | Exact |
| Returned output | Integrated state `x_t_max` | Endpoint estimate `x_1` | Endpoint estimate `x_1` |
| Runtime | CUDA, TF32 requested | MPS, FP32 | MPS, FP32 |

Historical optimizer state confirms Adam betas (0.9,0.999), epsilon 1e-8,
weight decay 0, and LR 1e-4. The clean optimizer uses the same Adam construction;
the selected LR differs. Neither inspected update loop adds EMA, a learning-rate
scheduler, or gradient clipping. Their removal is not an explanation supported
by these paths.

## Ranked causes and discriminating checks

### 1. Too little training under a changed optimization setup

This has direct historical support. Even the successful old run still had FID
172.6 at 10k. Failure at 1k does not establish that clean CelebA cannot work.
The higher LR, smaller full batch, changed network, and different seed prevent
extrapolating the exact recovery step to the clean run.

Check: first recover generation from old weights in clean code. Then reproduce
the old training settings and exact saved subset, with checkpoints at 1k, 5k,
10k, 15k, 20k, 25k, and 30k. Do not jump directly to a million-step sweep.
Compare LR 1e-4 versus 1e-3 only after establishing that reference condition.
Keep parameterization and unit epsilon MSE fixed.

### 2. Removing the historical early-time denominator floor

For `z_t=t*x+(1-t)*eps`, exact conversion is:

```text
v_hat = (z_t - eps_hat) / t
```

At fixed state, prediction error is amplified by 1/t. At t=0.01 the factor is
100. The historical and clean safe code instead reconstruct x first:

```text
x_hat = (z_t - (1-t)*eps_hat) / max(t, c)
v_hat = x_hat - eps_hat
```

This is NOT `(z_t-eps_hat)/max(t,c)`. At t=0.01, c=0.05, epsilon-error sensitivity
is 20.8. At the old start t=0.001 it is 20.98, despite the smaller start time.
The floor changes the field below c; it is more than protection against NaNs.
For a perfect paired epsilon target there, velocity becomes `(t/c)*x-eps`.

Check: same checkpoint, noise, device, batch, endpoints, output, and step count;
change only `sampling.use_safe_div` and `sampling.safe_div_denominator_floor`.
Both `True` and `0.05` must be set. Changing the floor while division is disabled
has no effect. Floor 0.01 is effectively inactive over [0.01,0.99].

Use the successful old checkpoint as well as the current one when available:
if old weights work with the historical floor but fail with exact division,
the successful historical result itself depended on that sampling policy.
If current weights improve under the floor, this shows a sampling effect for
that checkpoint; it does not prove all training differences are irrelevant.

### 3. Time conditioning, attention, and architecture settings

The computational SmallUNet implementation is preserved. The configured model
is different. Increasing widths does not recreate the historical time embedding
or attention locations. The clean CelebA config inherits time embedding 64;
old `time_emb_dim=null` resolves to `4*32=128`.

Check: construct from the entire saved `network` section, not just base width.
Include `channel_mult=(1,2,4)`, two residual blocks, dropout 0.1, and attention
resolutions (16,8). `attn_resolutions=()` does not disable middle attention.
The attention-resolution set is not encoded as learned weights, so strict
state loading alone cannot detect a wrong attention setting.

The sinusoidal time embedding consumes raw t in [0,1] in both versions. No
new time rescaling or time-direction reversal was found. Treat stronger time
features or a residual epsilon head as later experiments, not missing repairs.

### 4. Start/end times and the approximate initial distribution

Training and sampling now have separate controls. Old `schedule.t_min/t_max`
set both. Sampling outside the clean training interval is an extrapolation.
Both inspected historical and clean paths start from raw Gaussian noise at a
positive time; that only approximates the true interpolated marginal.

Check: reproduce the historical bounds first with historical weights. Then
change bounds independently from the floor and output. Do not start exact
epsilon sampling at t=0. A scaled-noise start removes only part of the mismatch
and changes the protocol; it was not active in the saved April configuration.

### 5. Dataset, dimensionality, and sample coverage

Flowers RGB64 has 12 times as many scalar coordinates as CelebA grayscale32.
Its diagnostic model is also much smaller and saw only eight images. The same
per-coordinate error gives a larger total L2 error at higher dimension.
However, current CelebA and MNIST failures mean dimension alone cannot explain
every recorded failure.

Check after CelebA recovery: same selected Flowers identities at RGB64, RGB32,
and grayscale32; then CelebA gray32 versus Flowers gray32. Hold the training
and sampling protocol fixed and record parameter counts. Compare per-coordinate
RMSE or L2/sqrt(D), not raw cross-resolution distance. Distinguish total noisy
training pairs from visits per image when changing N and batch size.

### 6. Solver resolution or accumulating off-path error

The Flowers 50/200/500-step results converge to essentially the same failed
samples. This weakens the claim that more steps alone repair that checkpoint.
It does not settle the current CelebA checkpoint. A solver can accurately
integrate an inaccurate learned field.

Check: 50 versus 200 steps with everything else fixed; inspect the first solver
stages and the full trajectory. Log RMS of state, predicted epsilon, velocity,
and the update `dt*v`. Compare the Euler trial state and corrected Heun state.
Locate whether errors appear near the start, gradually, or only in final decode.

### 7. Endpoint output convention

Old sampling returns the state at t_max. Clean `x_1` performs another network
prediction and converts it to x. It does not integrate to 1.
For exact epsilon conversion, `x_1-z = ((1-t)/t)*(z-eps_hat)`. At t=0.99 that
coefficient is about 0.0101, making this a lower-priority candidate unless the
late prediction is already large. It cannot explain state blow-up observed
before the endpoint.

Check: save the integrated state and its final decode from the same trajectory.
Compare `x_t_max` and `x_1` without changing anything else.

### 8. Device, precision, checkpoint/config mismatch, or mode handling

CUDA/TF32 and MPS/FP32 differ. Epsilon conversion subtracts similar quantities
near the noise endpoint, so precision should be controlled before attribution.
Current previews use the trained in-memory model, which makes a checkpoint
reload mismatch unlikely as the cause of the observed preview failure.
Both monitors enter eval mode and restore the previous training mode.

Check: establish both old/clean controls on one CUDA device in FP32 with TF32
off. Compare network predictions first, then converted velocities, then solver
states. Use `eval()` and no autocast. Test TF32 separately afterward.
Read prediction/loss spaces from the saved config; string metadata is not in the
weights. The clean model needs the new training fields, so do not pass the old
JSON to it unchanged. Sampling overrides must be applied to the sampling
section; mutating a config after model construction does not update stored
model attributes.

### 9. Metrics and display can misclassify failure

The critical note records 59.3% of current CelebA pixels outside [-1,1], despite
loss 0.0144 and projected f_mem=0. The projection metric clips/normalizes pixels;
it loses raw magnitude information. Historical canonical rho=0.5 differs from
the clean rho=0.3. Pixel, projected-pixel, and feature-space distances differ.

Check: retain raw tensors, fixed-range grids, out-of-range fraction, and quality
beside f_mem. Compare the same metric space and threshold. Avoid autoscaling
each image, which can conceal amplitude failure. Low f_mem on noise is not
successful generation.

## Checks already completed

Environment: `mem_param_env`, Python 3.10.20, PyTorch 2.5.1.post303, CPU.
Clean revision: `7606bed8197509833b3f814f2af6c028cb349a3e` before this report.
Old revision: `9c21d19b07a169e809d7c1664e1e517732ebe754`, with pre-existing local
changes including model and trainer files. These were preserved. Historical
saved config is authoritative; exact April executable provenance is not fully
established by today's working tree.

- `python -B tests/test_model_sampling.py`: 13 tests passed in 0.147 seconds.
  Initial pytest invocation could not run because pytest is not installed;
  the test module's existing unittest entrypoint ran the tests successfully.
- Old/clean analytical epsilon-field Euler and Heun comparisons: exact numerical
  equality for three interval/floor combinations, six comparisons total.
- Direct epsilon MSE with identical x, noise, time, and analytical predictions:
  exact equality between old and clean code.
- Zero-output epsilon test: clean Heun with exact division and x_1 returned
  `z_init/0.01`, maximum absolute FP32 error 1.90735e-5.
- Historical checkpoint strict load into the matching clean architecture passed.
- Saved historical tensors, optimizer settings, quality CSV, and two existing
  sample grids were inspected. No datasets were rebuilt.

These checks rule out specific basic incompatibilities. They do not replace
trained-model forward parity or full image-generation replay.

## Most informative replay order

| Case | Weights | Implementation | Sampling | Interpretation |
| --- | --- | --- | --- | --- |
| A | Old 25k | Old | Historical floor/bounds/state | Available reference |
| B | Same | Clean | Identical to A | A/B mismatch points to code/config/runtime |
| C | Same | Clean | B with exact division only | B/C difference isolates the floor |
| D | Same | Clean | Current exact/bounds/x_1 bundle | Tests current policy with successful weights |
| E | Same | Clean | D with floor 0.05 only | D/E isolates floor at current bounds |

If A and B match and generate faces, useful epsilon generation has been recovered
in clean code for the historical checkpoint. If D also works, investigate the
recent training settings and training duration. If B works but D fails, isolate
sampling changes before retraining. If A fails, inspect artifact selection and
runtime/provenance before blaming migration. Small grids establish a diagnostic
result; use more samples and repeated seeds for research claims.

## User-launched GPU replay example

Run from the clean repository in an allocated CUDA session. This example is
syntax-checked but has not been executed. It runs inference on existing weights,
creates one fresh results folder, and does not train or compute FID/f_mem.

```bash
module load Mambaforge
conda activate mem_param_env
PYTHONPATH=. python -u -B - <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from ml_collections import ConfigDict
from torchvision.utils import save_image

from mem_param_src.config import get_base_config
from mem_param_src.model import GDFM as CleanGDFM

root = Path.cwd()
old_root = root.parent / "old_mem_par"
sys.path.insert(0, str(old_root / "src"))
from gdfm.core import GDFM as OldGDFM

assert torch.cuda.is_available(), "Use an allocated CUDA GPU."
device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

run_id = "mem_gen_boundary_20260427_celeba32_N256_B256_eps_eps_steps1000000_s0-bd48e9dab3"
attempt = (old_root / "outputs/job_sets/mem_gen_boundary_20260427/runs"
           / run_id / "seeds/seed_0000/attempts/DONE_20260427_020458")
old_config = ConfigDict(json.loads((attempt / "artifacts/config.json").read_text()))
checkpoint = torch.load(attempt / "checkpoints/step_025000.pt",
                        map_location="cpu", weights_only=True)
noise = torch.load(attempt / "artifacts/z_init.pt", map_location="cpu",
                   weights_only=True)[:16].to(device)

config = get_base_config()
config.framework.prediction_space = config.framework.loss_space = "eps"
config.framework.loss_weighting = "unit"
config.network = ConfigDict(old_config.network.to_dict())
config.data = ConfigDict(old_config.data.to_dict())
config.training.start_time = old_config.schedule.t_min
config.training.end_time = old_config.schedule.t_max
config.training.use_safe_div = True
config.training.safe_div_denominator_floor = old_config.schedule.denom_eps

old_model = OldGDFM(old_config).to(device).eval()
clean_model = CleanGDFM(config).to(device).eval()
for model in (old_model, clean_model):
    model.load_state_dict(checkpoint["model"], strict=True)
    assert model.prediction_space == model.loss_space == "eps"

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
out = root / "outputs/session_runs" / f"historical_eps_replay_{stamp}"
out.mkdir(parents=True, exist_ok=False)
records = {}

def record(name, samples):
    samples = samples.detach().cpu()
    torch.save(samples, out / f"{name}.pt")
    finite = bool(torch.isfinite(samples).all())
    records[name] = dict(finite=finite, minimum=samples.min().item(),
                         maximum=samples.max().item(),
                         rms=samples.square().mean().sqrt().item(),
                         outside_fraction=(samples.abs() > 1).float().mean().item())
    if finite:
        save_image(samples, out / f"{name}.png", nrow=8,
                   normalize=True, value_range=(-1, 1))
    print(name, records[name], flush=True)
    return samples

base = dict(n_steps=50, method="heun", start_time=.001, end_time=.999,
            use_safe_div=True, safe_div_denominator_floor=.05, output="x_t_max",
            eps_to_v="via_x", heun_final_euler=False)
variants = {
    "B_clean_historical": dict(base),
    "C_clean_historical_exact": dict(base, use_safe_div=False),
    "D_clean_current": dict(base, start_time=.01, end_time=.99,
                            use_safe_div=False, output="x_1"),
    "E_clean_current_floor005": dict(base, start_time=.01, end_time=.99,
                                     output="x_1"),
}
with torch.no_grad():
    # Compare predictions before comparing trajectories.
    for value in (.001, .01, .5, .999):
        times = torch.full((len(noise),), value, device=device)
        old_pred = old_model.predict(noise, times)
        clean_pred = clean_model.predict(noise, times)
        torch.testing.assert_close(old_pred, clean_pred, rtol=1e-5, atol=1e-5)
    a = record("A_old_historical", old_model.sample(
        noise.clone(), n_steps=50, method="heun", device=device,
        t_min=.001, t_max=.999, protocol="default", start_mode="raw_noise",
        final_decode=False))
    for name, settings in variants.items():
        result = record(name, clean_model.sample(noise.clone(), **settings, device=device))
        if name == "B_clean_historical":
            records[name]["versus_old_rmse"] = (result-a).square().mean().sqrt().item()
            records[name]["versus_old_max_abs"] = (result-a).abs().max().item()
            print("A/B parity", records[name], flush=True)

(out / "report.json").write_text(json.dumps({
    "historical_attempt": str(attempt), "checkpoint_step": checkpoint["step"],
    "old_config": old_config.to_dict(), "clean_config": config.to_dict(),
    "variants": variants, "noise_prefix_count": len(noise),
    "torch_version": str(torch.__version__), "device": torch.cuda.get_device_name(device),
    "tf32": False, "results": records,
}, indent=2))
print(f"Saved: {out}", flush=True)
PY
```

Expected files: five raw `.pt` batches, five PNG grids if samples are finite,
and `report.json`. Inspect A/B numerical agreement and the grids before deciding
on further compute. This uses old source only in a diagnostic comparison; the
clean library remains independent of the old repository.

## If learned-field error remains: measure the right quantities

At fixed times (0.001 only for models trained there, then 0.01, 0.02, 0.05,
0.1, 0.5, 0.9, 0.99), construct exact noisy path states with saved training
images and fixed noise. Record epsilon MSE and its implied velocity error.
Compare against the simple predictor `eps_hat=z_t`; this exposes whether a low
reported epsilon loss beats an almost-identity baseline near zero.

Paired epsilon/velocity error includes conditional target variance. For these
small finite training sets, an additional reference field is available directly:

```text
p_i(z,t) = softmax_i(-||z-t*x_i||^2 / (2*(1-t)^2))
x_bar(z,t) = sum_i p_i(z,t)*x_i
v_empirical(z,t) = (x_bar(z,t)-z)/(1-t)
```

This is the exact conditional field for the uniform empirical distribution of
the saved training images, with unit Gaussian noise and the stated linear path.
Compare the learned field against it at supervised states and sampled states.
It removes paired-target variance from that comparison. It is a memorizing
empirical reference, not a measure of generalization or a proposed replacement
training target. The formula is derived from the stated Gaussian interpolation;
this diagnostic was not executed.

Changing to eps/v, inverse-time weighting, a residual epsilon head, or a different
time distribution may improve learning. Those are separate interventions. First
recover the original direct unit-MSE eps/eps condition and identify why the
new protocol differs.
