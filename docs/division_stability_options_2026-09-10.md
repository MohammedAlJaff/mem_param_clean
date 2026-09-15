# Separate division controls and numerical stability options

Date: 2026-09-10. Status: analysis and proposals; no implementation changes.

## Confirmed API and ownership

Training and sampling division controls are separate in the clean repository:

| Purpose | Enable flag | Denominator floor |
| --- | --- | --- |
| Training conversion | `training.use_safe_div` | `training.safe_div_denominator_floor` |
| Sampling conversion | `sampling.use_safe_div` | `sampling.safe_div_denominator_floor` |

The external `sampling.sample`, `sample_euler`, and `sample_heun` functions accept
`use_safe_div` and `safe_div_denominator_floor` explicitly. `GDFM.sample` forwards
both unchanged. Each Heun stage uses that sampling policy. Optional final x
conversion uses the sampling policy too. `GDFM.compute_loss` reads the training
policy stored when the model was constructed. The old implementation shared
`schedule.denom_eps` across conversion calls.

An API example for an already constructed model and noise bank:

```python
from mem_param_src.sampling import sample

model.eval()
samples = sample(
    model, z_init, n_steps=50, method="heun",
    start_time=0.01, end_time=0.99,
    use_safe_div=True, safe_div_denominator_floor=0.05,
    output="x_t_max", device=z_init.device,
)
```

This example was not executed on a trained model. It returns a tensor with the
same shape as z_init. The sampler handles no-grad; the caller sets eval mode.

Image-training CLI controls are `--sampling-safe-div`,
`--no-sampling-safe-div`, and `--sampling-safe-div-denominator-floor 0.05`.
Training controls can be set through the existing config or `--set` overrides.
There is no implemented selector for smooth floors, velocity damping, logit-time
integration, or other alternative clamp algorithms. The implemented policy is
sign-preserving division by `max(abs(denominator), floor)`.

Base defaults enable each flag with floor 0.01 and bounds [0.01,0.99]. That floor
is effectively inactive for the linear FM coefficients in that interval. The
recent CelebA/MNIST configs explicitly disable safe division. Changing a floor
without enabling its corresponding flag has no effect.

Sources: `mem_param_src/model.py`, `sampling.py`, `config.py`, and
`runners/image_train.py`. Previous recovery evidence is in
[the cluster audit](eps_eps_recovery_audit_2026-09-09.md).

## What actually becomes singular

All equations below use the repository convention:

```text
z_t = t*x + (1-t)*eps
v = x-eps
noise at t=0, data at t=1
```

| Conversion | Potentially small denominator |
| --- | --- |
| eps to x or v | t, near the noise endpoint |
| x to eps or v | 1-t, near the data endpoint |
| v to x or eps | None for the linear FM path; determinant is 1 |
| Diagonal eps/eps, x/x, v/v loss | No target conversion division |

Direct unit-MSE eps/eps training has no division to stabilize. Its sampling
conversion does. Cross-space eps/v training also has the sensitive conversion.
The configured positive bounds already avoid literal zero. The remaining
problems are error amplification, cancellation, solver error, and inaccurate
predictions along the generated trajectory.

At a fixed state and time, exact epsilon velocity obeys:

```text
v_hat = (z-eps_hat)/t
velocity error = -epsilon prediction error/t
```

No algebraic rewrite can recover an accurate velocity from an arbitrary epsilon
prediction with fixed absolute error as t approaches zero. A method must reduce
roundoff/discretization error, improve prediction accuracy, enforce the needed
cancellation, or deliberately change the field.

## Existing checkpoints: options affecting sampling

### A. Establish the historical floor as a reference

The existing implementation reconstructs x first:

```text
d = max(t, c)
x_hat = (z-(1-t)*eps_hat)/d
v_hat = x_hat-eps_hat
```

This differs from direct velocity damping `(z-eps_hat)/max(t,c)`. The current
floor changes the field below c even for the true paired epsilon. At t=0.01,
c=0.05, its epsilon-error sensitivity is 20.8 versus 100 for exact division.
Direct velocity damping would give sensitivity 20 and scale the entire exact
field by t/d. These should be separate named experimental policies.

Use c=0.05 as a historical control, not a proven universal setting. Test the
same weights/noise/device/bounds/output before attributing an improvement.
A floor can suppress magnitude errors without restoring image structure.

### B. Integrate in logit time: same continuous ODE, different discretization

Define:

```text
s = log(t/(1-t))
t = sigmoid(s)
dt/ds = t*(1-t)
```

Applying the chain rule gives the exact fields:

```text
eps prediction: dz/ds = (1-t)*(z-eps_hat)
x prediction:   dz/ds = t*(x_hat-z)
v prediction:   dz/ds = t*(1-t)*v_hat
```

These avoid explicit division by the small endpoint coefficients. The trained
network still receives physical t, not s. Integrate from logit(start_time) to
logit(end_time). Each Heun stage must evaluate the network at its own physical
time and trial state. Compute the transformed field directly; computing an
infinite velocity first and multiplying by zero is not a stable implementation.

This preserves the exact continuous trajectory on the chosen open interval,
uses existing weights, and changes the numerical grid and finite-step method.
With 50 intervals on [0.01,0.99], the first physical step is about 0.001994,
versus 0.0196 on the current uniform-t grid. Central physical steps become
larger. Compare at the same network-evaluation budget and check convergence.

Uniform logit nodes with Heun in t and Heun performed in s are distinct finite
step methods. Both deserve clear naming if implemented. Do not confuse either
with changing the training-time distribution.

Limit: this removes an explicit division, not the sensitivity of the original
learned flow. For eps_hat=0, the exact endpoint sample is still z_init/start_time.
The scalar logit-Heun check converged toward 100*z_init at start_time=0.01.
A wrong field can still generate bad samples in either coordinate system.

This transformation is derived here for linear FM. It follows the broader
practice of changing time variables and exploiting ODE structure discussed in
[DPM-Solver](https://arxiv.org/abs/2206.00927). It is not a claim that an existing
VP diffusion scheduler can be used unchanged for this repository's convention.

### C. Improve arithmetic and step control

- For exact FM epsilon conversion, evaluate `(z-eps_hat)/t` directly instead of
  reconstructing x and then subtracting eps. For x, use `(x_hat-z)/(1-t)`.
  This removes intermediate operations; it does not remove cancellation.
- Keep model evaluation, subtraction, and state accumulation in FP32 for the
  initial control. Test FP64 arithmetic or a complete FP64 forward separately
  if necessary. Casting an already inaccurate network output to FP64 cannot
  recover its lost precision. Control CUDA TF32 and autocast explicitly.
- Use smaller endpoint steps or adaptive step rejection based on a local error
  estimate, scaled per coordinate. Bound maximum work and retain accepted times.
  A rejected step should be retried, rather than clipping its update silently.
- Consider higher-order methods that integrate the known linear term
  analytically. DPM-Solver provides that principle, but adaptation needs the
  current path and time direction. A first-order frozen-epsilon update here is
  exactly Euler in t: `z_next=z+(t_next-t)/t*(z-eps_hat)`. Renaming it as an
  exponential method gives no improvement at the same nodes.

The Flowers replay already showed convergence to poor samples at 50/200/500
steps. Solver improvements must be checked on that learned field, not assumed
to solve it. There is no current evidence that roundoff dominates its error.

### D. Smooth reciprocal regularization and denoiser projection

A smooth floor such as `sqrt(t*t+c*c)` avoids the hard-floor kink, but modifies
the denominator even above c. The regularized reciprocal `t/(t*t+c*c)` is
bounded and tends to zero near t=0; it damps the field and introduces bias.
Neither is algebraically equivalent to the original flow. Smoothness alone is
not enough to prefer these over the reproducible historical hard floor.

For pixel data supported in [-1,1], project the estimated clean image x_hat into
that box and derive a consistent update from it. The exact conditional mean of
bounded pixels also lies in that box. Projection cannot increase squared error
to that mean, but it changes the approximate field and can saturate an inaccurate
estimate. Avoid extending pixel bounds to arbitrary latent representations.
The [official DPM-Solver implementation](https://github.com/LuChengTHU/dpm-solver)
also distinguishes pixel-space thresholding from latent-space sampling.

A first-order data-estimate update can be expressed as:

```text
alpha = (t_next-t)/(1-t)
z_next = (1-alpha)*z + alpha*x_hat_projected
```

For t <= t_next <= 1 and t<1, alpha lies in [0,1]. This expresses the Euler
update as interpolation and avoids constructing a large intermediate velocity.
It is an alternative sampler; adding ordinary Heun afterward does not inherit
this convex-combination guarantee. The initial Gaussian state may legitimately
be outside the pixel box. Clipping the noisy state itself is a different,
usually less justified intervention.

## Training: objective-preserving choices and objective changes

### E. Improve time coverage while preserving epsilon MSE

Use stratified times or more endpoint samples. If the original time density is
p(t) and sampling uses q(t), weighting each example by p(t)/q(t) preserves the
original expected loss. Without correction, changing time sampling changes the
objective. More low-t examples may improve estimation there but does not remove
the small low-t gradients inherent in epsilon loss for velocity-sized errors.

For uniform s=logit(t) on a finite interval, q(t)=1/(S*t*(1-t)), where S is the
logit interval length. To preserve uniform-t training on [a,b], use the weight
`S*t*(1-t)/(b-a)`. Keep loss evaluation on a fixed stated reference distribution.
This is a training change separate from logit-time sampling of the ODE.

### F. Use bounded time weights instead of a singular cross-space loss

For exact conversion at the same state:

```text
||v_hat-v||^2 = ||eps_hat-eps||^2 / t^2
```

The velocity loss can therefore be computed directly from epsilon residuals,
without constructing two large velocities and subtracting them. Its gradients
are still scaled by 1/t^2. To bound that scale, test:

```text
L = mean(||eps_hat-eps||^2 / max(t,c_train)^2)
```

This changes the time weighting while keeping epsilon as the regression target.
The conditional epsilon optimum is unchanged for unlimited capacity and a
positive time-only weight, but finite-model optimization changes. Normalize
weights if needed to avoid confounding the comparison with overall loss scale;
record the normalization and gradient norms. Gradient clipping can limit large
optimizer updates, but cannot repair sampling from an inaccurate field.

The clean code does not currently expose this capped epsilon weighting as a
named mode. `eps/v` with exact conversion gives the uncapped common-velocity
loss on the truncated interval. Do not silently relabel a weighted condition
as the historical direct unit-MSE eps/eps experiment.

### G. Why simply matching training and sampling floors can be wrong

The current eps/v training path converts the prediction with the training
floor, then compares against the true velocity x-eps. If eps_hat=eps, its
clamped prediction is:

```text
v_hat_clamped = (t/max(t,c_train))*x-eps
```

Below the floor, this generally differs from x-eps. Thus even a perfect paired
epsilon prediction need not minimize that clamped cross-space loss. Applying
the same floor during sampling does not remove the discrepancy.

For direct eps/eps, the training floor is unused, so it need not match sampling.
For a common-velocity comparison, use exact conversion on a positive interval
or explicitly defined residual weighting. Another possible objective converts
both prediction and target with the same clamped operator, but then it measures
a modified metric, not the original true-velocity MSE. That alternative is not
implemented. Separate configuration is scientifically useful; automatic equality
of the two floor values should not be imposed.

## Both training and sampling: enforce cancellation through the output

### H. Residual epsilon parameterization

Let a network u_theta predict a bounded-scale residual and construct:

```text
eps_hat(z,t) = z-t*u_theta(z,t)
```

Then sampling velocity is directly u_theta, with no division or subtraction of
nearly equal large values. On training path states:

```text
||eps_hat-eps||^2 = t^2*||u_theta-(x-eps)||^2
```

The clean code already exposes the right-hand-side control:

```python
config.framework.prediction_space = "v"
config.framework.loss_space = "v"
config.framework.loss_weighting = "eps_equivalent"
```

This uses the epsilon-equivalent objective through a native velocity network.
It is equivalent to a structured residual epsilon output, not to unchanged raw
epsilon-network optimization. It requires retraining. It guarantees eps_hat=z
at t=0; with a zero residual head, initial velocity is zero instead of the large
velocity from a zero raw epsilon head. It still weights velocity-sized errors
by t^2 during training, so weak low-time learning can remain.

A raw epsilon checkpoint cannot gain this property by wrapping its existing
output: computing u=(z-eps_hat)/t would reintroduce the original problem.
Use this as a separate representation control, preserving historical eps/eps.

### I. Variance-based input/output preconditioning

[EDM](https://arxiv.org/abs/2206.00364) motivates separating network scaling,
skip connections, and loss weights; its [official implementation](https://github.com/NVlabs/edm/blob/main/training/networks.py)
uses explicit input, output, and skip coefficients. The following is a derived
linear-FM epsilon construction, not a copied EDM formula or a tested fix.

Assume centered data with scalar per-coordinate variance sigma_x^2 > 0. Let:

```text
a=t, b=1-t
q=t^2*sigma_x^2+b^2
eps_hat = (b/q)*z + (t*sigma_x/sqrt(q))*F_theta(z/sqrt(q), t)
```

The implied velocity has a division-free endpoint form:

```text
v_hat = ((t*sigma_x^2-b)/q)*z - (sigma_x/sqrt(q))*F_theta
```

q remains strictly positive on [0,1]. The identity skip is built in at t=0,
and the input/output scales are controlled. Predicting scaled residuals can
reduce the burden of learning the near-identity epsilon map. It does not ensure
a good learned field. Data centering, variance estimation, epsilon-loss scaling,
and prediction-space reporting need explicit choices. This requires retraining
and changes the model class. It is more involved than the existing weighted-v
control, so test that simpler control first.

## Proposed sequence

1. Finish successful-old-checkpoint replay with matched old/clean settings.
2. On identical weights and noise, compare exact uniform-t Heun against the
   historical floor. Keep all other settings fixed.
3. Add an explicitly named exact logit-time Heun prototype. Compare at equal
   function evaluations and at higher resolution to check convergence.
4. Investigate arithmetic precision only with a matched FP32/FP64 comparison;
   do not infer precision failure from sample magnitudes alone.
5. For new training, compare raw eps/eps against the existing v/v,
   eps_equivalent control under matched training settings. Test bounded weights
   or more elaborate preconditioning afterward as distinct conditions.

Report raw sample range, out-of-range fraction, fixed-noise grids, quality,
time-resolved losses, and trajectory norms. Existing artifacts remain necessary
for reproducibility. A finite or bounded image is not sufficient evidence of
useful generation.

## Verification and limitations

- Re-read the actual separate sampler/training arguments and CLI mappings.
- Verified exact rational identities for the logit fields, residual epsilon
  loss, and clamped cross-loss discrepancy at several times.
- Verified the variance-based preconditioning identity numerically, including
  finite endpoint limits.
- A scalar zero-epsilon logit-Heun example returned endpoint factors 98.4447,
  99.8932, and 99.9932 at 50/200/800 steps, approaching the exact factor 100.
  This demonstrates that changing time coordinates does not fix a wrong field.
- These were small Python standard-library analytical checks. No project model,
  GPU, checkpoint inference, training, or new sampler implementation ran.
- The source links support general solver/preconditioning ideas. The equations
  specialized to this repository are explicit derivations and proposals, not
  externally established empirical results for these experiments.
