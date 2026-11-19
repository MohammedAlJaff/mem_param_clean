# Experimental design and experiment blocks

This document is a long-form source for the experimental-design appendix. It
is intentionally more complete than the final paper needs. A later writer can
shorten it without having to recover missing settings from job scripts. The
document describes the experimental questions and protocols. It does not claim
that every listed post-hoc evaluation is complete. Training status and compute
status remain in `RUN_JOURNAL.md`.

## 1. Study objective

The study asks how the target used by a generative model changes its path to
memorization. The experiments use linear flow matching. They compare three
ways to express the learning problem while holding the data, model, optimizer,
sampler, and evaluation fixed whenever the comparison permits it.

The experiments focus on small training sets. A small training set lets the
model see the same examples many times. This setting makes the transition from
generalization to memorization visible during training. The main outcome is
therefore a trajectory across checkpoints, not only a final value.

The central study uses grayscale CelebA images at 32 by 32 pixels and a small
U-Net. Additional blocks test whether the main behavior persists with a vision
transformer, handwritten digits, larger color images, different model
capacity, and alternative time weighting of the loss.

## 2. Target parameterizations

Let `x` be a clean training image and let `eps` be an independent Gaussian
noise image. Training uses the linear path

```text
z_t = t x + (1 - t) eps,
```

where `t=0` is the noise end and `t=1` is the data end. The corresponding path
velocity is

```text
v = x - eps.
```

For each training image, the code samples `t` uniformly from `[0.001, 0.999]`
and constructs `z_t`. The network can predict one of three quantities:

- `x`: the clean image;
- `eps`: the source noise;
- `v`: the path velocity.

The loss can also be evaluated in any of these three spaces. A label such as
`eps/v` means that the network directly predicts noise, its prediction is
converted to velocity, and mean squared error is computed against the true
velocity. The first name always gives the network output. The second name gives
the space of the loss. Training uses the exact algebraic conversion between
spaces. The standard prediction-by-loss experiment therefore has nine cells:

| Network output | Clean-image loss | Noise loss | Velocity loss |
| --- | --- | --- | --- |
| Clean image | `x/x` | `x/eps` | `x/v` |
| Noise | `eps/x` | `eps/eps` | `eps/v` |
| Velocity | `v/x` | `v/eps` | `v/v` |

The three diagonal cells use the same space for the network output and the
loss. The six off-diagonal cells separate the output representation from the
loss representation. This distinction is important. A difference between
`x/x` and `eps/eps` can come from either the network output or the effective
time weighting of the error. The complete 3 by 3 grid helps separate these
effects.

Unless a block says otherwise, every cell uses an unweighted mean squared
error. Raw loss values from different target spaces are on different scales.
They must not be interpreted as directly comparable measures of model quality.

## 3. Shared training protocol

### 3.1 Data selection and pairing

Each run saves the exact training tensor and the source identifiers used to
build it. No training block uses data augmentation or class balancing. Within
a matched group, all target cells use the same training subset. Training sizes
are prefix-nested within a seed. For example, the 512-image subset is the first
512 images of the corresponding 2,048-image subset. This rule makes a change
in training-set size easier to interpret because the smaller set is contained
in the larger set.

Seeds control data selection and the stochastic training state. Comparisons
are paired by seed. A target-cell comparison must therefore use cells with the
same seed and training size. Blocks that explicitly pair model initialization,
data order, sampled noise, and sampled times state that additional control in
their block description.

### 3.2 Image preprocessing

CelebA experiments use the official training split for optimization. Each
image is center-cropped to 140 pixels, resized to the requested resolution,
center-cropped again to the exact square size, and converted to either one
grayscale channel or three RGB channels. Pixel values are scaled to `[-1,1]`.
There is no random crop or horizontal flip.

MNIST experiments use the official training split. The original digit image is
scaled to `[-1,1]` and resized to 32 or 64 pixels with bilinear interpolation.
The experiments keep one grayscale channel. The saved class labels are used
only for class-aware evaluation, not for conditional generation.

Flowers102 appears only in the supporting format and capacity studies. Images
are resized, center-cropped, converted to RGB, and scaled to `[-1,1]`. Those
runs use the official training and validation splits.

### 3.3 Optimization

All production runs use Adam with learning rate `1e-4`, betas `(0.9,0.999)`,
epsilon `1e-8`, and no weight decay. The effective batch size is 256. A run
uses gradient accumulation when its model does not fit a physical batch of
256. Production training uses FP32, deterministic execution, and no automatic
mixed precision. Some larger-model blocks permit TF32 matrix multiplication;
this choice is fixed within each comparison and is recorded in the saved
configuration.

Each production trajectory contains one million optimizer updates. The main
CelebA studies save 72 checkpoints on a dense-early and sparse-late schedule,
including steps 0 and 1,000,000. The MNIST study saves 20 aligned checkpoints,
including step 0 and 19 specified points through step 1,000,000. Exact recovery
states include model weights, optimizer state, random-number-generator states,
the shuffled-data cursor, training history, and run identity. A resumed job
continues the same scientific trajectory rather than starting a new replicate.

### 3.4 Models

The small U-Net, called U-Net-S, uses base width 32, channel multipliers
`(1,2,4)`, two residual blocks per resolution, dropout 0.1, a 128-dimensional
time embedding, attention at 16 and 8 pixels, and attention in the middle
block. It has 3,917,409 trainable parameters.

The larger U-Net uses base width 64 and channel multipliers `(1,2,4,4)`. The
CelebA64 version uses attention at 16 and 8 pixels and in the middle block. It
has 24,386,627 trainable parameters. The MNIST64 version uses the same width
and depth but does not use attention.

The transformer backbone is Small-DiT-S/2. It uses 12 transformer blocks, a
hidden width of 384, six attention heads, an MLP ratio of 4, and 2 by 2 image
patches. It predicts one image-sized output and does not predict an additional
variance.

### 3.5 Checkpoint sampling

The common sampler integrates the learned ordinary differential equation from
time 0.001 to 0.999. It uses 50 equal intervals and the Heun method on every
interval. It returns the integrated state at time 0.999, without an extra final
Euler step and without clipping pixels to `[-1,1]`.

Prediction-to-velocity conversion can contain a small denominator near a path
endpoint. The production sampler uses a denominator floor of 0.05. For a noise
prediction, it first reconstructs the clean-image prediction and then converts
that prediction to velocity. This setting is named `via_x`. Training still
uses exact conversion; the floor applies only during sampling.

Each training run saves a small fixed-noise monitoring bank, normally 128
images. This bank supports fast sample grids and a low-cost projected-pixel
memorization diagnostic. It is a training monitor only. It is not the
paper-scale metric bank, and its projected memorization value is not a final
`f_mem` result.

## 4. Post-hoc evaluation protocol

The paper-scale protocol evaluates the complete training trajectory. It uses
the same metric definitions and the same generated noise prefix for all runs
that have a compatible image shape. Each checkpoint is evaluated independently
and is retained as one point on a training-dynamics curve.

### 4.1 Fixed generation bank

The grayscale 32 by 32 evaluation uses one persistent CPU-generated Gaussian
noise artifact with shape `[10000,1,32,32]` and seed 2003. The artifact records
its tensor hash, generator state, PyTorch version, and source-prefix hashes.
Its first 128 rows reproduce the monitoring noise used by the selected seed-0
runs. Paper-tier evaluation uses all 10,000 rows at each checkpoint. Development
smokes can use the first 512 rows, but their metric values are not paper
results.

The evaluator samples once per checkpoint. All quality, memorization, and
diagnostic metrics for that checkpoint use the same generated image bank. The
generated output remains unclipped so that the evaluation can detect numerical
overshoot. Cache identities include the checkpoint, training data, sampler,
noise, encoder, preprocessing, and sample count.

### 4.2 Natural-image quality

Natural-image evaluation reports two complementary Fréchet distances:

- Inception FID uses Inception-v3 features and is the conventional quality
  measure.
- DINOv2 Fréchet distance uses DINOv2 features and is named FD-DINOv2. It is
  not called FID because it does not use Inception features.

CelebA32 uses a frozen bank of 10,000 clean images from the official test
split. The bank is checked against the declared training sets for source-file
and transformed-tensor overlap. The evaluation does not select a new reference
set for each run.

### 4.3 Memorization fraction

For each generated image, the evaluator finds its nearest and second-nearest
training examples in a selected representation. Let their distances be `d1`
and `d2`, with `d1 <= d2`, and define

```text
r = d1 / d2.
```

A small ratio means that one training example is unusually closer than the
next best match. The memorization fraction `f_mem` is the fraction of generated
images with `r` below a fixed or calibrated threshold. The evaluator retains
`d1`, `d2`, `r`, and both neighbor identities. This retained evidence permits
threshold changes and nearest-neighbor figures without repeating image
generation.

For natural images, the representations are raw pixels, SSCD features, LPIPS
features, and DINOv2 features. Raw-pixel results include the fixed thresholds
`rho={0.3,0.5,0.7}`. Every representation also receives a separate calibrated
threshold. The first 1,024 images of the clean held-out bank calibrate
thresholds at false-positive levels `alpha={0.005,0.01,0.02,0.05}`. The
remaining 8,976 images validate those levels. The primary calibrated result
uses `alpha=0.01`. A threshold from one representation is never reused in
another representation.

### 4.4 MNIST quality and memorization

MNIST uses a fixed 128-dimensional JointAE representation. The encoder was
trained on 55,000 official MNIST training images. A separate 5,000-image split
selected its checkpoint. The official 10,000-image test split did not update
the encoder and supplies the final quality statistics.

The evaluator reports a global JointAE Fréchet distance and a class-aware
JointAE Fréchet distance. The class-aware value checks whether the generator
covers all ten digit classes instead of hiding class collapse inside one global
distribution. The evaluator also reports generated class counts and mean
classifier confidence.

MNIST memorization uses raw pixels and standardized, row-normalized JointAE
features. It uses a saved 1,000-image subset of the official test split: 500
images calibrate thresholds and 500 validate them. The nearest-neighbor rule
and retained evidence are otherwise the same as for natural images.

### 4.5 Held-out loss and numerical diagnostics

Each checkpoint receives held-out denoising loss on 512 reference images. One
view uses seeded random training times. Two additional views use fixed times
`t=0.5` and `t=0.75`. These losses show how fitting changes over training, but
raw losses from different target spaces remain on different scales.

The evaluator also records the finite-value fraction, minimum, maximum,
root-mean-square value, fraction of pixels outside `[-1,1]`, and stage runtime.
Plots include quality trajectories, memorization trajectories in every
representation, held-out-loss trajectories, raw-range diagnostics, runtime
diagnostics, selected sample grids, and final-checkpoint nearest-neighbor
panels.

## 5. Production experiment map

The production design contains 234 one-million-update trajectories. The table
counts training runs, not post-hoc evaluation tasks.

| Block | Main factors | Seeds | Runs | Role |
| --- | --- | ---: | ---: | --- |
| CelebA32 U-Net full grid | 4 sizes x 9 prediction/loss cells | 3 | 108 | Main factorial study |
| CelebA32 Small-DiT full grid | 4 sizes at seed 0; 2 sizes at seed 1 | 0, 1 | 54 | Architecture and seed robustness |
| MNIST diagonal | 2 resolutions x 2 sizes x 3 diagonal cells | 3 | 36 | Dataset and resolution robustness |
| CelebA64 RGB diagonal | 2 sizes x 3 diagonal cells | 3 | 18 | Higher-resolution color robustness |
| CelebA32 native-v weighting | 2 sizes x 3 weights | 3 | 18 | Loss-weighting mechanism control |
| **Total** |  |  | **234** |  |

### 5.1 CelebA32 U-Net prediction-by-loss factorial

This block is the main experiment. It uses grayscale CelebA at 32 by 32 pixels
and U-Net-S. It crosses training sizes `N={512,1024,2048,4096}`, seeds
`{0,1,2}`, network outputs `{x,eps,v}`, and loss spaces `{x,eps,v}`. The full
Cartesian product contains 108 runs. Every run uses unit loss weighting and one
million optimizer updates.

The block asks two related questions. First, does the quantity emitted by the
network change when memorization appears? Second, does the space in which error
is penalized change that trajectory? Comparisons within a row of the 3 by 3
grid hold the network output fixed and change the loss. Comparisons within a
column hold the loss space fixed and change the network output. The diagonal
`x/x`, `eps/eps`, and `v/v` cells provide the most common baseline comparison.

Within each seed and training size, all nine cells use the same source-image
identifiers. The smaller training sets are prefixes of the corresponding
4,096-image set. The primary analysis should show both final metrics and the
full checkpoint trajectories. The trajectory view is necessary because two
cells can reach similar final quality through different periods of fitting and
memorization.

Repository source: `jobs/berzelius/celeba32_unet_3x3_1m/` and
`configs/runners/celeba32_unet_3x3_1m.py`.

### 5.2 CelebA32 Small-DiT prediction-by-loss factorial

This block repeats the complete 3 by 3 design with Small-DiT-S/2. Seed 0 uses
all four training sizes `N={512,1024,2048,4096}`, which gives 36 runs. Seed 1
uses `N={512,2048}`, which adds 18 robustness runs. The complete block therefore
contains 54 trajectories.

The block tests whether the main target-parameterization behavior depends on a
convolutional U-Net. The seed-0 matrix supports a direct architecture contrast
at all four training sizes. The seed-1 extension tests repeatability at a small
and a medium training size. It does not provide a second seed at `N=1024` or
`N=4096`, so claims about seed robustness at those sizes must not be made.

Within each seed and size, the nine target cells share the source identifiers,
model initialization, data order, sampled training noise, sampled times, and
fixed monitoring noise. The 512-image subset is a prefix of the 2,048-image
subset for the seed-1 extension. The physical batch is 64 and four
microbatches give the common effective batch of 256.

Repository sources:
`jobs/berzelius/celeba32_dit_3x3_nscale_1m_viax/`,
`jobs/berzelius/celeba32_dit_robustness_s1_1m_viax/`, and their matching
configuration files.

### 5.3 MNIST diagonal and resolution block

This block uses the three diagonal objectives `x/x`, `eps/eps`, and `v/v`.
It crosses two resolutions, `32x32` and `64x64`; two training sizes,
`N={512,2048}`; and seeds `{0,1,2}`. The result is 36 runs.

MNIST32 uses U-Net-S. MNIST64 uses the larger four-stage U-Net with base width
64 and no attention. Both models use the same optimizer, effective batch, time
interval, sampler, and one-million-update budget. The same seed selects the
same original MNIST identities at both resolutions, and the 512-image set is a
prefix of the 2,048-image set.

This block tests whether the diagonal ordering seen for faces also appears for
simple, class-structured data. It also tests whether the behavior remains
visible after increasing image resolution. Resolution and model capacity both
change between MNIST32 and MNIST64. The comparison is therefore an
operating-regime comparison, not a pure estimate of the effect of resolution.
Class-aware JointAE distance, generated class counts, and classifier confidence
are important here because a digit generator can obtain a plausible global
score while omitting classes.

Repository source: `jobs/berzelius/mnist_diagonal_1m/` and
`configs/runners/mnist_diagonal_1m.py`.

### 5.4 CelebA64 RGB diagonal block

This block tests the three diagonal objectives on 64 by 64 RGB CelebA. It
crosses training sizes `N={512,2048}`, seeds `{0,1,2}`, and targets
`{x/x,eps/eps,v/v}`. The result is 18 runs.

All runs use the 24,386,627-parameter wide and deep attention U-Net selected by
the earlier capacity gate. Within each seed, the three target cells use matched
source identities and fixed monitoring noise. The 512-image subset is a prefix
of the 2,048-image subset. All trajectories contain one million updates and 72
checkpoints.

This block asks whether the diagonal result persists when both color and
spatial resolution increase. The data representation and the model capacity
change relative to CelebA32. The block should therefore be described as a
joint higher-resolution, color, and capacity robustness test. It is not a pure
resolution ablation.

Repository source: `jobs/berzelius/celeba64_rgb_diagonal_1m/` and
`configs/runners/celeba64_rgb_diagonal_1m.py`.

### 5.5 Native-velocity loss-weighting block

This block isolates a mechanism that is hidden in the diagonal comparison.
Every model directly predicts velocity and uses velocity-space mean squared
error. The only scientific factor is a scalar time-dependent loss weight:

| Name | Weight | Interpretation |
| --- | --- | --- |
| Clean-data-equivalent | `(1-t)^2` | Equal to the implied `x/x` squared error |
| Unit | `1` | Standard unweighted `v/v` loss |
| Noise-equivalent | `t^2` | Equal to the implied `eps/eps` squared error |

The factors follow from `x_hat = z_t + (1-t) v_hat` and
`eps_hat = z_t - t v_hat`. The weights are used in their raw form and are not
normalized. This choice preserves exact equality with the corresponding
diagonal mean squared errors.

The block crosses `N={512,2048}`, seeds `{0,1,2}`, and the three weights. It
contains 18 dedicated runs. It does not reuse the `v/v` trajectories from the
full 3 by 3 block. Within each seed and size, the three runs share source
identifiers, initial model weights, training order, random streams, and fixed
monitoring noise. The 512-image source set is the prefix of the corresponding
2,048-image set.

The comparison tests whether behavior associated with `x/x` or `eps/eps` can
be reproduced by changing only the time weighting while keeping a native
velocity output. A match would support a loss-weighting explanation. A
difference would show that the network output representation contributes
beyond the scalar weighting.

Repository sources: `jobs/berzelius/celeba32_unet_vv_weighting_1m/`,
`jobs/berzelius/celeba32_unet_vv_weighting_n2048_1m/`, and
`jobs/berzelius/celeba32_vv_weighting_paper_eval/`.

## 6. Supporting and pilot experiment blocks

The following blocks motivated or validated the production design. They can
support an appendix discussion, but they must not be pooled with the main
production matrix as if they were additional independent seeds.

### 6.1 Initial CelebA32 prediction-by-loss pilot

The initial factorial pilot used grayscale CelebA32, U-Net-S, `N=256`, seed 0,
and the full nine-cell prediction-by-loss grid. The planned model target was
one million updates, but the first scientific review occurred at 200,000
updates. The pilot used the same effective batch of 256 and exact training
conversion as the later production block.

The pilot established that the complete 3 by 3 design was executable and made
off-diagonal behavior visible. The later 108-run production matrix supersedes
it for main claims because that matrix uses larger nested training sets, three
seeds, and complete one-million-update trajectories.

Repository source: `jobs/berzelius/celeba32_prediction_loss_3x3/`.

### 6.2 Epsilon recovery and sampler block

The epsilon recovery block studied whether poor `eps/eps` samples came from
training or from conversion near the noise endpoint. It used grayscale
CelebA32, U-Net-S, `N=256`, seed 0, and one million updates. Five array rows
used the same `eps/eps` training configuration and changed only post-checkpoint
sampling: direct or `via_x` conversion, safe or exact division, and full Heun
or a final Euler interval. These rows are sampler repetitions of one training
condition, not five independent training seeds. Another row used `eps/eps`
with learning rate `1e-3`. The final two rows were `v/v` and `x/x` references
at learning rate `1e-4`.

A follow-up block added seeds 1 and 2 for `eps/eps` with direct safe sampling,
`v/v`, and `x/x`. Together with the matched seed-0 references, this gives a
three-seed diagonal check at `N=256`. The diagnostic sampler rows must still be
treated as repeated views of the same scientific training setting.

This block separates a numerical sampling question from a target-training
question. Its post-hoc 10,000-image evaluation used selected checkpoints at
100,000, 390,000, and 695,000 updates. It should be presented as mechanism and
protocol evidence, not added to the main 3 by 3 replicate count.

Repository sources: `jobs/berzelius/eps_recovery_1m/` and
`jobs/berzelius/eps_recovery_multiseed/`.

### 6.3 Data and image-format gate

This seven-run seed-0 study held `eps/eps`, `N=256`, Adam at `1e-4`, and an
effective batch of 256 fixed. It compared:

- CelebA32 grayscale;
- MNIST32 grayscale;
- CelebA32 RGB;
- CelebA64 grayscale;
- CelebA64 RGB;
- Flowers102 RGB at 64 by 64 pixels;
- Flowers102 RGB at 64 by 64 pixels with a deeper U-Net.

The first review gate was 200,000 updates. The study asked whether the useful
or unstable regime was specific to one dataset or image format. Several
properties change between rows, and the final Flowers row also changes model
depth. The block is therefore a broad transfer screen, not a factorial causal
analysis of dataset, color, resolution, and capacity.

Repository source: `jobs/berzelius/eps_eps_data_image_format/`.

### 6.4 64-pixel capacity and attention gate

This study used `eps/eps`, `N=512`, seed 0, and 64 by 64 images. It crossed four
data formats with three U-Net capacity profiles and two attention settings:

- data: CelebA grayscale, CelebA RGB, MNIST grayscale, or Flowers102 RGB;
- capacity: small, deep, or wide and deep;
- attention: no attention, or attention at 16 and 8 pixels plus middle
  attention.

The complete product contains 24 runs. Every run stopped at the 400,000-update
review gate and saved 14 checkpoints. Physical batch size depended on measured
memory use, but gradient accumulation preserved an effective batch of 256.

Each checkpoint was monitored with four sampler profiles: direct-safe,
`via_x`-safe, direct-exact, and `via_x`-exact. The direct-safe profile was the
required stable path; the other profiles were numerical diagnostics. This
block selected the wide and deep attention U-Net later used for the CelebA64
RGB production study. It supports statements about capacity and numerical
stability, but it has only one seed and does not compare all target spaces.

Repository source: `jobs/berzelius/eps_eps_64_n512/`.

### 6.5 Small-DiT diagonal pilot

The transformer pilot trained `x/x`, `eps/eps`, and `v/v` on the same
grayscale CelebA32 subset. It used `N=256`, seed 0, Small-DiT-S/2, an effective
batch of 256, and 100,000 updates. The pilot verified the transformer training
and sampling path before the 54-run production design. It is not an extra seed
for the production training sizes.

Repository source: `jobs/berzelius/celeba32_dit_diagonal_100k_viax/`.

### 6.6 Small-set numerical stress tests

Two very small CelebA32 blocks used `N=16` to make failures visible quickly.
The 10,000-update block compared epsilon conversion, safe versus exact
division, output clipping, and a native `v/v` reference. The 50,000-update
block removed U-Net attention and compared two path endpoint ranges, two
denominator floors, and diagonal `v/v` and `x/x` references. A separate
2,000-update epsilon smoke compared early training and sampler behavior before
the long recovery runs.

These tests used different learning rates, endpoint rules, clipping choices,
and very small data. They are engineering and numerical diagnostics. They must
not appear as independent evidence for the production scientific claims.

Repository sources: the two `celeba32_n16_*` configurations and
`jobs/berzelius/eps_recovery_smoke/`.

## 7. Evaluation and infrastructure checks that are not scientific blocks

Several jobs validate the measurement system rather than a scientific
hypothesis. The 512-image full-suite smoke evaluates all checkpoints of one
CelebA32 U-Net run, one MNIST32 run, and one Small-DiT run. The 10,000-image
replica smoke evaluates the final checkpoint of the same three examples. These
jobs test metric resolution, cache reuse, restart behavior, plots, and artifact
hashes. Their manifests mark them as development-only.

The MNIST JointAE training job creates a fixed evaluation encoder and official
test statistics. It is part of the metric definition, not a generative-model
experiment. Sampling-runtime jobs, resume tests, data probes, and notebook
examples are also infrastructure checks. They should not be counted in the
number of scientific runs or cited as additional experimental evidence.

## 8. Recommended analysis and reporting rules

Each analysis should preserve the pairing built into the design. Compare target
cells at the same seed, training size, model, and checkpoint. Show individual
seed trajectories together with any aggregate curve. State the aggregation and
uncertainty method when the final figure design is frozen.

Quality and memorization should be shown together. A low memorization fraction
does not by itself mean that a model is useful if image quality is poor. A good
quality value does not rule out copying. Held-out loss and raw sample range are
supporting diagnostics, not replacements for generated-image metrics.

Do not compare raw training loss across `x`, `eps`, and `v` spaces as if the
scales were identical. Do not treat the 128-image online monitor as a paper
metric. Do not call FD-DINOv2 or JointAE Fréchet distance FID. Do not treat the
512-image smoke results as publication estimates.

Architecture and format blocks change more than one property. The MNIST32 to
MNIST64 comparison changes model capacity as well as resolution. The CelebA32
to CelebA64 RGB comparison changes resolution, color, and model capacity.
Claims from these blocks should use the term robustness or operating-regime
comparison, not isolated causal effect.

For a paper-tier trajectory, require 10,000 generated images at every reported
checkpoint, the complete metric suite, finite outputs, bounded `f_mem` values,
verified input hashes, and successful cache-only replay. A run becomes a paper
result only after the strict evaluator writes its acceptance record.

## 9. Short reusable description

The following paragraph is a compact version for a main appendix. It omits the
pilot blocks and most implementation details.

> We studied linear flow matching on small image datasets using three target
> representations: clean data (`x`), Gaussian noise (`eps`), and path velocity
> (`v=x-eps`). The network output and the mean-squared-error space were varied
> independently, which produced a full 3 by 3 design. Our main block used a
> 3.9M-parameter U-Net on grayscale CelebA32 with training sizes 512, 1,024,
> 2,048, and 4,096 and three seeds. Every run used Adam, an effective batch of
> 256, and one million updates. We repeated the design with Small-DiT and used
> diagonal `x/x`, `eps/eps`, and `v/v` blocks on MNIST and CelebA64 RGB. A
> separate native-velocity control applied weights `(1-t)^2`, `1`, and `t^2`
> to isolate the time weighting implicit in clean-data and noise losses. We
> evaluated complete checkpoint trajectories with a common fixed-noise bank.
> Natural-image evaluation used Inception FID, FD-DINOv2, and raw-pixel, SSCD,
> LPIPS, and DINOv2 memorization fractions. MNIST used global and class-aware
> JointAE Fréchet distances and raw-pixel and JointAE memorization fractions.
> We also recorded held-out losses and numerical sample diagnostics at each
> checkpoint.
