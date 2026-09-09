# Configuration Reference

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

Updated: 2026-08-20

This is the human-readable reference for every field currently created by
`mem_param_src.config.get_base_config()`.

The programmatic defaults live in [`mem_param_src/config.py`](../mem_param_src/config.py).
When a setting is added, renamed, or removed, update both files. A test checks
that every leaf field in the base config is mentioned here.

The clean repository has separate image and synthetic training and evaluation
commands. A repeatable `--set` flag changes one existing dotted configuration
field, such as `sampling.use_safe_division`. Old command-line flags remain in a
separate legacy section near the end.

## How to read this reference

The `Use` column has four labels:

- **Active**: read by current clean source code or the sanity notebook.
- **Runner-owned**: read or resolved by the reusable training and evaluation
  commands.
- **Derived**: normally computed by a runner or runtime helper; avoid setting it
  manually unless reproducing a saved run.
- **Reserved**: named in the configuration, but not implemented yet.

Defaults are starting values, not automatically scientific recommendations.
Values copied from the old repository are explicitly labelled **legacy runner
guidance**. They are useful evidence, but they must be revalidated before paper
runs.

## The settings most likely to be changed

```python
cfg.framework.prediction_space = "x"       # "x", "eps", or "v"
cfg.framework.loss_space = "x"             # "x", "eps", or "v"

cfg.training.start_time = 0.01
cfg.training.end_time = 0.99
cfg.training.use_safe_division = True
cfg.training.safe_div_denom_eps = 0.01

cfg.sampling.start_time = 0.01
cfg.sampling.end_time = 0.99
cfg.sampling.use_safe_division = True
cfg.sampling.safe_div_denom_eps = 0.01
cfg.sampling.method = "heun"
cfg.sampling.n_steps = 50
cfg.sampling.output = "x_1"
```

Safe division is enabled with `use_safe_division = True`. Exact division is
selected with `False`.

Training and inference use independent controls. The training loss reads
`training.use_safe_division`. Sampling reads `sampling.use_safe_division`.
Resolved run records must save both values.

The safe-division setting is not free: it changes the conversion equation when
the denominator magnitude is below `safe_div_denom_eps`. For the linear FM
path:

- an `x` prediction converted to `eps` or `v` divides by `1 - t`;
- an `eps` prediction converted to `x` or `v` divides by `t`;
- direct conversions from `v` use a determinant equal to one and do not have
  the same endpoint singularity.

With the paired defaults `start_time = 0.01`, `end_time = 0.99`, and
`safe_div_denom_eps = 0.01`, the clamp threshold matches the smallest ideal
FM path coefficient inside the configured interval. Safe division should
therefore act mainly as a boundary guard instead of changing a broad part of
the interval. Floating-point rounding can still activate it at an endpoint.
The final paper setting remains subject to validation; see T014 in
[`TODO.txt`](../TODO.txt).

## Complete clean base configuration

### Study metadata

| Setting | Default | Meaning and guidance | Use |
| --- | --- | --- | --- |
| `study.name` | `"unnamed_study"` | Stable study identifier. Use a short descriptive name. | Runner-owned |
| `study.description` | `"Target parameterization and memorization dynamics."` | Human-readable purpose saved with a run. | Runner-owned |
| `study.dataset_key` | `""` | Public dataset label used for run identity. It may differ from the internal loader key in `data.dataset`. | Runner-owned |

### Framework and targets

| Setting | Default | Meaning and guidance | Use |
| --- | --- | --- | --- |
| `framework.family` | `"fm"` | Interpolation family. Only linear flow matching (`"fm"`) is supported in the clean model. | Active |
| `framework.prediction_space` | `"x"` | Meaning of the network output: clean data (`"x"`), noise (`"eps"`), or path velocity (`"v"`). Sweep all three for the parameterization study. | Active |
| `framework.loss_space` | `"x"` | Space in which prediction and target are compared: `"x"`, `"eps"`, or `"v"`. Keep fixed when running a common-loss control. | Active |
| `framework.loss_weighting` | `"unit"` | Scalar time weight. Use `"unit"` for `w(t) = 1`. Native `v/v` training also accepts `"x_equivalent"` for `w(t) = (1-t)^2` and `"eps_equivalent"` for `w(t) = t^2`. | Active |

Recommended comparisons are not one universal default:

- matched losses: `x/x`, `eps/eps`, and `v/v`;
- common velocity-loss control: `x/v`, `eps/v`, and `v/v`;
- all other architecture, data, optimizer, sampler, and evaluation settings
  should remain fixed inside one comparison.

For the linear FM path, a velocity prediction implies
`x_hat = z_t + (1-t) v_hat` and `eps_hat = z_t - t v_hat`. Therefore,
`x_equivalent` makes the weighted `v/v` MSE equal to the corresponding `x/x`
MSE. Similarly, `eps_equivalent` makes it equal to the corresponding
`eps/eps` MSE. These two weighting options are valid only for native `v/v`
training. The implementation does not normalize the weights because the raw
factors preserve equality with the diagonal objectives.

### Network

| Setting | Default | Meaning and guidance | Use |
| --- | --- | --- | --- |
| `network.arch` | `"mlp"` | Backbone family. Supported values are `"mlp"` and `"unet"`. | Active |
| `network.hidden_dim` | `128` | Hidden width of the MLP. Ignored by the UNet. | Active for MLP |
| `network.n_layers` | `3` | Number of MLP hidden layers. Ignored by the UNet. | Active for MLP |
| `network.time_emb_dim` | `64` | Time-embedding width. Used by both backbones; `None` lets the UNet choose its internal default. | Active |
| `network.base_channels` | `64` | Base UNet channel width. Ignored by the MLP. | Active for UNet |
| `network.channel_mult` | `(1, 2, 4, 4)` | UNet channel multiplier at each resolution level. | Active for UNet |
| `network.attn_resolutions` | `(8,)` | Image resolutions at which the UNet uses attention. | Active for UNet |
| `network.num_res_blocks` | `2` | UNet residual blocks per resolution level. | Active for UNet |
| `network.dropout` | `0.0` | UNet dropout probability. Use a value in `[0, 1)`. | Active for UNet |

Legacy runner guidance for the synthetic MLP was:

| Setting | Legacy value |
| --- | --- |
| `network.hidden_dim` | `256` |
| `network.n_layers` | `4` |
| `network.time_emb_dim` | `64` |

Legacy image-runner UNet profiles were:

| Image size | Base channels | Channel multipliers | Attention resolutions | Residual blocks | Dropout | Time embedding |
| --- | ---: | --- | --- | ---: | ---: | --- |
| `32` | `32` | `(1, 2, 4)` | `(16, 8)` | `2` | `0.1` | `None` |
| `64` | `64` | `(1, 2, 4, 4)` | `(16,)` | `2` | `0.0` | `None` |
| `128` | `64` | `(1, 2, 4, 4, 8)` | `(16,)` | `2` | `0.0` | `None` |
| `256` | `64` | `(1, 2, 4, 4, 8, 8)` | `(16,)` | `2` | `0.0` | `None` |

The 32-pixel profile was described in the old runner as Bonnaire-close, not as
a DDPM reproduction. These profiles are legacy evidence and have not yet been
validated in the clean repository.

#### Official Bonnaire architecture check (2026-09-07)

The local `BonnaireUNet` is an approximation, not the released network.
Checked the official repository at commit
`de6d4c3abecdce49b6e8ae3dc7898d098228457b`:

- [Training constructor](https://github.com/tbonnair/Why-Diffusion-Models-Don-t-Memorize/blob/de6d4c3abecdce49b6e8ae3dc7898d098228457b/Experiments/src/Training/run_Unet.py#L114-L121)
- [Network implementation](https://github.com/tbonnair/Why-Diffusion-Models-Don-t-Memorize/blob/de6d4c3abecdce49b6e8ae3dc7898d098228457b/Experiments/src/Utils/Unet.py)
- [Paper, Appendix A](https://arxiv.org/html/2505.17638v2#A1)

The released CelebA constructor uses grayscale input/output, multipliers
`(1, 2, 4)`, attention flags `(False, True, True)`, and dropout `0.1`.
It has two residual blocks per encoder level, three per decoder level, and
two middle blocks. Attention uses four heads at 16/8 pixels in encoder and
decoder, plus the middle block. GroupNorm uses eight groups. Time conditioning
adds projected embeddings; the time table uses integer DDPM steps 0--999.

Local CPU instantiation of that exact source gave 1,010,657 / 4,023,233 /
16,054,145 trainable parameters at widths 16 / 32 / 64. All three passed a
finite forward check with input/output shape `[1, 1, 32, 32]` in `memcompgen`.
These counts exclude the frozen sinusoidal lookup table.

The paper's main text says `(1, 2, 3)`; its appendix and released code agree on
`(1, 2, 4)`. Use the pinned code when specifying the released architecture.
Our backbone instead uses single-head attention, up to 32 normalization groups,
scale/shift time conditioning, and zero-initialized residual/output layers.
It has no decoder attention. Optional encoder attention is shared within a
level and skipped at the deepest level; middle attention always runs.
Setting `attn_resolutions=(16, 8)` alone does not reproduce the official model.

The paper trains a DDPM noise predictor. Our linear Flow Matching path is a
separate protocol choice, even if we later adopt the official backbone.

### Data

| Setting | Default | Meaning and guidance | Use |
| --- | --- | --- | --- |
| `data.dataset` | `"gmm_subspace"` | Internal loader key. Supported clean keys are `gmm_subspace`, `torus3d`, `mnist`, `cifar10`, `celeba32_grey`, `flowers102`, `afhq_v2`, and `ffhq128`. | Active |
| `data.root` | `"data"` | Root directory for real datasets. Synthetic loaders do not use it. | Active for images |
| `data.dataset_subdir` | `""` | Optional dataset folder under `data.root`; mainly used by AFHQ-v2 and FFHQ. | Active for selected images |
| `data.download` | `False` | Allow torchvision to download supported datasets. Prefer `False` on cluster jobs after staging data. | Active for torchvision data |
| `data.train_split` | `"train"` | Source split used for optimization. | Active for image collections |
| `data.reference_split` | `"valid"` | Held-out/reference split used for evaluation. Never silently reuse training data. | Active for image collections |
| `data.resize_size` | `0` | Resize before the final center crop. `0` means use `data.image_size`. | Active for CelebA and image collections |
| `data.n_train` | `256` | Number of real or synthetic examples used for optimization. Use at least `2` for nearest-neighbor metrics. | Active |
| `data.n_test` | `256` | Number of held-out/reference examples. Use at least `1`; larger banks give steadier quality estimates. | Active |
| `data.ambient_dim` | `256` | Observed dimension `D` for the torus embedding. The current GMM loader instead reads `data.data_dim`. | Active for torus |
| `data.intrinsic_dim` | `4` | True linear subspace dimension `d` for the GMM. Require `1 <= d <= data.data_dim`. The torus has fixed local dimension two. | Active for GMM |
| `data.data_dim` | `256` | Emitted GMM vector dimension and MLP input dimension. Keep equal to `data.ambient_dim` in synthetic studies until these duplicate names are cleaned up. | Active for GMM/MLP |
| `data.image_size` | `0` | Square model resolution. Must be positive for image loaders and UNet models. | Active for images |
| `data.image_channels` | `0` | Number of image channels. Image loaders support `1` or `3`. MNIST RGB repeats grayscale values and adds no color information. | Active for images |
| `data.flatten` | `False` | Flatten each example after loading. Keep `False` for UNet images. | Active |
| `data.balance_train_classes` | `False` | Select the same training count from each class. The count must be divisible by ten. | Active for MNIST/CIFAR-10 |
| `data.balance_reference_classes` | `False` | Select the same reference count from each class. The count must be divisible by ten. | Active for MNIST/CIFAR-10 |

Enable exact class balance through the existing runner override surface:

```bash
--set data.balance_train_classes=true \
--set data.balance_reference_classes=true
```
| `data.allow_same_source_reference` | `False` | Permit train/reference subsets from one corpus only when saved indices prove they are disjoint. Keep `False` except for an explicit FFHQ-style partition. | Active |
| `data.n_clusters` | `10` | Number of Gaussian components in the controlled GMM. Require at least `1`. | Active for GMM |
| `data.sigma_z` | `0.5` | Standard deviation inside each intrinsic GMM component. Must be positive. | Active for GMM |
| `data.sigma_eta` | `1.0e-3` | Small ambient noise added after embedding. Use `0` for an exact manifold or a non-negative value for thickness. | Active for synthetic data |
| `data.geometry_seed` | `0` | Seed for fixed geometry such as GMM basis/means or torus embedding basis. Keep separate from `training.seed`. | Active for synthetic data |
| `data.major_radius` | `2.0` | Torus major radius. Must be positive and normally greater than `minor_radius`. | Active for torus |
| `data.minor_radius` | `0.5` | Torus tube radius. Must be positive. | Active for torus |
| `data.recipe` | `"default"` | Dataset-profile label saved for provenance. Current clean loaders do not branch on it. | Runner-owned |

Legacy dataset guidance:

- Synthetic GMM runner: `D=2`, `d=2`, `n_train=10`, `n_test=1024`,
  `n_clusters=3`, `sigma_z=0.10`, and `sigma_eta=1e-3`.
- Torus runner: default `D=3`, fixed local manifold dimension `2`, radii
  `2.0` and `0.5`.
- Image runner: public dataset labels were `mnist`, `celeba`, `flowers`,
  `afhq`, and `ffhq`; these resolved to the internal loader keys above.
- CelebA preserves the historical 140-pixel aligned-face crop. It then uses
  `data.resize_size`, applies the final `data.image_size` crop, and emits
  grayscale or RGB according to `data.image_channels`. Sizes above 140 add no
  source detail because they upsample the cropped image.
- Flowers102 used official `test` for training and official `train` for held-out
  reference because the official training split is small.
- FFHQ used one seeded, disjoint same-corpus partition and therefore required
  `data.allow_same_source_reference = True`.

The legacy synthetic values are useful smoke/study starting points, not a
single recommended paper scale.

#### Saved data partitions

One finite source dataset can contribute three disjoint roles:

- `train` contains the selected examples used by the optimizer;
- `reference` contains selected held-out examples used as an evaluation
  baseline;
- unused examples remain in the source dataset and are not copied into the run.

The reference partition is not automatically the complete complement of the
training partition. Each image loader records stable sample identifiers in
exact tensor order. Indexed datasets use split-relative integer indices. Local
folder datasets use relative file paths.

Every version-2 run saves `data_bundle.pt` and `data_manifest.json`. The bundle
contains exact post-transform tensors, labels, indices, and loader metadata.
The manifest records source sizes, ordered-catalog SHA-256 values, selected
sample identifiers, tensor SHA-256 values, and the selection seed. Standalone
evaluation validates the manifest against the saved bundle without opening the
original dataset.

### Training

| Setting | Default | Meaning and guidance | Use |
| --- | --- | --- | --- |
| `training.seed` | `0` | Seed for sample selection, optimization, and run replication. Use multiple seeds for reported results. | Active |
| `training.n_training_steps` | `200` | Number of optimizer updates. `200` is a smoke-test budget, not a paper recommendation. | Runner-owned; notebook active |
| `training.batch_cap` | `64` | Maximum minibatch size; old runners used `min(n_train, batch_cap)`. | Runner-owned |
| `training.batch_size` | `64` | Resolved minibatch size. Normally derived from `batch_cap` and `n_train`. | Derived; notebook active |
| `training.lr` | `1.0e-3` | Optimizer learning rate. Legacy guidance: `1e-3` for synthetic MLP and `1e-4` for image UNet. | Runner-owned; notebook active |
| `training.weight_decay` | `0.0` | Optimizer weight decay. Legacy runners used zero. | Runner-owned; notebook active |
| `training.optimizer` | `"adam"` | Optimizer name. Old image runner allowed `adam` or `sgd`; synthetic used Adam. | Runner-owned |
| `training.momentum` | `0.9` | SGD momentum. Ignored by Adam. | Runner-owned |
| `training.ckpt_every` | `50` | Interval between checkpoints when `ckpt_schedule="interval"`. Must be positive. | Runner-owned |
| `training.ckpt_schedule` | `"interval"` | Named checkpoint policy. Old choices were `interval` and `bonnaire-1m`. | Runner-owned |
| `training.ckpt_steps` | `()` | Explicit checkpoint steps. Takes priority over the named schedule; step zero and the final step should always be included. | Runner-owned |
| `training.resolved_ckpt_steps` | `()` | Final sorted checkpoint steps after applying explicit/named/interval policy. Do not set by hand. | Derived |
| `training.log_every` | `10` | Print/log loss every this many optimizer steps. | Runner-owned |
| `training.show_progress` | `True` | Show interactive progress output when supported. Disable for quiet batch logs. | Runner-owned |
| `training.start_time` | `0.01` | Smallest uniformly sampled training time. Must satisfy `0 < start < end < 1`. | Active |
| `training.end_time` | `0.99` | Largest uniformly sampled training time. Must satisfy `0 < start < end < 1`. | Active |
| `training.use_safe_division` | `True` | `True`: clamp small conversion denominators. `False`: use the exact mathematical division. | Active |
| `training.safe_div_denom_eps` | `0.01` | Minimum denominator magnitude when training safe division is enabled. It matches the default distance from either path endpoint. | Active |

Legacy runner budgets were `10,000` steps for the synthetic MLP and `30,000`
for the image UNet. Both used a `bonnaire-1m` checkpoint schedule by default,
while explicitly passing `--ckpt_every` switched to an interval schedule. Those
budgets and schedules are historical starting points, not clean defaults.

### Evaluation

| Setting | Default | Meaning and guidance | Use |
| --- | --- | --- | --- |
| `evaluation.n_samples` | `64` | Number of generated samples evaluated from the saved noise bank. Must not exceed the available bank. | Runner-owned; notebook active |
| `evaluation.save_generated_samples` | `False` | Save the full raw generated tensor at each image monitoring and post-training evaluation checkpoint. Does not enable generation by itself. | Runner-owned |
| `evaluation.sample_bank_size` | `1024` | Number of fixed initial-noise samples saved for replay across checkpoints. Legacy runners used `1024`. | Runner-owned |
| `evaluation.batch_size` | `64` | No-grad evaluation batch size. Reduce for memory, not scientific meaning. | Runner-owned |
| `evaluation.f_mem_threshold` | `0.3` | Canonical threshold `rho` in `f_mem = mean(d1/d2 < rho)`. Clean image records also save the auxiliary `0.3`, `0.5`, and `0.7` sweep. | Active |
| `evaluation.sample_panel_count` | `8` | Number of examples shown/saved in qualitative panels. | Runner-owned |
| `evaluation.chunk_size` | `256` | Query chunk size for distance metrics. Change for memory/performance only. | Active |
| `evaluation.loss_eval_times` | `(0.5, 0.75)` | Fixed times used for checkpoint loss probes in addition to random-time loss. | Runner-owned |
| `evaluation.loss_eval_batch_size` | `0` | Batch size for loss probes. Legacy image runner used `0` as an automatic/full-batch sentinel; synthetic used the training batch cap. | Runner-owned |
| `evaluation.sliced_w2_n_directions` | `128` | Number of fixed intrinsic-space directions saved for synthetic sliced-Wasserstein evaluation. | Active |

The old synthetic runner used its legacy `ratio_threshold=0.3` flag. The clean
configuration names the same scientific choice `f_mem_threshold`. The image
records retain a rho sweep for threshold-sensitivity analysis.

### Sampling

| Setting | Default | Meaning and guidance | Use |
| --- | --- | --- | --- |
| `sampling.method` | `"heun"` | ODE solver. Supported values are `"euler"` and `"heun"`; Heun is the normal quality setting. | Active |
| `sampling.n_steps` | `50` | Number of equal solver intervals between sampling start and end. Verify convergence before fixing a paper value. | Active |
| `sampling.start_time` | `0.01` | First ODE time. Independent from the training start time. Must satisfy `0 < start < end < 1`. | Active |
| `sampling.end_time` | `0.99` | Final ODE time. Independent from the training end time. | Active |
| `sampling.use_safe_division` | `True` | Safe or exact division used when converting model outputs to sampling velocity. | Active |
| `sampling.safe_div_denom_eps` | `0.01` | Minimum denominator magnitude for sampling conversions. It matches the default distance from either path endpoint. | Active |
| `sampling.output` | `"x_1"` | `"x_1"` returns the conventional data-endpoint sample. `"x_t_max"` returns the integrated state at the configured maximum time. | Active |

The model equations call the interpolated state `z_t`. The public output name
`x_t_max` follows the selected sampling-output terminology. It returns that
integrated state at `sampling.end_time` without an additional model call.

The `x_1` output is the conventional data-endpoint sample. The sampler evaluates
the model at `(x_t_max, t_max)` and applies the path conversion to data space.
This conversion does not add another ODE interval from `t_max` to one.

Legacy sampling-step defaults differed by task:

- image runner: Heun with `24` steps;
- synthetic runner: Heun with `128` steps;
- memorization-boundary evaluator: Heun with `50` steps;
- clean base config: Heun with `50` steps.

These differences mean there is no inherited universal recommendation. Choose
the smallest step count that passes a convergence comparison for the exact
dataset, model, endpoint range, and parameterization sweep.

### Output and monitoring

| Setting | Default | Meaning and guidance | Use |
| --- | --- | --- | --- |
| `outputs.run_name` | `"run"` | Human-readable run identifier used in artifact paths and logs. | Runner-owned |
| `monitoring.batch_size` | `64` | No-grad batch size for train-time monitors. Old image runner defaulted it to `batch_cap`. | Runner-owned |
| `generation_monitor.enabled` | `False` | Save fixed-noise image grids at step zero and every image-training checkpoint. | Runner-owned |
| `generation_monitor.n_samples` | `64` | Number of saved `z_init` samples used in every checkpoint grid. Must not exceed `evaluation.sample_bank_size`. | Runner-owned |
| `generation_monitor.n_train_samples` | `0` | Fixed training examples above generated samples; zero keeps generated-only grids. Uses the first available examples in saved tensor order. Requires generation monitoring. | Runner-owned |
| `generation_monitor.show` | `True` | Reserved display preference. The command runner saves PNG files without opening interactive windows. | Reserved |
| `random_projected_f_mem.enabled` | `False` | Enable rough projected-pixel f_mem at image preview checkpoints. Requires generation monitoring. | Runner-owned |
| `random_projected_f_mem.n_samples` | `512` | Generated samples used by f_mem, separate from grid size. The runner expands and saves the noise bank as needed when enabled. | Runner-owned |
| `random_projected_f_mem.resize_size` | `16` | Image resolution before random projection. | Runner-owned |
| `random_projected_f_mem.projection_dim` | `256` | Dimension of the fixed random projection. | Runner-owned |
| `random_projected_f_mem.projection_seed` | `None` | Derives `training.seed + 3001` when enabled; saved in the resolved config. | Runner-owned/Derived |

### Runtime

| Setting | Default | Meaning and guidance | Use |
| --- | --- | --- | --- |
| `runtime.precision` | `"fp32"` | Requested numerical mode. Legacy choices were `fp32`, CUDA `tf32`, and CUDA `bf16`. Use `fp32` as the correctness baseline. | Runner-owned |
| `runtime.tf32_enabled` | `False` | Resolved CUDA backend state. Should be derived from device and precision. | Derived |
| `runtime.autocast_dtype` | `"none"` | Resolved autocast type; old runtime used `"bfloat16"` for BF16 and `"none"` otherwise. | Derived |
| `runtime.pin_memory` | `True` | Request pinned host memory. Effective only for CUDA transfers. | Runner-owned/Derived |
| `runtime.non_blocking_h2d` | `False` | Use non-blocking host-to-device copies when pinned memory is active. | Derived |

### Console and W&B tracking

| Setting | Default | Meaning and guidance | Use |
| --- | --- | --- | --- |
| `tracking.console.enabled` | `True` | Enable structured console summaries. | Runner-owned |
| `tracking.console.summary_every_steps` | `10` | Step interval for console summaries; initially copied from `training.log_every`. | Runner-owned |
| `tracking.console.summary_every_seconds` | `60.0` | Wall-clock interval for time-based summaries. | Runner-owned |
| `tracking.console.stage_banners` | `True` | Print clear stage transitions such as data, training, and evaluation. | Runner-owned |
| `tracking.console.tee_to_attempt_log` | `True` | Mirror console output into the run attempt log. | Runner-owned |
| `tracking.wandb.enabled` | `False` | Enable W&B loss and progress logging for image training. Synthetic tracking is not supported. | Runner-owned |
| `tracking.wandb.mode` | `"offline"` | `"offline"` or `"online"`. The image `--wandb` flag selects online mode unless `--wandb-mode` overrides it. | Runner-owned |
| `tracking.wandb.project` | `""` | Project name. Empty derives `{dataset}-{model}`. | Runner-owned/Derived |
| `tracking.wandb.name` | `""` | Run name. Empty derives dataset, model, `N`, batch cap, prediction/loss spaces, training steps, and seed. | Runner-owned/Derived |
| `tracking.wandb.group` | `""` | Comparison group. Empty derives dataset, model, and training steps. | Runner-owned/Derived |
| `tracking.wandb.entity` | `WANDB_ENTITY` or `"maj90"` | W&B entity/account, read from the environment when available. | Runner-owned |
| `tracking.wandb.dir` | `"logs/wandb"` | Local W&B storage directory relative to the selected run directory. | Runner-owned |
| `tracking.wandb.tags` | `[]` | List of run tags. Keep scientific factors machine-readable. | Runner-owned |
| `tracking.wandb.notes` | `""` | Free-text run notes. Do not encode factors only in notes. | Runner-owned |
| `tracking.wandb.strict` | `False` | If `True`, tracking failures abort the run; if `False`, training may continue with a warning. | Runner-owned |

## Clean runner commands

Training commands load one Python experiment config through `get_config()`.
They reject unknown `--set` fields and non-empty run directories.

```bash
mem-param-image-train \
  --config configs/runners/flowers102_smoke.py \
  --run-dir outputs/flowers_run \
  --set framework.prediction_space=v \
  --evaluate-after-training

mem-param-image-train \
  --config configs/runners/flowers102_smoke.py \
  --run-dir outputs/flowers_wandb_run \
  --wandb \
  --generation-monitor

mem-param-synthetic-train \
  --config configs/runners/gmm_smoke.py \
  --run-dir outputs/gmm_run
```

The image runner logs loss, learning rate, elapsed time, and progress at
`training.log_every`. Run `wandb login` before online tracking. Checkpoint
monitoring saves `monitoring/samples_step_<step>.png` from the same `z_init`
prefix at step zero and every checkpoint. Omit `--generation-monitor` to keep
W&B loss tracking without the sampling cost. These grids are qualitative
monitors, not checkpoint evaluation metrics.

The image training CLI also accepts these controls. Omitted flags preserve the
config. Explicit flags take precedence over corresponding `--set` overrides.
Every resolved value is saved in the run's `config.json`.

| CLI flag | Saved field | Meaning |
| --- | --- | --- |
| `--steps` | `training.n_training_steps` | Number of optimizer updates. |
| `--lr` | `training.lr` | Optimizer learning rate. |
| `--seed` | `training.seed` | Data/model seed and base for the fixed-noise seed. |
| `--batch-size` | `training.batch_cap` | Maximum training batch size, capped at N. |
| `--checkpoint-every` | `training.ckpt_every` | Select interval checkpoints and clear the explicit schedule. |
| `--sampler` | `sampling.method` | `heun` or `euler`. |
| `--sampling-steps` | `sampling.n_steps` | Number of solver intervals. |
| `--sampling-safe-division` / `--no-sampling-safe-division` | `sampling.use_safe_division` | Enable or disable sampling denominator clamping. |
| `--sampling-denom-eps` | `sampling.safe_div_denom_eps` | Denominator floor; active only when sampling safe division is enabled. |
| `--sampling-start-time` | `sampling.start_time` | Start of integration. |
| `--sampling-end-time` | `sampling.end_time` | End of integration. |
| `--sampling-output` | `sampling.output` | Clean endpoint estimate `x_1` or integrated state `x_t_max`. |

These sampling flags leave the training objective and training-time interval
unchanged. Unit-MSE eps/eps training compares noise directly. Sampling still
converts predicted noise into velocity, where division by small coefficients
can amplify error. Compare sampler settings using the same checkpoints and
saved noise to isolate their effect. The cause of the eps/eps discrepancy
remains unresolved.

Add `--training-grid-samples 16` to place fixed training examples above generated
samples. The combined panels are saved to `monitoring/comparison_grids/step_<step>.png`.
The top panel uses the first 16 entries of `data_bundle.pt["train"]`, or all
entries if fewer exist. Both panels use the fixed display range [-1,1].
Set the count to zero for the original generated-only grids.

Generation monitoring also updates `monitoring/plots/loss.png` from
`train_history.json` at each checkpoint with recorded loss. The values are
logged batch losses, without smoothing. With f_mem monitoring enabled,
`monitoring/plots/f_mem.png` shows the saved projected-pixel records, including
step zero. These figures need only the image training runner. The loss history
is saved at checkpoints so completed records remain available during training.

Add `--f-mem-monitor --f-mem-samples 512` with `--generation-monitor` for a
rough memorization check. `--no-f-mem-monitor` disables it. The sample-count
flag alone does not enable the calculation. It works with or without W&B.
The grid and metric share one generated batch from saved `z_init` prefixes.
The grid retains `generation_monitor.n_samples` images. Generation uses the
larger of the two counts, batched with `monitoring.batch_size`.

Use `--save-generated-samples` to save this entire batch to
`monitoring/generated_samples/step_<step>.pt`. It preserves raw CPU tensors
before clamping, resizing, or projection, including samples omitted from the grid.
Their order matches the saved `z_init` prefix. Sampling is not repeated.
`--no-save-generated-samples` disables tensor storage without disabling grids
or metrics. The base default is off; the MNIST16 monitoring config enables it.
Post-training image evaluation uses the same setting and saves each batch under
`evaluations/<name>/generated_samples/`. Its existing final-sample file remains.
For an older run, use the image evaluator with
`--set evaluation.save_generated_samples=true` to retain each replayed batch.

This monitor reuses the old runner's fixed random-projected pixel metric:
normalize/clamp to [0,1], resize to 16 x 16, then project to 256 dimensions.
It compares against all saved training images at `evaluation.f_mem_threshold`
(rho=0.3). It is not FID or the full-resolution post-training pixel metric.
Clamping can hide unstable raw magnitudes; interpret f_mem beside image grids
and generation quality. A small generated count gives a coarse estimate.

Projection and cached training vectors are saved under `monitoring/` as
`f_mem_projection.pt` and `f_mem_train_vectors.pt`. Each checkpoint writes
`f_mem_step_<step>.json`, including sample count, rho, metric space, and distances.
W&B logs `generation/random_projected_pixels/f_mem`, `mean_r`, `rho`, `n_samples`,
and `N` against `generation/step`. Projection settings live in the saved config
and metric records. The disabled option adds no metric sampling or artifacts.

Evaluation commands require only the saved run directory. They load the saved
config, data manifest, tensors, fixed noise, metric state, and checkpoints.
They never rebuild the dataset.

```bash
mem-param-image-eval --run-dir outputs/flowers_run

mem-param-synthetic-eval \
  --run-dir outputs/gmm_run \
  --eval-config configs/eval/exact_division.py \
  --eval-name exact_division \
  --set sampling.output=x_1
```

An optional evaluation config may contain only `sampling` and `evaluation`
sections. The setting order is saved run, evaluation config, then `--set`.
Changed evaluation settings require a new `--eval-name`.

## Findings from the old runners

The inspected old working repository was:

```text
/Users/mohammadaljaff/Desktop/PhD/projects/mem_param/mem_par
```

The preservation repository was not inspected.

Primary legacy sources:

```text
src/mp_config/base.py
src/mp_runners/image_config.py
src/mp_runners/image_train_runner.py
src/mp_runners/synthetic_train_runner.py
src/mp_runners/eval_synthetic.py
src/mp_runners/mem_boundary_eval.py
prototyping/runners/README.md
```

The old CLI is not available in the clean repository. The entries below are a
migration inventory, not commands that currently work here.

### Renamed or removed legacy settings

| Old setting | Clean setting or status | Reason |
| --- | --- | --- |
| `schedule.t_min` | Split into `training.start_time` and `sampling.start_time` | Training and inference ranges must be independent. |
| `schedule.t_max` | Split into `training.end_time` and `sampling.end_time` | Same reason. |
| `schedule.denom_eps` | Split into the two `safe_div_denom_eps` fields | Training and sampling division policies must be independent. |
| implicit safe division | Two `use_safe_division` flags | Exact versus clamped division is now explicit. |
| `schedule.time_sampling` | Removed | Clean training uses uniform times only. |
| `schedule.logit_mu` | Removed | Logit-normal time sampling was removed. |
| `schedule.logit_sigma` | Removed | Logit-normal time sampling was removed. |
| `schedule.beta_min` | Removed | VP schedules are outside the current FM runner scope. |
| `schedule.beta_max` | Removed | VP schedules are outside the current FM runner scope. |
| `network.patch_size`, `depth`, `heads` | Removed | JiT/transformer backbone is not retained. |
| `network.bottleneck_dim`, `proj_dropout`, `attn_dropout` | Removed | Same JiT scope removal. |
| `network.in_context_len`, `in_context_start` | Removed | Same JiT scope removal. |
| old `sampling.output="state"` | `sampling.output="x_t_max"` | State after the final solver step. |
| old `sampling.output="clean_x"` | `sampling.output="x_1"` | Conventional data-endpoint sample from the final solver state. |

### Legacy image-runner CLI inventory

| Old flag | Default or choices | Meaning / clean destination |
| --- | --- | --- |
| `--device` | `cpu` | Execution device; operational, not in the clean base config. |
| `--out_dir` | `outputs/runners/image_train_runner` | Artifact root; operational. |
| `--run_name` | `image_train_runner` | `outputs.run_name`. |
| `--seed` | `0` | `training.seed`. |
| `--dataset` | `celeba`; five public choices | Resolves `study.dataset_key`, internal `data.dataset`, splits, and profile. |
| `--image_size` | dataset profile; `32/64/128/256` | `data.image_size` plus matching model profile. |
| `--model` | `unet` | `network.arch`. |
| `--base_channels` | profile | `network.base_channels`. |
| `--channel_mult` | profile | Comma-separated `network.channel_mult`. |
| `--attn_resolutions` | profile | Comma-separated `network.attn_resolutions`. |
| `--num_res_blocks` | profile | `network.num_res_blocks`. |
| `--dropout` | profile | `network.dropout`. |
| `--prediction_space` | `v`; `x/eps/v` | `framework.prediction_space`. |
| `--loss_space` | `v`; `x/eps/v` | `framework.loss_space`. |
| `--n_train` | `128` | `data.n_train`. |
| `--n_test` | `128` | `data.n_test`. |
| `--n_training_steps` | `30000` | `training.n_training_steps`. |
| `--stop_after_step` | `None` | Planned chunk boundary while preserving the logical final target; not in clean config. |
| `--resume_from_checkpoint` | empty | Explicit resume checkpoint; not in clean config. |
| `--resume_from_attempt` | empty | Resume latest checkpoint in an attempt; not in clean config. |
| `--ckpt_every` | `2000` | `training.ckpt_every`; explicitly passing it selected interval scheduling. |
| `--ckpt_schedule` | `bonnaire-1m` | `training.ckpt_schedule`; old choices `interval/bonnaire-1m`. |
| `--ckpt_steps` | empty | Explicit comma-separated `training.ckpt_steps`. |
| `--log_every` | automatic | `training.log_every`; old auto rule was `min(500, ckpt_every)`, at least one. |
| `--batch_cap` | `64` | `training.batch_cap`. |
| `--optimizer` | `adam`; `adam/sgd` | `training.optimizer`. |
| `--lr` | `1e-4` | `training.lr`. |
| `--weight_decay` | `0` | `training.weight_decay`. |
| `--momentum` | `0.9` | `training.momentum`, used only by SGD. |
| `--precision` | `fp32`; `fp32/tf32/bf16` | `runtime.precision`. |
| `--data_summary` | off | Materialize and summarize data, then exit; no clean field. |
| `--generation_monitor_samples` | `64` | `generation_monitor.n_samples`. |
| `--monitor_batch_size` | `batch_cap` | `monitoring.batch_size`. |
| `--eval_sample_bank_size` | `1024` | Both `evaluation.sample_bank_size` and old `evaluation.n_samples`. |
| `--sampling_steps` | `24` | `sampling.n_steps`. |
| `--root` | `data` | `data.root`. |
| `--download` / `--no_download` | off | `data.download`. |
| `--wandb` / `--no_wandb` | on | `tracking.wandb.enabled`; clean default is off. |
| `--wandb_mode` | `online` | `tracking.wandb.mode`; clean default is offline. |
| `--wandb_project` | automatic | `tracking.wandb.project`. |
| `--wandb_name` | automatic | Legacy run display name; no clean field yet. |
| `--wandb_group` | automatic | Legacy comparison group; no clean field yet. |
| `--wandb_tags` | empty | `tracking.wandb.tags`. |
| `--compact_model_summary` / inverse | on | Console presentation; no clean field yet. |
| `--generation_monitor` | off | `generation_monitor.enabled`. |
| `--generation_monitor_show` / inverse | on | `generation_monitor.show`. |
| `--random_projected_f_mem_samples` | `512` | `random_projected_f_mem.n_samples`. |
| `--random_projected_f_mem_resize` | `16` | `random_projected_f_mem.resize_size`. |
| `--random_projected_f_mem_proj_dim` | `256` | `random_projected_f_mem.projection_dim`. |
| `--random_projected_f_mem_proj_seed` | `training.seed + 3001` | `random_projected_f_mem.projection_seed`. |

### Legacy synthetic-runner CLI inventory

Shared meanings are the same as the image table. Important different or extra
flags were:

| Old flag | Default or choices | Meaning / clean destination |
| --- | --- | --- |
| `--device` | `auto` | Resolve to CUDA when available, otherwise CPU. |
| `--dataset` | `gmm_subspace`; `gmm_subspace/torus3d` | `data.dataset`. |
| `--ambient_dim` | `2`; torus defaulted to `3` | Observed dimension `D`; now `data.ambient_dim` for torus and `data.data_dim` for GMM. |
| `--intrinsic_data_dim` | `2` | GMM intrinsic dimension `d`; now `data.intrinsic_dim`. Rejected for torus. |
| `--n_train` | `10` | `data.n_train`. |
| `--n_test` | `1024` | `data.n_test`. |
| `--n_clusters` | `3` | `data.n_clusters`. |
| `--sigma_z` | `0.10` | `data.sigma_z`. |
| `--sigma_eta` | `1e-3` | `data.sigma_eta`. |
| `--geometry_seed` | `0` | `data.geometry_seed`. |
| `--major_radius` | `2.0` | `data.major_radius`. |
| `--minor_radius` | `0.5` | `data.minor_radius`. |
| `--hidden_dim` | `256` | `network.hidden_dim`. |
| `--n_layers` | `4` | `network.n_layers`. |
| `--time_emb_dim` | `64` | `network.time_emb_dim`. |
| `--n_training_steps` | `10000` | `training.n_training_steps`. |
| `--batch_cap` | `10` | `training.batch_cap`. |
| `--lr` | `1e-3` | `training.lr`. |
| `--log_every` | `100` | `training.log_every`. |
| `--ckpt_every` | `2000` | `training.ckpt_every`. |
| `--eval_sample_bank_size` | `1024` | Evaluation fixed-noise bank. |
| `--sampling_method` | `heun`; `euler/heun` | `sampling.method`. |
| `--sampling_steps` | `128` | `sampling.n_steps`. |
| `--ratio_threshold` | `0.3` | Legacy name for `evaluation.f_mem_threshold`. |
| `--chunk_size` | `512` | `evaluation.chunk_size`. |
| `--sliced_w2_projections` | `128` | Replaced by `evaluation.sliced_w2_n_directions`. |
| `--wandb_entity` | environment or `maj90` | `tracking.wandb.entity`. |
| `--wandb_notes` | empty | `tracking.wandb.notes`. |
| `--wandb_strict` | off | `tracking.wandb.strict`. |

The remaining shared flags were `--out_dir`, `--run_name`, `--seed`,
`--prediction_space`, `--loss_space`, `--weight_decay`, checkpoint controls,
and the W&B enable/mode/project/name/group/tags controls described above.

### Legacy replay and memorization-boundary flags

The simple synthetic replay runner exposed only:

| Old flag | Meaning |
| --- | --- |
| `--run_dir` | Required completed `DONE_` attempt directory. Replay loaded saved tensors, geometry, noise, config, and checkpoints instead of rebuilding data. |
| `--device` | Evaluation device, with `auto` resolution. |

The specialized memorization-boundary evaluator had a separate operational
surface. These are not candidates for the base scientific config unless a
clean evaluator needs them:

| Old flag | Legacy default | Meaning |
| --- | --- | --- |
| `--source_root` | dated job-set path | Read-only training-attempt root. |
| `--out_dir` | separate dated eval path | Evaluation artifact destination. |
| `--manifest_csv` | `None` | Load a prepared run manifest. |
| `--resume_manifest_tsv` | `None` | Join checkpoints from resumed attempts. |
| `--all_done` | off | Discover all completed attempts. |
| `--expected_manifest_count` | `84` | Validate expected run count. |
| `--n_train` | `16384` | Target one training-size slice. |
| `--seed` | `0` | Target one seed. |
| `--params` | `x_x,eps_eps,v_v,x_v` | Parameterization pairs to evaluate. |
| `--sample_count` | `10000` | Generated samples per checkpoint. |
| `--fid_reference_count` | `10000` | Held-out samples used for FID statistics. |
| `--heldout_split` | `valid` | Held-out CelebA split. |
| `--device` | `cuda` | Evaluation device. |
| `--sampler` | saved value | Optional `euler/heun` override. |
| `--sampling_steps` | saved value | Optional solver-step override. |
| `--sample_batch_size` | `256` | Generation batch size. |
| `--fid_batch_size` | `128` | FID feature batch size. |
| `--loss_batch_size` | `256` | Held-out loss batch size. |
| `--fid_feature_extractor` | `inception-v3-compat` | FID feature implementation. |
| `--checkpoint_steps` | all | Optional comma-separated checkpoint subset. |
| `--limit_checkpoints` | `None` | Debug cap on checkpoint count. |
| `--manifest_index` | `None` | Evaluate one manifest row. |
| `--manifest_indices` | `None` | Evaluate selected manifest rows. |
| `--use_slurm_array_task_id` | off | Select manifest row from Slurm array ID. |
| `--manifest_only` | off | Build/validate the manifest without evaluation. |
| `--write_manifest` | off | Write the resolved manifest. |
| `--combine_only` | off | Combine existing per-run outputs only. |
| `--overwrite` | off | Recompute existing evaluation outputs. |
| `--allow_checkpoint_count_mismatch` | off | Bypass strict checkpoint-count validation. |
| `--verify_deterministic_inputs` / inverse | on | Recheck fixed saved inputs before evaluation. |
| `--skip_combined` | off | Skip combined cross-run outputs. |

## Current recommendations versus unresolved choices

Use these as operational starting points:

- `framework.family = "fm"`;
- `framework.loss_weighting = "unit"`;
- `sampling.method = "heun"`;
- `sampling.output = "x_1"`;
- `runtime.precision = "fp32"` for correctness baselines;
- held-out reference data and fixed saved noise for checkpoint comparisons;
- multiple seeds and a rho sensitivity sweep for reported results.

These choices are intentionally unresolved and must not be treated as settled
paper defaults:

- training and sampling start/end times;
- exact versus safe division;
- denominator epsilon;
- final sampling step count;
- synthetic versus image training budget;
- dataset-specific batch size and architecture profile;
- memorization threshold used for a headline scalar.

Whenever one of those is fixed, record the evidence and decision in
[`MIGRATION_JOURNAL.md`](../MIGRATION_JOURNAL.md).
