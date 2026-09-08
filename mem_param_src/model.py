"""
Canonical GDFM model wrapper.

This file is the main mathematical contract for the repo. It defines:

1. the interpolation path between noise and data;
2. how network outputs are converted between `x`, `eps`, and velocity `v`;
3. the training loss for one clean data batch;
4. deterministic ODE sampling from an initial noise tensor.

Backbones live in `mem_param_src.networks`. This wrapper decides what a backbone
prediction means.
"""

import math
from collections.abc import Mapping

import torch
from torch import nn

from mem_param_src.model_setup import build_network


def _get_value(container, name, default=None):
    """
    Read one field from either a mapping-like config or an attribute object.
    """
    if container is None:
        return default

    if isinstance(container, Mapping):
        return container.get(name, default)

    return getattr(container, name, default)


def _named_section(cfg, *names):
    """
    Return the first existing subsection among `names`.
    """
    for name in names:
        value = _get_value(cfg, name, None)
        if value is not None:
            return value
    return None


def _cfg_value(cfg, names, aliases=(), default=None, required=False):
    """
    Resolve one config field from nested or legacy layouts.
    """
    for section_name in names:
        section = _named_section(cfg, section_name)
        if section is not None:
            for alias in aliases:
                value = _get_value(section, alias, None)
                if value is not None:
                    return value

    for alias in aliases:
        value = _get_value(cfg, alias, None)
        if value is not None:
            return value

    if required:
        joined = ", ".join(aliases)
        raise ValueError(f"Missing required config field: {joined}")

    return default


class GDFM(nn.Module):
    """
    Generalized linear-target denoising model.

    Convention:
        t = 0  -> noise/source end
        t = 1  -> data/target end

    Core path:
        z_t = a(t) x + b(t) eps
        v_t = a'(t) x + b'(t) eps

    The network can predict `x`, `eps`, or `v`. The `convert(...)` method is
    the single place that translates among these spaces.
    """

    def __init__(self, config):
        """
        Build the model from one study config.

        First read the path and target conventions. Then build the backbone.
        """
        super().__init__()
        self.config = config

        # =========================================================================
        # Config and conventions
        # =========================================================================

        # ---------------------------------------------------------------------
        # Read the framework-level choices that define what the network
        # predicts and what target space the loss is computed in.
        # ---------------------------------------------------------------------
        self.family_name = _cfg_value(
            config, ("framework",), ("family",), required=True
        )
        self.prediction_space = _cfg_value(
            config, ("framework",), ("prediction_space",), required=True
        )
        self.loss_space = _cfg_value(
            config, ("framework",), ("loss_space",), required=True
        )
        self.loss_weighting = _cfg_value(
            config,
            ("training", "framework"),
            ("loss_weighting",),
            default="unit",
        )

        self.training_start_time = _cfg_value(
            config,
            ("training",),
            ("start_time",),
            required=True,
        )
        self.training_end_time = _cfg_value(
            config,
            ("training",),
            ("end_time",),
            required=True,
        )
        self.training_use_safe_division = _cfg_value(
            config,
            ("training",),
            ("use_safe_division",),
            required=True,
        )
        self.training_safe_div_denom_eps = _cfg_value(
            config,
            ("training",),
            ("safe_div_denom_eps",),
            required=True,
        )
        self._validate_training_settings()

        # =========================================================================
        # Path family and backbone
        # =========================================================================

        self._setup_family()
        self._setup_loss_weighting()
        self.net = build_network(config)

    def _setup_family(self):
        """
        Bind the family-specific coefficient functions.

        After this, the rest of the class can use `a(t)`, `b(t)`, `a'(t)`,
        `b'(t)`, and the active `t` sampler without branching on the family.
        """
        if self.family_name == "fm":
            self.a = self._a_fm
            self.b = self._b_fm
            self.da_dt = self._da_dt_fm
            self.db_dt = self._db_dt_fm
            self._sample_t = self._sample_t_uniform
            return

        raise ValueError(f"Unknown family: {self.family_name}")

    def _validate_training_settings(self):
        """Check the required training-time and division settings."""
        start = float(self.training_start_time)
        end = float(self.training_end_time)
        epsilon = float(self.training_safe_div_denom_eps)

        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("training.start_time and training.end_time must be finite")
        if not 0.0 < start < end < 1.0:
            raise ValueError(
                "Expected 0 < training.start_time < training.end_time < 1, "
                f"got {start} and {end}"
            )
        if not isinstance(self.training_use_safe_division, bool):
            raise ValueError("training.use_safe_division must be true or false")
        if not math.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
            raise ValueError(
                "training.safe_div_denom_eps must be a finite number between 0 and 1"
            )

    def _setup_loss_weighting(self):
        """
        Bind the training-loss weight function.

        Unit weighting means `w(t) = 1` for all samples and all times.
        The equivalent options express matched `x` or `eps` losses as a
        weighted velocity loss for a native `v/v` model.
        """
        if self.loss_weighting in ("unit", "constant"):
            self.w = self._w_unit
            return
        if self.loss_weighting in ("x_equivalent", "eps_equivalent"):
            if self.prediction_space != "v" or self.loss_space != "v":
                raise ValueError(
                    f"loss_weighting={self.loss_weighting!r} requires "
                    "prediction_space='v' and loss_space='v'."
                )
            self.w = {
                "x_equivalent": self._w_x_equivalent,
                "eps_equivalent": self._w_eps_equivalent,
            }[self.loss_weighting]
            return
        if self.loss_weighting == "equal":
            raise ValueError(
                "loss_weighting='equal' is ambiguous. Use loss_weighting='unit' "
                "for w(t)=1. Use loss_space='v' for the common-loss control."
            )

        raise ValueError(f"Unknown loss weighting: {self.loss_weighting}")

    # =========================================================================
    # Prediction and sampling
    # =========================================================================

    def predict(self, z, t):
        """
        Evaluate the backbone at `(z_t, t)`.

        The returned tensor is in `self.prediction_space`, not necessarily in
        velocity space.
        """
        return self.net(z, t)

    @torch.no_grad()
    def sample(
        self,
        z,
        *,
        n_steps,
        method,
        start_time,
        end_time,
        use_safe_division,
        safe_div_denom_eps,
        output,
        device=None,
    ):
        """
        Integrate the learned ODE from noise to data.

        Inputs:
            z: initial noise tensor, already in the ambient data space
            n_steps: number of equal solver steps from `start_time` to `end_time`
            method: `"euler"` or `"heun"`
            output: `"x_t_max"` for the solver state or `"x_1"` for the data endpoint
        """
        from mem_param_src.sampling import sample

        return sample(
            self,
            z,
            n_steps=n_steps,
            method=method,
            start_time=start_time,
            end_time=end_time,
            use_safe_division=use_safe_division,
            safe_div_denom_eps=safe_div_denom_eps,
            output=output,
            device=device,
        )

    # =========================================================================
    # Loss and targets
    # =========================================================================

    def compute_loss(self, x, eps=None, t=None):
        """
        Compute the training loss for one batch of clean data samples `x`.

        Data flow:
            1. sample or accept a noise tensor `eps`
            2. sample or accept times `t`
            3. build the noisy interpolation `z_t`
            4. predict in the model's prediction space
            5. convert that prediction into the configured loss space
            6. compare against the true target in that same space
        """
        batch_size = x.shape[0]
        device = x.device

        if eps is None:
            eps = torch.randn_like(x)
        if t is None:
            t = self.sample_t(batch_size, device)

        z_t = self.z_t_interpolate(x, eps, t)
        prediction = self.predict(z_t, t)
        prediction_in_loss_space = self.convert(
            prediction,
            z_t,
            t,
            source=self.prediction_space,
            target=self.loss_space,
            use_safe_division=self.training_use_safe_division,
            safe_div_denom_eps=self.training_safe_div_denom_eps,
        )
        target = self.get_true_target(x, eps, t, self.loss_space)

        t_b = self._bcast(t, x)
        weight = self.w(t_b)
        per_sample = (
            (weight * (prediction_in_loss_space - target) ** 2).flatten(1).mean(1)
        )
        return per_sample.mean()

    def sample_t(self, batch_size, device):
        """
        Draw training-time scalar times from the active family rule.
        """
        return self._sample_t(batch_size, device)

    def z_t_interpolate(self, x, eps, t):
        """
        Build the path point `z_t = a(t) x + b(t) eps`.
        """
        t_b = self._bcast(t, x)
        return self.a(t_b) * x + self.b(t_b) * eps

    def _sample_t_uniform(self, batch_size, device):
        """
        Sample times uniformly from the configured training interval.
        """
        return (
            torch.rand(batch_size, device=device)
            * (self.training_end_time - self.training_start_time)
            + self.training_start_time
        )

    # =========================================================================
    # FM family coefficients
    # =========================================================================

    def _a_fm(self, t):
        """Flow-matching path coefficient multiplying the clean data term."""
        return t

    def _b_fm(self, t):
        """Flow-matching path coefficient multiplying the noise term."""
        return 1.0 - t

    def _da_dt_fm(self, t):
        """Time derivative of `a(t)` for the FM family."""
        return torch.ones_like(t)

    def _db_dt_fm(self, t):
        """Time derivative of `b(t)` for the FM family."""
        return -torch.ones_like(t)

    def _w_unit(self, t):
        """Unit loss weighting: `w(t) = 1`."""
        return torch.ones_like(t)

    def _w_x_equivalent(self, t):
        """Match `x/x` MSE with `v/v` MSE: `w(t) = (1 - t)^2`."""
        return self.b(t).square()

    def _w_eps_equivalent(self, t):
        """Match `eps/eps` MSE with `v/v` MSE: `w(t) = t^2`."""
        return self.a(t).square()

    # =========================================================================
    # Target-space conversions
    # =========================================================================

    def get_true_target(self, x, eps, t, space):
        """
        Return the exact training target in the requested representation space.

        If the loss is written in `x`, `eps`, or `v`, the target must be in
        that same space.
        """
        if space == "x":
            return x
        if space == "eps":
            return eps
        if space == "v":
            t_b = self._bcast(t, x)
            return self.da_dt(t_b) * x + self.db_dt(t_b) * eps
        raise ValueError(f"Unknown target space: {space}")

    def _x_hat_to_eps_hat(
        self, x_hat, z, t, use_safe_division, safe_div_denom_eps
    ):
        """
        Convert an `x` prediction into an `eps` prediction at the same `z_t`.
        """
        return self._divide(
            z - self.a(t) * x_hat,
            self.b(t),
            use_safe_division,
            safe_div_denom_eps,
        )

    def _eps_hat_to_x_hat(
        self, eps_hat, z, t, use_safe_division, safe_div_denom_eps
    ):
        """
        Convert an `eps` prediction into an `x` prediction at the same `z_t`.
        """
        return self._divide(
            z - self.b(t) * eps_hat,
            self.a(t),
            use_safe_division,
            safe_div_denom_eps,
        )

    def _x_hat_to_v_hat(
        self, x_hat, z, t, use_safe_division, safe_div_denom_eps
    ):
        """
        Convert an `x` prediction into a velocity prediction.
        """
        eps_hat = self._x_hat_to_eps_hat(
            x_hat, z, t, use_safe_division, safe_div_denom_eps
        )
        return self.da_dt(t) * x_hat + self.db_dt(t) * eps_hat

    def _eps_hat_to_v_hat(
        self, eps_hat, z, t, use_safe_division, safe_div_denom_eps
    ):
        """
        Convert an `eps` prediction into a velocity prediction.
        """
        x_hat = self._eps_hat_to_x_hat(
            eps_hat, z, t, use_safe_division, safe_div_denom_eps
        )
        return self.da_dt(t) * x_hat + self.db_dt(t) * eps_hat

    def _v_hat_to_x_hat(
        self, v_hat, z, t, use_safe_division, safe_div_denom_eps
    ):
        """
        Recover an `x` prediction from a velocity prediction.
        """
        a = self.a(t)
        b = self.b(t)
        da = self.da_dt(t)
        db = self.db_dt(t)
        return self._divide(
            b * v_hat - db * z,
            da * b - a * db,
            use_safe_division,
            safe_div_denom_eps,
        )

    def _v_hat_to_eps_hat(
        self, v_hat, z, t, use_safe_division, safe_div_denom_eps
    ):
        """
        Recover an `eps` prediction from a velocity prediction.
        """
        a = self.a(t)
        b = self.b(t)
        da = self.da_dt(t)
        db = self.db_dt(t)
        return self._divide(
            da * z - a * v_hat,
            da * b - a * db,
            use_safe_division,
            safe_div_denom_eps,
        )

    def convert(
        self,
        prediction,
        z_t,
        t,
        source,
        target,
        *,
        use_safe_division,
        safe_div_denom_eps,
    ):
        """
        Convert one prediction tensor between the supported target spaces.

        Supported spaces:
            - `x`
            - `eps`
            - `v`
        """
        self._validate_division_settings(use_safe_division, safe_div_denom_eps)

        if source == target:
            return prediction

        t_b = self._bcast(t, z_t)
        converters = {
            ("x", "eps"): self._x_hat_to_eps_hat,
            ("x", "v"): self._x_hat_to_v_hat,
            ("eps", "x"): self._eps_hat_to_x_hat,
            ("eps", "v"): self._eps_hat_to_v_hat,
            ("v", "x"): self._v_hat_to_x_hat,
            ("v", "eps"): self._v_hat_to_eps_hat,
        }

        key = (source, target)
        if key not in converters:
            raise ValueError(f"Unknown conversion: {source} -> {target}")

        return converters[key](
            prediction,
            z_t,
            t_b,
            use_safe_division,
            safe_div_denom_eps,
        )

    def _validate_division_settings(self, use_safe_division, safe_div_denom_eps):
        """Check one explicit division choice."""
        if not isinstance(use_safe_division, bool):
            raise ValueError("use_safe_division must be true or false")

        epsilon = float(safe_div_denom_eps)
        if not math.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
            raise ValueError(
                "safe_div_denom_eps must be a finite number between 0 and 1"
            )

    def _divide(
        self, numerator, denominator, use_safe_division, safe_div_denom_eps
    ):
        """
        Divide either exactly or with a configured minimum denominator size.

        Safe division changes the equation whenever the denominator magnitude is
        smaller than `safe_div_denom_eps`.
        """
        if not use_safe_division:
            return numerator / denominator

        sign = denominator.sign()
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        safe_denominator = sign * denominator.abs().clamp(
            min=float(safe_div_denom_eps)
        )
        return numerator / safe_denominator

    def _bcast(self, t, ref):
        """
        Broadcast a batch of scalar times to the rank of `ref`.
        """
        if t.dim() != 1:
            raise ValueError(f"Expected t shape [B], got {tuple(t.shape)}")
        if ref.shape[0] != t.shape[0]:
            raise ValueError(
                f"Batch mismatch: ref has {ref.shape[0]} samples, t has {t.shape[0]}"
            )
        return t.view([t.shape[0]] + [1] * (ref.dim() - 1))

    def n_params(self):
        """Return the total parameter count of the full model plus backbone."""
        return sum(parameter.numel() for parameter in self.parameters())

    def __repr__(self):
        """Return a short summary for logs and notebooks."""
        network_cfg = _named_section(self.config, "network")
        return (
            "GDFM(\n"
            f"  family           = {self.family_name}\n"
            f"  prediction_space = {self.prediction_space}\n"
            f"  loss_space       = {self.loss_space}\n"
            f"  loss_weighting   = {self.loss_weighting}\n"
            f"  training_time    = {self.training_start_time} -> {self.training_end_time}\n"
            f"  training_safe_division = {self.training_use_safe_division}\n"
            f"  params           = {self.n_params():,}\n"
            f"  network          = {network_cfg}\n"
            ")"
        )


def print_model_summary(model):
    """
    Print a compact structural summary for a GDFM model or raw backbone.

    If `model` has a `.net` attribute, we treat that as the backbone and print
    both the outer model and the inner network. Otherwise we treat `model`
    itself as the network being inspected.
    """
    backbone = getattr(model, "net", model)

    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(model)

    if backbone is not model:
        print("\n--- Network ---")
        print(backbone)

    print("\n--- Parameter counts ---")
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"Total     : {total:,}")
    print(f"Trainable : {trainable:,}")

    print("\n--- Layer-wise params ---")
    for name, parameter in model.named_parameters():
        print(f"{name:40s} {tuple(parameter.shape)}  ({parameter.numel():,})")


def print_forward_shapes(model, x, t):
    """
    Print layer-by-layer tensor shapes for the active backbone.

    For a full `GDFM` instance we trace `model.net`. For a raw backbone we
    trace the module directly.
    """
    backbone = getattr(model, "net", model)

    print("=" * 60)
    print("FORWARD SHAPES")
    print("=" * 60)

    def hook(name):
        def wrapped(module, inputs, outputs):
            input_shape = tuple(inputs[0].shape) if isinstance(inputs, tuple) else "?"
            output_shape = tuple(outputs.shape)
            print(f"{name:30s} {input_shape} -> {output_shape}")

        return wrapped

    handles = []
    for name, module in backbone.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            handles.append(module.register_forward_hook(hook(name)))

    with torch.no_grad():
        _ = backbone(x, t)

    for handle in handles:
        handle.remove()


__all__ = ["GDFM", "print_forward_shapes", "print_model_summary"]
