"""Standalone ODE samplers for GDFM models."""

import math

import torch


def _infer_device(model, z, device):
    """Choose the device used by the sampler."""
    if device is not None:
        return torch.device(device)
    if z is not None:
        return z.device
    return next(model.parameters()).device


def _batch_time(t, ref):
    """Normalize scalar or batched times to shape ``[B]``."""
    if torch.is_tensor(t):
        if t.dim() == 0:
            return t.to(device=ref.device, dtype=ref.dtype).repeat(ref.shape[0])
        if t.dim() == 1:
            if t.shape[0] != ref.shape[0]:
                raise ValueError(
                    f"Batch mismatch: ref has B={ref.shape[0]}, t has B={t.shape[0]}"
                )
            return t.to(device=ref.device, dtype=ref.dtype)
        raise ValueError(f"Expected scalar t or [B] time tensor, got {tuple(t.shape)}")

    return torch.full((ref.shape[0],), float(t), device=ref.device, dtype=ref.dtype)


def _validate_time_settings(n_steps, start_time, end_time):
    """Check the sampling step count and time range."""
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps <= 0:
        raise ValueError(f"sampling.n_steps must be a positive integer, got {n_steps}")

    start = float(start_time)
    end = float(end_time)
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("sampling.start_time and sampling.end_time must be finite")
    if not 0.0 < start < end < 1.0:
        raise ValueError(
            "Expected 0 < sampling.start_time < sampling.end_time < 1, "
            f"got {start} and {end}"
        )


def _validate_division_settings(use_safe_div, safe_div_denominator_floor):
    """Check the sampling-time division controls."""
    if not isinstance(use_safe_div, bool):
        raise ValueError("sampling.use_safe_div must be true or false")

    epsilon = float(safe_div_denominator_floor)
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
        raise ValueError(
            "sampling.safe_div_denominator_floor must be a finite number between 0 and 1"
        )


def _validate_eps_to_v(eps_to_v):
    """Keep the current and historical epsilon sampling formulas explicit."""
    if eps_to_v not in ("direct", "via_x"):
        raise ValueError("sampling.eps_to_v must be 'direct' or 'via_x'")


def _validate_final_pixel_clamp(final_pixel_clamp):
    """Keep final image clipping explicit and opt-in."""
    if not isinstance(final_pixel_clamp, bool):
        raise ValueError("sampling.final_pixel_clamp must be true or false")


def time_grid(model, *, n_steps, start_time, end_time, device=None):
    """Build ``n_steps`` equal intervals from sampling start to end."""
    _validate_time_settings(n_steps, start_time, end_time)
    device = _infer_device(model, None, device)
    return torch.linspace(
        float(start_time), float(end_time), n_steps + 1, device=device
    )


def predict_in_prediction_space(model, z, t):
    """Evaluate the model in its native prediction space."""
    return model.predict(z, _batch_time(t, z))


def convert_prediction(
    model,
    prediction,
    state,
    t,
    target,
    *,
    use_safe_div,
    safe_div_denominator_floor,
):
    """Convert a model prediction into the requested target space."""
    t = _batch_time(t, state)
    return model.convert(
        prediction,
        state,
        t,
        source=model.prediction_space,
        target=target,
        use_safe_div=use_safe_div,
        safe_div_denominator_floor=safe_div_denominator_floor,
    )


def model_to_x(
    model, state, t, *, use_safe_div, safe_div_denominator_floor
):
    """Convert the model prediction at ``(state, t)`` into clean-data space."""
    prediction = predict_in_prediction_space(model, state, t)
    return convert_prediction(
        model,
        prediction,
        state,
        t,
        target="x",
        use_safe_div=use_safe_div,
        safe_div_denominator_floor=safe_div_denominator_floor,
    )


def model_to_eps(
    model, state, t, *, use_safe_div, safe_div_denominator_floor
):
    """Convert the model prediction at ``(state, t)`` into noise space."""
    prediction = predict_in_prediction_space(model, state, t)
    return convert_prediction(
        model,
        prediction,
        state,
        t,
        target="eps",
        use_safe_div=use_safe_div,
        safe_div_denominator_floor=safe_div_denominator_floor,
    )


def model_to_velocity(
    model, state, t, *, use_safe_div, safe_div_denominator_floor, eps_to_v="direct"
):
    """Convert a sampling prediction to velocity on the linear FM path.

    Direct epsilon sampling uses ``v = (z_t - eps_hat) / t``. Floor this
    denominator only when requested. Training conversions stay in model.py.
    ``via_x`` retains the historical clamped reconstruction for comparison.
    """
    _validate_eps_to_v(eps_to_v)
    prediction = predict_in_prediction_space(model, state, t)
    if model.prediction_space == "eps" and eps_to_v == "direct":
        _validate_division_settings(use_safe_div, safe_div_denominator_floor)
        times = _batch_time(t, state)
        denominator = times.reshape((-1,) + (1,) * (state.ndim - 1))
        if use_safe_div:
            denominator = denominator.clamp_min(float(safe_div_denominator_floor))
        return (state - prediction) / denominator

    return convert_prediction(
        model,
        prediction,
        state,
        t,
        target="v",
        use_safe_div=use_safe_div,
        safe_div_denominator_floor=safe_div_denominator_floor,
    )


@torch.no_grad()
def sample_euler(
    model,
    z,
    *,
    n_steps,
    start_time,
    end_time,
    use_safe_div,
    safe_div_denominator_floor,
    eps_to_v="direct",
    device=None,
):
    """Integrate from ``start_time`` to ``end_time`` with forward Euler."""
    _validate_division_settings(use_safe_div, safe_div_denominator_floor)
    _validate_eps_to_v(eps_to_v)
    device = _infer_device(model, z, device)
    state = z.to(device)
    times = time_grid(
        model,
        n_steps=n_steps,
        start_time=start_time,
        end_time=end_time,
        device=device,
    )

    for step in range(n_steps):
        time_now = times[step]
        time_next = times[step + 1]
        velocity = model_to_velocity(
            model,
            state,
            time_now,
            use_safe_div=use_safe_div,
            safe_div_denominator_floor=safe_div_denominator_floor,
            eps_to_v=eps_to_v,
        )
        state = state + (time_next - time_now) * velocity

    return state


@torch.no_grad()
def sample_heun(
    model,
    z,
    *,
    n_steps,
    start_time,
    end_time,
    use_safe_div,
    safe_div_denominator_floor,
    eps_to_v="direct",
    heun_final_euler=True,
    device=None,
):
    """Use Heun, optionally taking only the Euler predictor on the last interval."""
    _validate_division_settings(use_safe_div, safe_div_denominator_floor)
    _validate_eps_to_v(eps_to_v)
    if not isinstance(heun_final_euler, bool):
        raise ValueError("sampling.heun_final_euler must be true or false")
    device = _infer_device(model, z, device)
    state = z.to(device)
    times = time_grid(
        model,
        n_steps=n_steps,
        start_time=start_time,
        end_time=end_time,
        device=device,
    )

    for step in range(n_steps):
        time_now = times[step]
        time_next = times[step + 1]
        step_size = time_next - time_now
        velocity_now = model_to_velocity(
            model,
            state,
            time_now,
            use_safe_div=use_safe_div,
            safe_div_denominator_floor=safe_div_denominator_floor,
            eps_to_v=eps_to_v,
        )
        trial_state = state + step_size * velocity_now
        if heun_final_euler and step == n_steps - 1:
            # Finish without a velocity evaluation at end_time.
            state = trial_state
            break
        velocity_next = model_to_velocity(
            model,
            trial_state,
            time_next,
            use_safe_div=use_safe_div,
            safe_div_denominator_floor=safe_div_denominator_floor,
            eps_to_v=eps_to_v,
        )
        state = state + 0.5 * step_size * (velocity_now + velocity_next)

    return state


@torch.no_grad()
def sample(
    model,
    z,
    *,
    n_steps,
    method,
    start_time,
    end_time,
    use_safe_div,
    safe_div_denominator_floor,
    output,
    eps_to_v="direct",
    heun_final_euler=True,
    final_pixel_clamp=False,
    return_raw=False,
    device=None,
):
    """Return a final sample, optionally clipping only its completed image range."""
    if output not in ("x_t_max", "x_1"):
        raise ValueError(
            "sampling.output must be 'x_t_max' or 'x_1', "
            f"got {output!r}"
        )

    sampler = {"euler": sample_euler, "heun": sample_heun}.get(method)
    if sampler is None:
        raise ValueError(f"Unknown sampling method: {method}")

    if not isinstance(heun_final_euler, bool):
        raise ValueError("sampling.heun_final_euler must be true or false")
    _validate_final_pixel_clamp(final_pixel_clamp)
    if not isinstance(return_raw, bool):
        raise ValueError("return_raw must be true or false")
    solver_options = {}
    if method == "heun":
        solver_options["heun_final_euler"] = heun_final_euler
    state_t_max = sampler(
        model,
        z,
        n_steps=n_steps,
        start_time=start_time,
        end_time=end_time,
        use_safe_div=use_safe_div,
        safe_div_denominator_floor=safe_div_denominator_floor,
        eps_to_v=eps_to_v,
        device=device,
        **solver_options,
    )
    if output == "x_t_max":
        raw_sample = state_t_max
    else:
        raw_sample = model_to_x(
            model,
            state_t_max,
            float(end_time),
            use_safe_div=use_safe_div,
            safe_div_denominator_floor=safe_div_denominator_floor,
        )
    final_sample = raw_sample.clamp(-1.0, 1.0) if final_pixel_clamp else raw_sample
    if return_raw:
        return final_sample, raw_sample
    return final_sample


__all__ = [
    "convert_prediction",
    "model_to_eps",
    "model_to_velocity",
    "model_to_x",
    "predict_in_prediction_space",
    "sample",
    "sample_euler",
    "sample_heun",
    "time_grid",
]
