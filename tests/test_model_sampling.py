"""Focused tests for training times, conversions, and ODE sampling."""

import unittest

import torch
from torch import nn

from mem_param_src.config import get_base_config
from mem_param_src.model import GDFM
from mem_param_src.model_setup import build_network
from mem_param_src.networks.small_unet import SelfAttention
from mem_param_src.sampling import (
    model_to_velocity, sample, sample_euler, sample_heun, time_grid,
)


def _small_model(*, prediction_space="x", loss_space="x", loss_weighting="unit"):
    config = get_base_config()
    config.framework.prediction_space = prediction_space
    config.framework.loss_space = loss_space
    config.framework.loss_weighting = loss_weighting
    config.data.data_dim = 2
    config.network.hidden_dim = 8
    config.network.n_layers = 1
    config.network.time_emb_dim = 4
    return GDFM(config)


class _FixedPrediction(nn.Module):
    def __init__(self, prediction):
        super().__init__()
        self.register_buffer("prediction", prediction)

    def forward(self, state, t):
        del state, t
        return self.prediction


class _TrainablePrediction(nn.Module):
    def __init__(self, prediction):
        super().__init__()
        self.prediction = nn.Parameter(prediction.clone())

    def forward(self, state, t):
        del state, t
        return self.prediction


class _TimePrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.call_times = []

    def forward(self, state, t):
        self.call_times.append(float(t[0]))
        return t.view([-1] + [1] * (state.ndim - 1)).expand_as(state)


class _ConstantVelocityModel(nn.Module):
    prediction_space = "v"

    def __init__(self, velocity):
        super().__init__()
        self.velocity = float(velocity)
        self.conversion_settings = []

    def predict(self, state, t):
        del t
        return torch.full_like(state, self.velocity)

    def convert(
        self,
        prediction,
        state,
        t,
        source,
        target,
        *,
        use_safe_div,
        safe_div_denominator_floor,
    ):
        del state, t
        self.conversion_settings.append(
            (use_safe_div, safe_div_denominator_floor, source, target)
        )
        return prediction


class TestTrainingSettings(unittest.TestCase):
    def test_base_config_uses_separate_plain_names(self):
        config = get_base_config()

        self.assertNotIn("schedule", config)
        self.assertEqual(config.framework.loss_weighting, "unit")
        self.assertEqual(config.training.start_time, 0.01)
        self.assertEqual(config.training.end_time, 0.99)
        self.assertIs(config.training.use_safe_div, True)
        self.assertEqual(config.training.safe_div_denominator_floor, 0.01)
        self.assertEqual(config.sampling.start_time, 0.01)
        self.assertEqual(config.sampling.end_time, 0.99)
        self.assertIs(config.sampling.use_safe_div, True)
        self.assertEqual(config.sampling.safe_div_denominator_floor, 0.01)
        self.assertEqual(config.evaluation.f_mem_threshold, 0.3)
        self.assertNotIn("ratio_threshold", config.evaluation)
        self.assertAlmostEqual(
            1.0 - config.training.end_time,
            config.training.safe_div_denominator_floor,
        )
        self.assertAlmostEqual(
            1.0 - config.sampling.end_time,
            config.sampling.safe_div_denominator_floor,
        )
        self.assertEqual(config.sampling.output, "x_1")
        self.assertEqual(config.sampling.eps_to_v, "direct")
        self.assertIs(config.sampling.heun_final_euler, True)

    def test_training_does_not_borrow_missing_sampling_setting(self):
        config = get_base_config()
        del config.training["start_time"]

        with self.assertRaisesRegex(ValueError, "start_time"):
            GDFM(config)

    def test_uniform_training_times_stay_inside_training_range(self):
        model = _small_model()
        model.training_start_time = 0.2
        model.training_end_time = 0.4

        times = model.sample_t(10_000, torch.device("cpu"))

        self.assertGreaterEqual(float(times.min()), 0.2)
        self.assertLess(float(times.max()), 0.4)
        self.assertFalse(hasattr(model, "_sample_t_logit_normal"))


class TestAttentionConfiguration(unittest.TestCase):
    def test_empty_resolution_and_disabled_middle_remove_all_attention(self):
        config = get_base_config()
        config.network.arch = "unet"
        config.data.image_size = 8
        config.data.image_channels = 1
        config.network.base_channels = 8
        config.network.channel_mult = (1,)
        config.network.time_emb_dim = 8
        config.network.attn_resolutions = ()
        config.network.middle_attention = False
        config.network.num_res_blocks = 1

        network = build_network(config)

        self.assertEqual(
            [module for module in network.modules() if isinstance(module, SelfAttention)],
            [],
        )
        output = network(torch.randn(2, 1, 8, 8), torch.tensor([0.2, 0.8]))
        self.assertEqual(tuple(output.shape), (2, 1, 8, 8))


class TestVelocityLossWeighting(unittest.TestCase):
    def test_effective_weights_match_linear_fm_coefficients(self):
        times = torch.tensor([0.2, 0.5, 0.8])
        x_equivalent = _small_model(
            prediction_space="v",
            loss_space="v",
            loss_weighting="x_equivalent",
        )
        eps_equivalent = _small_model(
            prediction_space="v",
            loss_space="v",
            loss_weighting="eps_equivalent",
        )

        self.assertTrue(torch.allclose(x_equivalent.w(times), (1.0 - times) ** 2))
        self.assertTrue(torch.allclose(eps_equivalent.w(times), times**2))

    def test_weighted_velocity_losses_match_diagonal_losses(self):
        clean = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
        noise = torch.tensor([[1.0, 0.25], [-2.0, 1.5]])
        times = torch.tensor([0.2, 0.8])
        time_scale = times[:, None]
        velocity = clean - noise
        velocity_error = torch.tensor([[0.4, -0.2], [-0.3, 0.1]])
        velocity_prediction = velocity + velocity_error

        x_prediction = clean + (1.0 - time_scale) * velocity_error
        eps_prediction = noise - time_scale * velocity_error

        v_as_x = _small_model(
            prediction_space="v",
            loss_space="v",
            loss_weighting="x_equivalent",
        )
        v_as_x.net = _FixedPrediction(velocity_prediction)
        x_diagonal = _small_model(prediction_space="x", loss_space="x")
        x_diagonal.net = _FixedPrediction(x_prediction)

        v_as_eps = _small_model(
            prediction_space="v",
            loss_space="v",
            loss_weighting="eps_equivalent",
        )
        v_as_eps.net = _FixedPrediction(velocity_prediction)
        eps_diagonal = _small_model(prediction_space="eps", loss_space="eps")
        eps_diagonal.net = _FixedPrediction(eps_prediction)

        self.assertTrue(
            torch.allclose(
                v_as_x.compute_loss(clean, eps=noise, t=times),
                x_diagonal.compute_loss(clean, eps=noise, t=times),
            )
        )
        self.assertTrue(
            torch.allclose(
                v_as_eps.compute_loss(clean, eps=noise, t=times),
                eps_diagonal.compute_loss(clean, eps=noise, t=times),
            )
        )

    def test_equivalent_weights_require_native_velocity_training(self):
        for weighting in ("x_equivalent", "eps_equivalent"):
            with self.subTest(weighting=weighting):
                with self.assertRaisesRegex(ValueError, "requires.*prediction_space"):
                    _small_model(
                        prediction_space="x",
                        loss_space="v",
                        loss_weighting=weighting,
                    )


class TestDivisionChoice(unittest.TestCase):
    def test_direct_epsilon_velocity_preserves_stationary_prediction_below_floor(self):
        model = _small_model(prediction_space="eps", loss_space="eps")
        state = torch.tensor([[2.0, -1.0]])
        model.net = _FixedPrediction(state)

        direct = model_to_velocity(
            model,
            state,
            0.01,
            use_safe_div=True,
            safe_div_denominator_floor=0.05,
        )
        via_x = model_to_velocity(
            model,
            state,
            0.01,
            use_safe_div=True,
            safe_div_denominator_floor=0.05,
            eps_to_v="via_x",
        )

        self.assertTrue(torch.equal(direct, torch.zeros_like(state)))
        self.assertTrue(torch.allclose(via_x, -0.8 * state))

    def test_direct_epsilon_velocity_matches_known_path_with_scalar_and_batch_times(self):
        model = _small_model(prediction_space="eps", loss_space="eps")
        for shape in ((2, 2), (2, 1, 2, 2)):
            clean = torch.arange(torch.tensor(shape).prod().item()).reshape(shape)
            clean = clean.to(torch.float64) / 3.0 + 1.0
            noise = torch.linspace(-1.0, 1.0, clean.numel(), dtype=clean.dtype)
            noise = noise.reshape(shape)
            model.net = _FixedPrediction(noise)
            for times in (0.01, torch.tensor(0.01), torch.tensor([0.01, 0.2])):
                with self.subTest(shape=shape, times=times):
                    batch_times = torch.as_tensor(times, dtype=clean.dtype)
                    batch_times = batch_times.expand(clean.shape[0])
                    time_scale = batch_times.view([-1] + [1] * (clean.ndim - 1))
                    state = time_scale * clean + (1.0 - time_scale) * noise
                    expected_velocity = clean - noise

                    exact = model_to_velocity(
                        model,
                        state,
                        times,
                        use_safe_div=False,
                        safe_div_denominator_floor=0.05,
                    )
                    safe = model_to_velocity(
                        model,
                        state,
                        times,
                        use_safe_div=True,
                        safe_div_denominator_floor=0.05,
                    )
                    legacy_exact = model_to_velocity(
                        model,
                        state,
                        times,
                        use_safe_div=False,
                        safe_div_denominator_floor=0.05,
                        eps_to_v="via_x",
                    )

                    expected_safe = time_scale / time_scale.clamp_min(0.05)
                    expected_safe = expected_safe * expected_velocity
                    self.assertTrue(torch.allclose(exact, expected_velocity))
                    self.assertTrue(torch.allclose(safe, expected_safe))
                    self.assertTrue(torch.allclose(legacy_exact, exact))

    def test_sampler_choices_preserve_training_loss_and_gradients(self):
        clean = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
        noise = torch.tensor([[1.0, 0.25], [-2.0, 1.5]])
        times = torch.tensor([0.01, 0.2])
        time_scale = times[:, None]
        state = time_scale * clean + (1.0 - time_scale) * noise

        for loss_space in ("eps", "v"):
            model = _small_model(prediction_space="eps", loss_space=loss_space)
            model.training_safe_div_denominator_floor = 0.05
            model.net = _TrainablePrediction(noise + 0.3)
            prediction = model.net.prediction
            if loss_space == "eps":
                expected_loss = ((prediction - noise) ** 2).mean()
            else:
                # Training retains its existing conversion through clean data.
                clean_prediction = state - (1.0 - time_scale) * prediction
                clean_prediction = clean_prediction / time_scale.clamp_min(0.05)
                expected_loss = (
                    (clean_prediction - prediction - (clean - noise)) ** 2
                ).mean()
            expected_gradient = torch.autograd.grad(expected_loss, prediction)[0]

            for eps_to_v, final_euler in (("direct", True), ("via_x", False)):
                with self.subTest(loss_space=loss_space, eps_to_v=eps_to_v):
                    model.config.sampling.eps_to_v = eps_to_v
                    model.config.sampling.heun_final_euler = final_euler
                    model.sample(
                        noise,
                        n_steps=2,
                        method="heun",
                        start_time=0.01,
                        end_time=0.2,
                        use_safe_div=True,
                        safe_div_denominator_floor=0.05,
                        eps_to_v=eps_to_v,
                        heun_final_euler=final_euler,
                        output="x_t_max",
                    )
                    actual_loss = model.compute_loss(clean, eps=noise, t=times)
                    actual_gradient = torch.autograd.grad(actual_loss, prediction)[0]

                    self.assertTrue(torch.allclose(actual_loss, expected_loss))
                    self.assertTrue(
                        torch.allclose(actual_gradient, expected_gradient)
                    )

    def test_velocity_to_noise_uses_stable_direct_solution(self):
        model = _small_model(prediction_space="v", loss_space="eps")
        clean = torch.tensor([[2.0, -1.0]])
        noise = torch.tensor([[1.0, 0.25]])
        times = torch.tensor([0.999999])
        state = model.z_t_interpolate(clean, noise, times)
        velocity = clean - noise

        exact_noise = model.convert(
            velocity,
            state,
            times,
            source="v",
            target="eps",
            use_safe_div=False,
            safe_div_denominator_floor=0.05,
        )
        safe_noise = model.convert(
            velocity,
            state,
            times,
            source="v",
            target="eps",
            use_safe_div=True,
            safe_div_denominator_floor=0.05,
        )

        self.assertTrue(torch.allclose(exact_noise, noise, atol=2.0e-6))
        self.assertTrue(torch.allclose(safe_noise, noise, atol=2.0e-6))

    def test_exact_and_safe_division_are_distinct_near_path_end(self):
        model = _small_model()
        clean = torch.tensor([[2.0, -1.0]])
        noise = torch.tensor([[1.0, 1.0]])
        times = torch.tensor([0.999])
        state = model.z_t_interpolate(clean, noise, times)

        exact_noise = model.convert(
            clean,
            state,
            times,
            source="x",
            target="eps",
            use_safe_div=False,
            safe_div_denominator_floor=0.05,
        )
        safe_noise = model.convert(
            clean,
            state,
            times,
            source="x",
            target="eps",
            use_safe_div=True,
            safe_div_denominator_floor=0.05,
        )

        self.assertTrue(torch.allclose(exact_noise, noise, atol=2.0e-4))
        self.assertTrue(torch.allclose(safe_noise, noise * 0.02, atol=2.0e-4))

    def test_training_loss_uses_training_division_choice(self):
        model = _small_model(prediction_space="x", loss_space="eps")
        clean = torch.tensor([[2.0, -1.0]])
        noise = torch.ones_like(clean)
        times = torch.tensor([0.999])
        model.net = _FixedPrediction(clean)

        model.training_use_safe_div = False
        exact_loss = model.compute_loss(clean, eps=noise, t=times)
        model.training_use_safe_div = True
        safe_loss = model.compute_loss(clean, eps=noise, t=times)

        self.assertLess(float(exact_loss), 1.0e-6)
        self.assertAlmostEqual(float(safe_loss), 0.81, places=4)


class TestSamplingSettings(unittest.TestCase):
    def test_epsilon_conversion_choice_reaches_all_sampler_entrypoints(self):
        initial = torch.tensor([[2.0, -1.0]])
        model = _small_model(prediction_space="eps", loss_space="eps")
        model.net = _FixedPrediction(initial)

        for eps_to_v, expected_scale in (("direct", 1.0), ("via_x", 0.984)):
            for entrypoint in ("external", "model", "euler", "heun"):
                with self.subTest(eps_to_v=eps_to_v, entrypoint=entrypoint):
                    settings = dict(
                        n_steps=1,
                        start_time=0.01,
                        end_time=0.03,
                        use_safe_div=True,
                        safe_div_denominator_floor=0.05,
                        eps_to_v=eps_to_v,
                    )
                    if entrypoint in ("external", "model"):
                        settings.update(method="heun", output="x_t_max")
                    if entrypoint == "model":
                        result = model.sample(initial, **settings)
                    else:
                        sampler = {
                            "external": sample, "euler": sample_euler,
                            "heun": sample_heun,
                        }[entrypoint]
                        result = sampler(model, initial, **settings)

                    # The legacy first velocity is -0.8 * initial for dt=0.02.
                    self.assertTrue(
                        torch.allclose(result, expected_scale * initial)
                    )

    def test_heun_final_euler_changes_last_interval_and_model_calls(self):
        initial = torch.zeros(2, 2)
        cases = (
            (1, True, 0.08, [0.1]),
            (1, False, 0.4, [0.1, 0.9]),
            (2, True, 0.32, [0.1, 0.5, 0.5]),
            (2, False, 0.4, [0.1, 0.5, 0.5, 0.9]),
        )
        for n_steps, final_euler, expected, expected_times in cases:
            for entrypoint in ("external", "model"):
                with self.subTest(
                    n_steps=n_steps, final_euler=final_euler, entrypoint=entrypoint
                ):
                    model = _small_model(prediction_space="v", loss_space="v")
                    model.net = _TimePrediction()
                    settings = dict(
                        n_steps=n_steps,
                        method="heun",
                        start_time=0.1,
                        end_time=0.9,
                        use_safe_div=False,
                        safe_div_denominator_floor=0.05,
                        heun_final_euler=final_euler,
                        output="x_t_max",
                    )
                    if entrypoint == "external":
                        result = sample(model, initial, **settings)
                    else:
                        result = model.sample(initial, **settings)

                    self.assertTrue(
                        torch.allclose(result, torch.full_like(initial, expected))
                    )
                    self.assertEqual(len(model.net.call_times), len(expected_times))
                    self.assertTrue(
                        torch.allclose(
                            torch.tensor(model.net.call_times),
                            torch.tensor(expected_times),
                        )
                    )

    def test_default_heun_final_euler_still_allows_endpoint_image_readout(self):
        model = _small_model(prediction_space="v", loss_space="v")
        model.net = _TimePrediction()
        initial = torch.zeros(2, 2)

        result = sample(
            model,
            initial,
            n_steps=2,
            method="heun",
            start_time=0.1,
            end_time=0.9,
            use_safe_div=False,
            safe_div_denominator_floor=0.05,
            output="x_1",
        )

        # Final state is 0.32; image readout adds (1 - 0.9) * v(0.9) = 0.09.
        self.assertTrue(torch.allclose(result, torch.full_like(initial, 0.41)))
        self.assertEqual(len(model.net.call_times), 4)
        self.assertTrue(
            torch.allclose(
                torch.tensor(model.net.call_times),
                torch.tensor([0.1, 0.5, 0.5, 0.9]),
            )
        )

    def test_invalid_conversion_and_final_euler_settings_fail_clearly(self):
        model = _ConstantVelocityModel(0.0)
        initial = torch.zeros(1, 2)

        for setting, value in (
            ("eps_to_v", "unknown"), ("heun_final_euler", "false")
        ):
            with self.subTest(setting=setting):
                with self.assertRaisesRegex(ValueError, setting):
                    sample(
                        model,
                        initial,
                        n_steps=2,
                        method="heun",
                        start_time=0.1,
                        end_time=0.9,
                        use_safe_div=True,
                        safe_div_denominator_floor=0.05,
                        output="x_t_max",
                        **{setting: value},
                    )

    def test_step_count_builds_equal_intervals_between_sampling_limits(self):
        model = _ConstantVelocityModel(0.0)

        times = time_grid(
            model,
            n_steps=4,
            start_time=0.2,
            end_time=0.8,
            device="cpu",
        )

        expected = torch.tensor([0.2, 0.35, 0.5, 0.65, 0.8])
        self.assertTrue(torch.allclose(times, expected))

    def test_x_t_max_reaches_sampling_end_time(self):
        model = _ConstantVelocityModel(2.0)
        initial = torch.zeros(3, 2)

        for method in ("euler", "heun"):
            with self.subTest(method=method):
                final_state = sample(
                    model,
                    initial,
                    n_steps=3,
                    method=method,
                    start_time=0.2,
                    end_time=0.8,
                    use_safe_div=False,
                    safe_div_denominator_floor=0.05,
                    output="x_t_max",
                )

                self.assertTrue(
                    torch.allclose(final_state, torch.full_like(initial, 1.2))
                )
        self.assertTrue(model.conversion_settings)
        self.assertTrue(
            all(
                settings == (False, 0.05, "v", "v")
                for settings in model.conversion_settings
            )
        )

    def test_x_1_returns_conventional_data_endpoint_sample(self):
        clean = torch.tensor([[2.0, -1.0]])
        initial = torch.zeros_like(clean)

        for method in ("euler", "heun"):
            with self.subTest(method=method):
                model = _small_model(prediction_space="x", loss_space="x")
                model.net = _FixedPrediction(clean)
                clean_prediction = sample(
                    model,
                    initial,
                    n_steps=2,
                    method=method,
                    start_time=0.1,
                    end_time=0.9,
                    use_safe_div=True,
                    safe_div_denominator_floor=0.05,
                    output="x_1",
                )

                self.assertTrue(torch.equal(clean_prediction, clean))

    def test_invalid_sampling_settings_fail_clearly(self):
        model = _ConstantVelocityModel(0.0)
        initial = torch.zeros(1, 2)

        with self.assertRaisesRegex(ValueError, "positive integer"):
            sample(
                model,
                initial,
                n_steps=0,
                method="euler",
                start_time=0.1,
                end_time=0.9,
                use_safe_div=True,
                safe_div_denominator_floor=0.05,
                output="x_t_max",
            )
        with self.assertRaisesRegex(ValueError, "sampling.start_time"):
            sample(
                model,
                initial,
                n_steps=2,
                method="euler",
                start_time=0.9,
                end_time=0.1,
                use_safe_div=True,
                safe_div_denominator_floor=0.05,
                output="x_t_max",
            )
        with self.assertRaisesRegex(ValueError, "sampling.output"):
            sample(
                model,
                initial,
                n_steps=2,
                method="euler",
                start_time=0.1,
                end_time=0.9,
                use_safe_div=True,
                safe_div_denominator_floor=0.05,
                output="prediction",
            )


class TestFinalPixelClamp(unittest.TestCase):
    def test_final_pixel_clamp_only_changes_completed_output(self):
        model = _ConstantVelocityModel(0.0)
        initial = torch.tensor([[-2.0, -0.5, 0.25, 2.0]])

        clipped, raw = sample(
            model,
            initial,
            n_steps=1,
            method="euler",
            start_time=0.1,
            end_time=0.9,
            use_safe_div=False,
            safe_div_denominator_floor=0.05,
            output="x_t_max",
            final_pixel_clamp=True,
            return_raw=True,
        )
        untouched = sample(
            model,
            initial,
            n_steps=1,
            method="euler",
            start_time=0.1,
            end_time=0.9,
            use_safe_div=False,
            safe_div_denominator_floor=0.05,
            output="x_t_max",
        )

        self.assertTrue(torch.equal(raw, initial))
        self.assertTrue(torch.equal(clipped, initial.clamp(-1.0, 1.0)))
        self.assertTrue(torch.equal(untouched, initial))

    def test_final_pixel_clamp_requires_a_boolean(self):
        with self.assertRaisesRegex(ValueError, "final_pixel_clamp"):
            sample(
                _ConstantVelocityModel(0.0),
                torch.zeros(1, 2),
                n_steps=1,
                method="euler",
                start_time=0.1,
                end_time=0.9,
                use_safe_div=False,
                safe_div_denominator_floor=0.05,
                output="x_t_max",
                final_pixel_clamp="true",
            )


if __name__ == "__main__":
    unittest.main()
