"""Focused tests for training times, conversions, and ODE sampling."""

import unittest

import torch
from torch import nn

from mem_param_src.config import get_base_config
from mem_param_src.model import GDFM
from mem_param_src.sampling import sample, time_grid


def _small_model(*, prediction_space="x", loss_space="x"):
    config = get_base_config()
    config.framework.prediction_space = prediction_space
    config.framework.loss_space = loss_space
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
        use_safe_division,
        safe_div_denom_eps,
    ):
        del state, t
        self.conversion_settings.append(
            (use_safe_division, safe_div_denom_eps, source, target)
        )
        return prediction


class TestTrainingSettings(unittest.TestCase):
    def test_base_config_uses_separate_plain_names(self):
        config = get_base_config()

        self.assertNotIn("schedule", config)
        self.assertEqual(config.training.start_time, 0.01)
        self.assertEqual(config.training.end_time, 0.99)
        self.assertIs(config.training.use_safe_division, True)
        self.assertEqual(config.training.safe_div_denom_eps, 0.01)
        self.assertEqual(config.sampling.start_time, 0.01)
        self.assertEqual(config.sampling.end_time, 0.99)
        self.assertIs(config.sampling.use_safe_division, True)
        self.assertEqual(config.sampling.safe_div_denom_eps, 0.01)
        self.assertAlmostEqual(
            1.0 - config.training.end_time,
            config.training.safe_div_denom_eps,
        )
        self.assertAlmostEqual(
            1.0 - config.sampling.end_time,
            config.sampling.safe_div_denom_eps,
        )
        self.assertEqual(config.sampling.output, "x_1")

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


class TestDivisionChoice(unittest.TestCase):
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
            use_safe_division=False,
            safe_div_denom_eps=0.05,
        )
        safe_noise = model.convert(
            velocity,
            state,
            times,
            source="v",
            target="eps",
            use_safe_division=True,
            safe_div_denom_eps=0.05,
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
            use_safe_division=False,
            safe_div_denom_eps=0.05,
        )
        safe_noise = model.convert(
            clean,
            state,
            times,
            source="x",
            target="eps",
            use_safe_division=True,
            safe_div_denom_eps=0.05,
        )

        self.assertTrue(torch.allclose(exact_noise, noise, atol=2.0e-4))
        self.assertTrue(torch.allclose(safe_noise, noise * 0.02, atol=2.0e-4))

    def test_training_loss_uses_training_division_choice(self):
        model = _small_model(prediction_space="x", loss_space="eps")
        clean = torch.tensor([[2.0, -1.0]])
        noise = torch.ones_like(clean)
        times = torch.tensor([0.999])
        model.net = _FixedPrediction(clean)

        model.training_use_safe_division = False
        exact_loss = model.compute_loss(clean, eps=noise, t=times)
        model.training_use_safe_division = True
        safe_loss = model.compute_loss(clean, eps=noise, t=times)

        self.assertLess(float(exact_loss), 1.0e-6)
        self.assertAlmostEqual(float(safe_loss), 0.81, places=4)


class TestSamplingSettings(unittest.TestCase):
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
                    use_safe_division=False,
                    safe_div_denom_eps=0.05,
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
                    use_safe_division=True,
                    safe_div_denom_eps=0.05,
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
                use_safe_division=True,
                safe_div_denom_eps=0.05,
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
                use_safe_division=True,
                safe_div_denom_eps=0.05,
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
                use_safe_division=True,
                safe_div_denom_eps=0.05,
                output="prediction",
            )


if __name__ == "__main__":
    unittest.main()
