"""Integration tests for the unconditional Small DiT backbone."""

import copy
import inspect
import unittest
from types import SimpleNamespace

import torch

from mem_param_src.config import get_base_config
from mem_param_src.model import GDFM
from mem_param_src.model_setup import build_network
from mem_param_src.networks import SmallDiT


def _reduced_config():
    """Return the cheap 8x8, depth-1 DiT used by integration tests."""
    config = get_base_config()
    config.network.arch = "dit"
    config.network.depth = 1
    config.network.hidden_size = 32
    config.network.num_heads = 4
    config.network.mlp_ratio = 2.0
    config.network.patch_size = 2
    config.network.learn_sigma = False
    config.data.image_size = 8
    config.data.image_channels = 1
    return config


def _reduced_network():
    """Build the cheap DiT through the public factory."""
    return build_network(_reduced_config())


class TestDiTFactory(unittest.TestCase):
    """Exercise canonical, attribute, mapping, and flat factory layouts."""

    def test_canonical_config_constructs_gdfm_without_net_replacement(self):
        model = GDFM(_reduced_config())

        self.assertIsInstance(model.net, SmallDiT)
        self.assertEqual(model.net.image_size, 8)
        self.assertEqual(model.net.image_channels, 1)
        self.assertEqual(model.net.depth, 1)

    def test_nested_mapping_and_attribute_layouts_construct_reduced_dit(self):
        layouts = (
            {
                "network": {
                    "arch": "dit",
                    "depth": 2,
                    "hidden_size": 48,
                    "num_heads": 6,
                    "mlp_ratio": 2.0,
                    "patch_size": 4,
                },
                "data": {"image_size": 8, "image_channels": 3},
            },
            SimpleNamespace(
                network=SimpleNamespace(
                    arch="dit",
                    depth=2,
                    hidden_size=48,
                    num_heads=6,
                    mlp_ratio=2.0,
                    patch_size=4,
                ),
                data=SimpleNamespace(image_size=8, image_channels=3),
            ),
        )

        for config in layouts:
            with self.subTest(config_type=type(config).__name__):
                network = build_network(config)
                self.assertIsInstance(network, SmallDiT)
                self.assertEqual(network.image_size, 8)
                self.assertEqual(network.image_channels, 3)
                self.assertEqual(network.depth, 2)
                self.assertEqual(network.num_heads, 6)

    def test_flat_mapping_layout_and_legacy_head_alias_construct_reduced_dit(self):
        config = {
            "network": "dit",
            "image_size": 8,
            "image_channels": 2,
            "depth": 1,
            "hidden_size": 32,
            "heads": 4,
            "mlp_ratio": 2.0,
            "patch_size": 2,
        }

        network = build_network(config)

        self.assertIsInstance(network, SmallDiT)
        self.assertEqual(network.image_channels, 2)
        self.assertEqual(network.num_heads, 4)

    def test_data_section_has_precedence_over_flat_image_fields(self):
        config = {
            "network": {
                "arch": "dit",
                "depth": 1,
                "hidden_size": 32,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "patch_size": 2,
            },
            "data": {"image_size": 8, "image_channels": 1},
            "image_size": 16,
            "image_channels": 3,
        }

        network = build_network(config)

        self.assertEqual(network.image_size, 8)
        self.assertEqual(network.image_channels, 1)


class TestDiTContract(unittest.TestCase):
    """Check the reduced model's public shape, time, and initialization contract."""

    def test_forward_is_image_shaped_and_receives_unmodified_continuous_times(self):
        network = _reduced_network()
        images = torch.randn(2, 1, 8, 8)
        times = torch.tensor([0.125, 0.875], dtype=torch.float32)
        observed_times = []

        handle = network.t_embedder.register_forward_pre_hook(
            lambda _module, args: observed_times.append(args[0])
        )
        try:
            prediction = network(images, times)
        finally:
            handle.remove()

        self.assertIs(type(prediction), torch.Tensor)
        self.assertEqual(tuple(prediction.shape), tuple(images.shape))
        self.assertIs(observed_times[0], times)
        self.assertTrue(torch.equal(observed_times[0], times))
        self.assertEqual(
            list(inspect.signature(network.forward).parameters), ["z", "t"]
        )

    def test_integer_times_are_rejected(self):
        network = _reduced_network()

        with self.assertRaisesRegex(ValueError, "floating-point continuous times"):
            network(torch.randn(2, 1, 8, 8), torch.tensor([0, 1]))

    def test_adaln_and_output_projection_keep_zero_initialization(self):
        network = _reduced_network()

        self.assertEqual(torch.count_nonzero(network.final_layer.linear.weight), 0)
        self.assertEqual(torch.count_nonzero(network.final_layer.linear.bias), 0)
        for block in network.blocks:
            self.assertEqual(
                torch.count_nonzero(block.adaLN_modulation[-1].weight), 0
            )
            self.assertEqual(
                torch.count_nonzero(block.adaLN_modulation[-1].bias), 0
            )

    def test_state_dict_round_trip_preserves_nontrivial_outputs(self):
        torch.manual_seed(7)
        source = _reduced_network().eval()
        with torch.no_grad():
            source.final_layer.linear.weight.normal_(mean=0.0, std=0.02)
            source.final_layer.linear.bias.normal_(mean=0.0, std=0.02)
        images = torch.randn(2, 1, 8, 8)
        times = torch.tensor([0.2, 0.8])
        with torch.no_grad():
            expected = source(images, times)

        saved_state = copy.deepcopy(source.state_dict())
        restored = _reduced_network().eval()
        restored.load_state_dict(saved_state)
        with torch.no_grad():
            actual = restored(images, times)

        self.assertGreater(float(expected.abs().max()), 0.0)
        self.assertTrue(torch.equal(actual, expected))

    def test_invalid_architecture_dimensions_fail_clearly(self):
        valid = {
            "image_size": 8,
            "image_channels": 1,
            "depth": 1,
            "hidden_size": 32,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "patch_size": 2,
            "learn_sigma": False,
        }
        invalid_cases = (
            ({"image_size": 7}, "divisible by patch_size"),
            ({"hidden_size": 40, "num_heads": 6}, "divisible by num_heads"),
            ({"hidden_size": 30, "num_heads": 6}, "divisible by 4"),
            ({"learn_sigma": True}, "requires learn_sigma=False"),
        )

        for overrides, message in invalid_cases:
            with self.subTest(overrides=overrides):
                arguments = {**valid, **overrides}
                with self.assertRaisesRegex(ValueError, message):
                    SmallDiT(**arguments)


class TestDiTGDFMIntegration(unittest.TestCase):
    """Exercise loss and sampling through GDFM using the reduced model."""

    def test_all_nine_prediction_loss_cells_have_finite_forward_backward(self):
        clean = torch.randn(2, 1, 8, 8)
        noise = torch.randn_like(clean)
        times = torch.tensor([0.2, 0.8])

        for prediction_space in ("x", "eps", "v"):
            for loss_space in ("x", "eps", "v"):
                with self.subTest(
                    prediction_space=prediction_space,
                    loss_space=loss_space,
                ):
                    config = _reduced_config()
                    config.framework.prediction_space = prediction_space
                    config.framework.loss_space = loss_space
                    model = GDFM(config)

                    loss = model.compute_loss(clean, eps=noise, t=times)
                    loss.backward()

                    self.assertTrue(torch.isfinite(loss))
                    gradients = [
                        parameter.grad
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    ]
                    self.assertTrue(gradients)
                    self.assertTrue(
                        all(torch.isfinite(gradient).all() for gradient in gradients)
                    )

    def test_existing_heun_sampler_accepts_dit_without_special_branch(self):
        model = GDFM(_reduced_config()).eval()
        initial_noise = torch.randn(2, 1, 8, 8)

        sample = model.sample(
            initial_noise,
            n_steps=3,
            method="heun",
            start_time=0.1,
            end_time=0.9,
            use_safe_div=True,
            safe_div_denominator_floor=0.1,
            eps_to_v="direct",
            heun_final_euler=True,
            output="x_t_max",
        )

        self.assertEqual(tuple(sample.shape), tuple(initial_noise.shape))
        self.assertTrue(torch.isfinite(sample).all())


class TestActualSmallDiT(unittest.TestCase):
    """Bound one actual 32x32 Small configuration construction and forward."""

    def test_small_defaults_construct_and_run_one_no_grad_forward(self):
        config = get_base_config()
        config.network.arch = "dit"
        config.data.image_size = 32
        config.data.image_channels = 1

        model = GDFM(config).eval()
        parameter_count = sum(parameter.numel() for parameter in model.net.parameters())
        with torch.no_grad():
            output = model.net(torch.randn(1, 1, 32, 32), torch.tensor([0.5]))

        self.assertEqual(model.net.depth, 12)
        self.assertEqual(model.net.hidden_size, 384)
        self.assertEqual(model.net.num_heads, 6)
        self.assertEqual(model.net.patch_size, 2)
        self.assertGreater(parameter_count, 30_000_000)
        self.assertLess(parameter_count, 40_000_000)
        self.assertEqual(tuple(output.shape), (1, 1, 32, 32))


if __name__ == "__main__":
    unittest.main()
