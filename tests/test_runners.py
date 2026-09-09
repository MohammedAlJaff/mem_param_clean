"""Focused tests for runner configuration, artifacts, training, and replay."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from mem_param_src.config import (
    apply_overrides,
    get_base_config,
    load_python_config,
    parse_cli_overrides,
)
from mem_param_src.datasets.common import (
    relative_path_sample_ids,
    seeded_indices,
    source_catalog_sha256,
    validate_data_bundle,
    validate_finite_data_identity,
)
from mem_param_src.runners import validate_run_kind
from mem_param_src.runners.artifacts import (
    ARTIFACT_FORMAT_VERSION,
    list_checkpoints,
    load_torch,
    prepare_evaluation_dir,
    prepare_run_dir,
    read_json,
    tensor_sha256,
    validate_data_manifest,
    write_json,
)
from mem_param_src.runners.evaluation import (
    evaluate_image_run,
    evaluate_synthetic_run,
)
from mem_param_src.runners.image_monitoring import generate_monitor_samples
from mem_param_src.runners.image_plotting import save_training_comparison_grid
from mem_param_src.runners.image_train import main as image_train_main
from mem_param_src.runners.synthetic_eval import main as synthetic_eval_main
from mem_param_src.runners.synthetic_train import main as synthetic_train_main
from mem_param_src.runners.tracking import resolve_wandb_identity
from mem_param_src.runners.training import resolve_checkpoint_steps, train_run


def _tiny_gmm_config():
    """Return a fast synthetic integration-test config."""
    config = get_base_config()
    config.framework.prediction_space = "v"
    config.framework.loss_space = "v"
    config.data.dataset = "gmm_subspace"
    config.data.n_train = 8
    config.data.n_test = 8
    config.data.data_dim = 4
    config.data.ambient_dim = 4
    config.data.intrinsic_dim = 2
    config.data.n_clusters = 2
    config.network.hidden_dim = 8
    config.network.n_layers = 1
    config.network.time_emb_dim = 4
    config.training.n_training_steps = 2
    config.training.batch_cap = 4
    config.training.ckpt_every = 1
    config.training.log_every = 1
    config.training.show_progress = False
    config.evaluation.n_samples = 4
    config.evaluation.sample_bank_size = 4
    config.evaluation.batch_size = 2
    config.evaluation.chunk_size = 4
    config.evaluation.sliced_w2_n_directions = 4
    config.sampling.n_steps = 2
    config.sampling.output = "x_1"
    config.runtime.pin_memory = False
    return config


def _tiny_image_config():
    """Return a fast image integration-test config."""
    config = get_base_config()
    config.framework.prediction_space = "v"
    config.framework.loss_space = "v"
    config.data.dataset = "mnist"
    config.data.image_size = 8
    config.data.image_channels = 1
    config.network.arch = "unet"
    config.network.base_channels = 8
    config.network.channel_mult = (1,)
    config.network.attn_resolutions = ()
    config.network.num_res_blocks = 1
    config.network.time_emb_dim = 8
    config.training.n_training_steps = 1
    config.training.batch_cap = 2
    config.training.ckpt_every = 1
    config.training.log_every = 1
    config.training.show_progress = False
    config.evaluation.n_samples = 2
    config.evaluation.sample_bank_size = 2
    config.evaluation.batch_size = 1
    config.evaluation.chunk_size = 2
    config.sampling.n_steps = 1
    config.sampling.output = "x_1"
    config.runtime.pin_memory = False
    return config


def _tiny_image_bundle():
    """Return deterministic image tensors without reading an external dataset."""
    generator = torch.Generator().manual_seed(41)
    train_source_id = "fixture:train"
    reference_source_id = "fixture:reference"
    return {
        "train": torch.randn(4, 1, 8, 8, generator=generator),
        "reference": torch.randn(3, 1, 8, 8, generator=generator),
        "train_labels": torch.tensor([0, 1, 0, 1]),
        "reference_labels": torch.tensor([0, 1, 0]),
        "train_indices": torch.arange(4),
        "reference_indices": torch.arange(3),
        "train_sample_ids": [0, 1, 2, 3],
        "reference_sample_ids": [0, 1, 2],
        "train_sample_id_kind": "index",
        "reference_sample_id_kind": "index",
        "train_source_size": 4,
        "reference_source_size": 3,
        "train_source_catalog_sha256": source_catalog_sha256(train_source_id, range(4)),
        "reference_source_catalog_sha256": source_catalog_sha256(
            reference_source_id, range(3)
        ),
        "train_source_id": train_source_id,
        "reference_source_id": reference_source_id,
        "train_split": "train",
        "reference_split": "reference",
        "train_role": "train",
        "reference_role": "heldout_reference",
        "display_shape": (1, 8, 8),
    }


class TestRunnerConfiguration(unittest.TestCase):
    def test_cli_overrides_are_typed_and_strict(self):
        config = get_base_config()
        overrides = parse_cli_overrides(
            [
                "training.n_training_steps=7",
                "sampling.use_safe_division=false",
                "framework.prediction_space=v",
                "framework.loss_space=v",
                "framework.loss_weighting=x_equivalent",
                "network.channel_mult=[1, 2]",
                "evaluation.f_mem_threshold=0.3",
            ]
        )
        apply_overrides(config, overrides)

        self.assertEqual(config.training.n_training_steps, 7)
        self.assertIs(config.sampling.use_safe_division, False)
        self.assertEqual(config.framework.prediction_space, "v")
        self.assertEqual(config.framework.loss_space, "v")
        self.assertEqual(config.framework.loss_weighting, "x_equivalent")
        self.assertEqual(config.network.channel_mult, (1, 2))
        self.assertEqual(config.evaluation.f_mem_threshold, 0.3)
        with self.assertRaisesRegex(ValueError, "Unknown configuration field"):
            apply_overrides(config, {"training.unknown": 1})
        with self.assertRaisesRegex(ValueError, "Unknown configuration field"):
            apply_overrides(config, {"evaluation.ratio_threshold": 0.3})

    def test_evaluation_config_rejects_training_sections(self):
        with tempfile.TemporaryDirectory() as temporary:
            allowed_path = Path(temporary) / "allowed.py"
            allowed_path.write_text(
                "from ml_collections import ConfigDict\n"
                "def get_config():\n"
                "    config = ConfigDict()\n"
                "    config.sampling = ConfigDict()\n"
                "    config.sampling.n_steps = 3\n"
                "    return config\n",
                encoding="utf-8",
            )
            forbidden_path = Path(temporary) / "forbidden.py"
            forbidden_path.write_text(
                "from ml_collections import ConfigDict\n"
                "def get_config():\n"
                "    config = ConfigDict()\n"
                "    config.training = ConfigDict()\n"
                "    config.training.seed = 5\n"
                "    return config\n",
                encoding="utf-8",
            )

            loaded = load_python_config(
                allowed_path,
                allowed_sections=("sampling", "evaluation"),
            )
            self.assertEqual(loaded.sampling.n_steps, 3)
            with self.assertRaisesRegex(ValueError, "forbidden sections"):
                load_python_config(
                    forbidden_path,
                    allowed_sections=("sampling", "evaluation"),
                )

    def test_domain_runners_accept_every_current_loader(self):
        config = get_base_config()
        for dataset in (
            "mnist",
            "cifar10",
            "celeba32_grey",
            "flowers102",
            "afhq_v2",
            "ffhq128",
        ):
            config.data.dataset = dataset
            validate_run_kind(config, "image")
        for dataset in ("gmm_subspace", "torus3d"):
            config.data.dataset = dataset
            validate_run_kind(config, "synthetic")


class TestDataPartitionIdentity(unittest.TestCase):
    def test_seeded_membership_is_repeatable_and_seed_sensitive(self):
        first = seeded_indices(20, 6, 3)
        repeat = seeded_indices(20, 6, 3)
        changed = seeded_indices(20, 6, 4)

        self.assertTrue(torch.equal(first, repeat))
        self.assertFalse(torch.equal(first, changed))

    def test_relative_paths_and_catalog_hash_preserve_ordered_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            paths = [root / "class_a" / "a.png", root / "class_b" / "b.png"]
            sample_ids = relative_path_sample_ids(paths, root)

        self.assertEqual(sample_ids, ["class_a/a.png", "class_b/b.png"])
        first = source_catalog_sha256("fixture:all", sample_ids)
        repeat = source_catalog_sha256("fixture:all", sample_ids)
        reordered = source_catalog_sha256("fixture:all", reversed(sample_ids))
        self.assertEqual(first, repeat)
        self.assertNotEqual(first, reordered)

    def test_same_source_partitions_require_disjoint_aligned_ids(self):
        config = get_base_config()
        config.data.allow_same_source_reference = True
        source_id = "fixture:all"
        source_hash = source_catalog_sha256(source_id, range(6))
        bundle = {
            "train": torch.tensor([[0.0], [1.0]]),
            "reference": torch.tensor([[2.0], [3.0]]),
            "train_labels": torch.tensor([0, 1]),
            "reference_labels": torch.tensor([0, 1]),
            "train_indices": torch.tensor([0, 1]),
            "reference_indices": torch.tensor([2, 3]),
            "train_sample_ids": [0, 1],
            "reference_sample_ids": [2, 3],
            "train_sample_id_kind": "index",
            "reference_sample_id_kind": "index",
            "train_source_size": 6,
            "reference_source_size": 6,
            "train_source_catalog_sha256": source_hash,
            "reference_source_catalog_sha256": source_hash,
            "train_source_id": source_id,
            "reference_source_id": source_id,
        }

        self.assertIs(validate_data_bundle(bundle, config), bundle)
        bundle["reference_sample_ids"] = [1, 3]
        with self.assertRaisesRegex(ValueError, "sample identifiers overlap"):
            validate_finite_data_identity(bundle)


class TestRunnerArtifacts(unittest.TestCase):
    def test_run_and_evaluation_directories_never_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            prepared = prepare_run_dir(run_dir)

            self.assertTrue((prepared / "checkpoints").is_dir())
            self.assertTrue((prepared / "evaluations").is_dir())
            with self.assertRaises(FileExistsError):
                prepare_run_dir(run_dir)

            first_eval = prepare_evaluation_dir(run_dir, "default")
            self.assertTrue(first_eval.is_dir())
            with self.assertRaises(FileExistsError):
                prepare_evaluation_dir(run_dir, "default")
            with self.assertRaisesRegex(ValueError, "path-safe"):
                prepare_evaluation_dir(run_dir, "../outside")

    def test_checkpoint_schedules_include_required_endpoints(self):
        config = get_base_config()
        config.training.n_training_steps = 5
        config.training.ckpt_every = 2
        self.assertEqual(resolve_checkpoint_steps(config), (0, 2, 4, 5))

        config.training.ckpt_steps = (1, 3)
        self.assertEqual(resolve_checkpoint_steps(config), (0, 1, 3, 5))

        config.training.ckpt_steps = ()
        config.training.ckpt_schedule = "bonnaire-1m"
        self.assertEqual(resolve_checkpoint_steps(config), (0, 5))

        config.training.n_training_steps = 1_000_000
        bonnaire_steps = resolve_checkpoint_steps(config)
        self.assertEqual(len(bonnaire_steps), 72)
        self.assertEqual(bonnaire_steps[:3], (0, 250, 500))
        self.assertEqual(bonnaire_steps[-1], 1_000_000)


class TestRunnerCommands(unittest.TestCase):
    def test_console_entry_points_return_success_codes(self):
        repo_root = Path(__file__).resolve().parents[1]
        run_dir = repo_root / "outputs" / "unused_test_run"
        config_path = repo_root / "configs" / "runners" / "gmm_smoke.py"

        with (
            patch(
                "mem_param_src.runners.synthetic_train.train_run",
                return_value=run_dir,
            ),
            patch("mem_param_src.runners.synthetic_train.evaluate_synthetic_run"),
        ):
            train_code = synthetic_train_main(
                [
                    "--config",
                    str(config_path),
                    "--run-dir",
                    str(run_dir),
                    "--evaluate-after-training",
                ]
            )
        with patch(
            "mem_param_src.runners.synthetic_eval.evaluate_synthetic_run",
            return_value=run_dir / "evaluations" / "default",
        ):
            eval_code = synthetic_eval_main(["--run-dir", str(run_dir)])

        self.assertEqual(train_code, 0)
        self.assertEqual(eval_code, 0)

    def test_image_command_resolves_wandb_and_monitor_flags(self):
        repo_root = Path(__file__).resolve().parents[1]
        run_dir = repo_root / "outputs" / "unused_image_test_run"
        config_path = repo_root / "configs" / "runners" / "flowers102_smoke.py"

        with patch(
            "mem_param_src.runners.image_train.train_run",
            return_value=run_dir,
        ) as mocked_train:
            code = image_train_main(
                [
                    "--config",
                    str(config_path),
                    "--run-dir",
                    str(run_dir),
                    "--wandb",
                    "--generation-monitor",
                    "--f-mem-monitor",
                    "--f-mem-samples",
                    "7",
                    "--training-grid-samples",
                    "3",
                    "--save-generated-samples",
                ]
            )

        config = mocked_train.call_args.args[0]
        self.assertEqual(code, 0)
        self.assertIs(config.tracking.wandb.enabled, True)
        self.assertEqual(config.tracking.wandb.mode, "online")
        self.assertIs(config.generation_monitor.enabled, True)
        self.assertIs(config.random_projected_f_mem.enabled, True)
        self.assertEqual(config.random_projected_f_mem.n_samples, 7)
        self.assertEqual(config.generation_monitor.n_train_samples, 3)
        self.assertIs(config.evaluation.save_generated_samples, True)

    def test_image_sampling_flags_keep_training_division_independent(self):
        repo_root = Path(__file__).resolve().parents[1]
        config_path = repo_root / "configs/runners/celeba32_eps_eps_monitor.py"
        base_args = ["--config", str(config_path), "--run-dir", "unused"]
        with patch("mem_param_src.runners.image_train.train_run") as train:
            image_train_main(base_args + [
                "--set", "sampling.n_steps=10",
                "--set", "training.ckpt_steps=[0, 1]",
                "--steps", "1200", "--lr", "0.0002", "--seed", "9",
                "--batch-size", "8", "--checkpoint-every", "300",
                "--sampler", "heun", "--sampling-steps", "200",
                "--sampling-safe-division", "--sampling-denom-eps", "0.05",
                "--sampling-start-time", "0.02", "--sampling-end-time", "0.98",
                "--sampling-output", "x_t_max",
            ])
        config = train.call_args.args[0]
        self.assertEqual(config.training.n_training_steps, 1200)
        self.assertEqual(config.training.lr, 0.0002)
        self.assertEqual(config.training.seed, 9)
        self.assertEqual(config.training.batch_cap, 8)
        self.assertEqual(resolve_checkpoint_steps(config), (0, 300, 600, 900, 1200))
        self.assertEqual(config.sampling.method, "heun")
        self.assertEqual(config.sampling.n_steps, 200)
        self.assertIs(config.sampling.use_safe_division, True)
        self.assertEqual(config.sampling.safe_div_denom_eps, 0.05)
        self.assertEqual(config.sampling.start_time, 0.02)
        self.assertEqual(config.sampling.end_time, 0.98)
        self.assertEqual(config.sampling.output, "x_t_max")
        self.assertIs(config.training.use_safe_division, False)
        self.assertEqual(config.training.safe_div_denom_eps, 0.01)
        self.assertEqual(config.framework.loss_weighting, "unit")

        with patch("mem_param_src.runners.image_train.train_run") as train:
            image_train_main(base_args + [
                "--set", "sampling.use_safe_division=true",
                "--no-sampling-safe-division",
            ])
        self.assertIs(train.call_args.args[0].sampling.use_safe_division, False)


class TestSyntheticRunnerIntegration(unittest.TestCase):
    def test_saved_run_supports_standalone_deterministic_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "synthetic_run"
            train_run(
                _tiny_gmm_config(),
                run_dir,
                run_kind="synthetic",
                device="cpu",
            )

            self.assertTrue((run_dir / "config.json").is_file())
            self.assertTrue((run_dir / "data_bundle.pt").is_file())
            self.assertTrue((run_dir / "data_manifest.json").is_file())
            self.assertTrue((run_dir / "z_init.pt").is_file())
            self.assertTrue((run_dir / "sliced_directions.pt").is_file())
            self.assertEqual([step for step, _ in list_checkpoints(run_dir)], [0, 1, 2])
            bundle = load_torch(run_dir / "data_bundle.pt")
            manifest = validate_data_manifest(
                run_dir,
                bundle,
                run_kind="synthetic",
                verify_hashes=True,
            )
            self.assertEqual(
                manifest["artifact_format_version"], ARTIFACT_FORMAT_VERSION
            )

            with patch(
                "mem_param_src.datasets.build_data_bundle",
                side_effect=AssertionError("evaluation rebuilt the dataset"),
            ):
                default_dir = evaluate_synthetic_run(run_dir, device="cpu")
                repeat_dir = evaluate_synthetic_run(
                    run_dir,
                    eval_name="repeat",
                    device="cpu",
                )

            default_samples = load_torch(default_dir / "final_samples_x_1.pt")
            repeat_samples = load_torch(repeat_dir / "final_samples_x_1.pt")
            self.assertTrue(torch.equal(default_samples, repeat_samples))
            self.assertEqual(
                read_json(default_dir / "checkpoint_metrics.json"),
                read_json(repeat_dir / "checkpoint_metrics.json"),
            )

            eval_config = Path(temporary) / "exact_eval.py"
            eval_config.write_text(
                "from ml_collections import ConfigDict\n"
                "def get_config():\n"
                "    config = ConfigDict()\n"
                "    config.sampling = ConfigDict()\n"
                "    config.sampling.use_safe_division = False\n"
                "    return config\n",
                encoding="utf-8",
            )
            named_dir = evaluate_synthetic_run(
                run_dir,
                eval_name="x_t_max",
                eval_config=eval_config,
                override_items=(
                    "sampling.use_safe_division=true",
                    "sampling.output=x_t_max",
                ),
                device="cpu",
            )
            settings = read_json(named_dir / "settings.json")
            self.assertIs(settings["sampling"]["use_safe_division"], True)
            self.assertEqual(settings["sampling"]["output"], "x_t_max")
            self.assertTrue((named_dir / "final_samples_x_t_max.pt").is_file())

            with self.assertRaisesRegex(ValueError, "Provide --eval-name"):
                evaluate_synthetic_run(
                    run_dir,
                    override_items=("sampling.output=x_t_max",),
                    device="cpu",
                )
            with self.assertRaises(FileExistsError):
                evaluate_synthetic_run(
                    run_dir,
                    eval_name="repeat",
                    device="cpu",
                )


class TestImageRunnerIntegration(unittest.TestCase):
    def test_image_runner_uses_the_shared_saved_run_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "image_run"
            with patch(
                "mem_param_src.runners.training.build_data_bundle",
                return_value=_tiny_image_bundle(),
            ):
                train_run(
                    _tiny_image_config(),
                    run_dir,
                    run_kind="image",
                    device="cpu",
                )

            eval_dir = evaluate_image_run(run_dir, device="cpu")
            self.assertFalse((eval_dir / "generated_samples").exists())
            records = read_json(eval_dir / "checkpoint_metrics.json")
            samples = load_torch(eval_dir / "final_samples_x_1.pt")
            bundle = load_torch(run_dir / "data_bundle.pt")
            manifest = validate_data_manifest(
                run_dir,
                bundle,
                run_kind="image",
                verify_hashes=True,
            )

            self.assertEqual([record["step"] for record in records], [0, 1])
            self.assertTrue(all("f_mem" in record for record in records))
            self.assertEqual(tuple(samples.shape), (2, 1, 8, 8))
            self.assertTrue(torch.isfinite(samples).all())
            self.assertEqual(
                manifest["partitions"]["train"]["sample_ids"], [0, 1, 2, 3]
            )
            self.assertEqual(
                manifest["partitions"]["train"]["tensor"]["sha256"],
                tensor_sha256(bundle["train"]),
            )

            # A legacy saved config must accept the new optional replay setting.
            saved_config = read_json(run_dir / "config.json")
            del saved_config["evaluation"]["save_generated_samples"]
            write_json(run_dir / "config.json", saved_config)
            saved_eval = evaluate_image_run(
                run_dir,
                eval_name="with_samples",
                override_items=("evaluation.save_generated_samples=true",),
                device="cpu",
            )
            for step in (0, 1):
                saved_samples = load_torch(
                    saved_eval / "generated_samples" / f"step_{step:06d}.pt"
                )
                self.assertEqual(tuple(saved_samples.shape), (2, 1, 8, 8))
                self.assertTrue(torch.isfinite(saved_samples).all())
            self.assertTrue(torch.equal(saved_samples, samples))


class TestImageTrainingMonitoring(unittest.TestCase):
    def test_wandb_identity_uses_comparison_fields(self):
        config = _tiny_image_config()
        config.study.dataset_key = "mnist"
        config.data.n_train = 4
        identity = resolve_wandb_identity(config)

        self.assertEqual(identity["project"], "mnist-unet")
        self.assertEqual(
            identity["name"],
            "mnist-unet-N4-B2-v_v-steps1-s0",
        )
        self.assertEqual(identity["group"], "mnist-unet-steps1")

    def test_wandb_logs_loss_and_fixed_noise_checkpoint_grids(self):
        config = _tiny_image_config()
        config.study.dataset_key = "mnist"
        config.training.n_training_steps = 2
        config.training.ckpt_every = 1
        config.tracking.wandb.enabled = True
        config.tracking.wandb.mode = "offline"
        config.generation_monitor.enabled = True
        config.generation_monitor.n_samples = 2
        config.monitoring.batch_size = 1

        fake_run = MagicMock()
        fake_run.id = "test-run-id"
        fake_run.url = "https://wandb.invalid/test-run-id"
        fake_wandb = MagicMock()
        fake_wandb.init.return_value = fake_run
        fake_wandb.Image.side_effect = lambda path: {"image_path": str(path)}
        seen_noise = []

        def generate_samples(model, z_init, resolved_config, device):
            count = int(resolved_config.generation_monitor.n_samples)
            seen_noise.append(z_init[:count].clone())
            return torch.zeros(count, 1, 8, 8)

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "image_wandb_run"
            with (
                patch(
                    "mem_param_src.runners.training.build_data_bundle",
                    return_value=_tiny_image_bundle(),
                ),
                patch(
                    "mem_param_src.runners.tracking.importlib.import_module",
                    return_value=fake_wandb,
                ),
                patch(
                    "mem_param_src.runners.image_monitoring.generate_monitor_samples",
                    side_effect=generate_samples,
                ),
            ):
                train_run(config, run_dir, run_kind="image", device="cpu")

            self.assertTrue(
                (run_dir / "monitoring" / "samples_step_000000.png").is_file()
            )
            self.assertTrue(
                (run_dir / "monitoring" / "samples_step_000001.png").is_file()
            )
            self.assertTrue(
                (run_dir / "monitoring" / "samples_step_000002.png").is_file()
            )
            self.assertEqual(len(seen_noise), 3)
            self.assertFalse((run_dir / "monitoring/generated_samples").exists())
            self.assertTrue(
                all(torch.equal(seen_noise[0], item) for item in seen_noise)
            )

            payloads = [call.args[0] for call in fake_run.log.call_args_list]
            train_payloads = [item for item in payloads if "train/loss" in item]
            generation_payloads = [
                item for item in payloads if "generation/samples" in item
            ]
            self.assertEqual([item["train/step"] for item in train_payloads], [1, 2])
            self.assertEqual(
                [item["generation/step"] for item in generation_payloads],
                [0, 1, 2],
            )
            init_config = fake_wandb.init.call_args.kwargs["config"]
            self.assertEqual(
                init_config["training"]["resolved_ckpt_steps"],
                [0, 1, 2],
            )
            run_record = read_json(run_dir / "run.json")
            self.assertIs(run_record["wandb"]["active"], False)
            self.assertIs(run_record["wandb"]["success"], True)
            fake_run.finish.assert_called_once_with(exit_code=0)

    def test_f_mem_monitor_reuses_samples_and_logs_training_matches(self):
        config = _tiny_image_config()
        config.training.n_training_steps = 2
        config.generation_monitor.enabled = True
        config.generation_monitor.n_samples = 2
        config.generation_monitor.n_train_samples = 3
        config.evaluation.save_generated_samples = True
        config.random_projected_f_mem.enabled = True
        config.random_projected_f_mem.n_samples = 4
        config.random_projected_f_mem.resize_size = 8
        config.random_projected_f_mem.projection_dim = 16
        config.tracking.wandb.enabled = True
        fake_run = MagicMock()
        fake_wandb = MagicMock()
        fake_run.id = "f_mem_test"
        fake_run.url = "https://wandb.invalid/f_mem_test"
        fake_wandb.init.return_value = fake_run
        fake_wandb.Image.side_effect = str
        bundle = _tiny_image_bundle()
        seen_noise = []

        def sample(model, z_init, resolved_config, device):
            if len(seen_noise) == 2:
                # Earlier records and curves must be available before training ends.
                self.assertEqual(read_json(run_dir / "train_history.json")[-1]["step"], 2)
                self.assertTrue((run_dir / "monitoring/plots/loss.png").is_file())
                self.assertTrue((run_dir / "monitoring/plots/f_mem.png").is_file())
            seen_noise.append(z_init.clone())
            # Exact copies must count as memorized against training, not held-out data.
            return bundle["train"][:4].clone()

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "f_mem_run"
            with (
                patch(
                    "mem_param_src.runners.training.build_data_bundle",
                    return_value=bundle,
                ),
                patch(
                    "mem_param_src.runners.tracking.importlib.import_module",
                    return_value=fake_wandb,
                ),
                patch(
                    "mem_param_src.runners.image_monitoring.generate_monitor_samples",
                    side_effect=sample,
                ) as generate,
                patch(
                    "mem_param_src.runners.image_plotting.save_training_comparison_grid",
                    wraps=save_training_comparison_grid,
                ) as grid,
            ):
                train_run(config, run_dir, run_kind="image", device="cpu")
            self.assertEqual(generate.call_count, 3)
            self.assertTrue(all(torch.equal(seen_noise[0], z) for z in seen_noise))
            self.assertEqual(load_torch(run_dir / "z_init.pt").shape[0], 4)
            self.assertTrue(
                all(torch.equal(c.args[0], bundle["train"][:3]) for c in grid.call_args_list)
            )
            self.assertTrue(all(c.args[1].shape[0] == 2 for c in grid.call_args_list))
            saved_config = read_json(run_dir / "config.json")
            self.assertEqual(
                saved_config["random_projected_f_mem"]["projection_seed"],
                config.training.seed + 3001,
            )
            self.assertTrue((run_dir / "monitoring/f_mem_projection.pt").is_file())
            self.assertTrue((run_dir / "monitoring/f_mem_train_vectors.pt").is_file())
            for step in (0, 1, 2):
                saved_samples = load_torch(
                    run_dir / "monitoring/generated_samples" / f"step_{step:06d}.pt"
                )
                self.assertTrue(torch.equal(saved_samples, bundle["train"][:4]))
                self.assertTrue(
                    (run_dir / f"monitoring/comparison_grids/step_{step:06d}.png").is_file()
                )
                record = read_json(run_dir / f"monitoring/f_mem_step_{step:06d}.json")
                self.assertEqual(record["n_samples"], 4)
                self.assertEqual(record["N"], len(bundle["train"]))
                self.assertEqual(record["f_mem"], 1.0)
                self.assertEqual(record["rho"], 0.3)
                self.assertEqual(record["space"], "random_projected_pixels")
            payloads = [c.args[0] for c in fake_run.log.call_args_list]
            metrics = [
                p for p in payloads if "generation/random_projected_pixels/f_mem" in p
            ]
            self.assertEqual([p["generation/step"] for p in metrics], [0, 1, 2])
            self.assertTrue(
                all(
                    p["generation/random_projected_pixels/f_mem"] == 1.0
                    for p in metrics
                )
            )

    def test_f_mem_monitor_requires_generation_and_positive_count(self):
        config = _tiny_image_config()
        config.random_projected_f_mem.enabled = True
        with self.assertRaisesRegex(ValueError, "requires image generation"):
            train_run(config, "unused", run_kind="image", device="cpu")
        config.generation_monitor.enabled = True
        config.random_projected_f_mem.n_samples = 0
        with self.assertRaisesRegex(ValueError, "n_samples must be"):
            train_run(config, "unused", run_kind="image", device="cpu")

    def test_training_grid_requires_monitoring_and_nonnegative_count(self):
        config = _tiny_image_config()
        config.generation_monitor.n_train_samples = 8
        with self.assertRaisesRegex(ValueError, "require image generation"):
            train_run(config, "unused", run_kind="image", device="cpu")
        config.generation_monitor.n_train_samples = -1
        with self.assertRaisesRegex(ValueError, "must be nonnegative"):
            train_run(config, "unused", run_kind="image", device="cpu")

    def test_shared_sampler_uses_larger_f_mem_count(self):
        config = _tiny_image_config()
        config.generation_monitor.n_samples = 2
        config.random_projected_f_mem.enabled = True
        config.random_projected_f_mem.n_samples = 5
        config.monitoring.batch_size = 2
        model = MagicMock()
        model.training = True
        model.sample.side_effect = lambda z, **kwargs: z.clone()
        noise = torch.randn(5, 1, 8, 8)
        samples = generate_monitor_samples(model, noise, config, "cpu")
        self.assertTrue(torch.equal(samples, noise))
        self.assertEqual(
            [c.args[0].shape[0] for c in model.sample.call_args_list], [2, 2, 1]
        )

    def test_monitor_flag_off_does_not_sample(self):
        config = _tiny_image_config()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "image_without_monitor"
            with (
                patch(
                    "mem_param_src.runners.training.build_data_bundle",
                    return_value=_tiny_image_bundle(),
                ),
                patch(
                    "mem_param_src.runners.training._record_image_monitor",
                    side_effect=AssertionError("monitor should remain disabled"),
                ),
            ):
                train_run(config, run_dir, run_kind="image", device="cpu")

            self.assertFalse((run_dir / "monitoring").exists())

    def test_monitor_restores_training_mode_and_rejects_nonfinite_samples(self):
        config = _tiny_image_config()
        config.generation_monitor.n_samples = 2
        config.monitoring.batch_size = 1
        z_init = torch.randn(2, 1, 8, 8)

        model = MagicMock()
        model.training = True
        model.sample.side_effect = lambda z, **kwargs: torch.zeros_like(z)
        model.eval.side_effect = lambda: setattr(model, "training", False)
        model.train.side_effect = lambda mode=True: setattr(model, "training", mode)
        samples = generate_monitor_samples(model, z_init, config, torch.device("cpu"))

        self.assertEqual(tuple(samples.shape), (2, 1, 8, 8))
        self.assertIs(model.training, True)
        sampled_noise = [call.args[0].cpu() for call in model.sample.call_args_list]
        self.assertTrue(torch.equal(torch.cat(sampled_noise), z_init))

        model.sample.reset_mock()
        model.sample.side_effect = lambda z, **kwargs: torch.full_like(z, float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            generate_monitor_samples(model, z_init, config, torch.device("cpu"))
        self.assertIs(model.training, True)

    def test_wandb_failure_policy_and_synthetic_scope(self):
        config = _tiny_image_config()
        config.tracking.wandb.enabled = True
        config.tracking.wandb.mode = "offline"
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "nonstrict_wandb_failure"
            with (
                patch(
                    "mem_param_src.runners.training.build_data_bundle",
                    return_value=_tiny_image_bundle(),
                ),
                patch(
                    "mem_param_src.runners.tracking.importlib.import_module",
                    side_effect=ModuleNotFoundError("wandb unavailable"),
                ),
            ):
                train_run(config, run_dir, run_kind="image", device="cpu")
            run_record = read_json(run_dir / "run.json")
            self.assertEqual(run_record["state"], "trained")
            self.assertIn("wandb unavailable", run_record["wandb"]["warning"])

        image_failure = _tiny_image_config()
        image_failure.tracking.wandb.enabled = True
        image_failure.tracking.wandb.mode = "offline"
        image_failure.generation_monitor.enabled = True
        image_failure.generation_monitor.n_samples = 2
        image_failure.monitoring.batch_size = 1
        fake_run = MagicMock()
        fake_run.id = "image-failure"
        fake_run.url = None
        fake_wandb = MagicMock()
        fake_wandb.init.return_value = fake_run
        fake_wandb.Image.side_effect = RuntimeError("image encode failed")
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "nonstrict_wandb_image_failure"
            with (
                patch(
                    "mem_param_src.runners.training.build_data_bundle",
                    return_value=_tiny_image_bundle(),
                ),
                patch(
                    "mem_param_src.runners.tracking.importlib.import_module",
                    return_value=fake_wandb,
                ),
                patch(
                    "mem_param_src.runners.image_monitoring.generate_monitor_samples",
                    return_value=torch.zeros(2, 1, 8, 8),
                ),
            ):
                train_run(image_failure, run_dir, run_kind="image", device="cpu")
            run_record = read_json(run_dir / "run.json")
            self.assertEqual(run_record["state"], "trained")
            self.assertIn("image encode failed", run_record["wandb"]["warning"])
            self.assertEqual(len(read_json(run_dir / "train_history.json")), 1)

        strict = _tiny_image_config()
        strict.tracking.wandb.enabled = True
        strict.tracking.wandb.mode = "offline"
        strict.tracking.wandb.strict = True
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "strict_wandb_failure"
            with (
                patch(
                    "mem_param_src.runners.training.build_data_bundle",
                    return_value=_tiny_image_bundle(),
                ),
                patch(
                    "mem_param_src.runners.tracking.importlib.import_module",
                    side_effect=ModuleNotFoundError("wandb unavailable"),
                ),
                self.assertRaisesRegex(ModuleNotFoundError, "wandb unavailable"),
            ):
                train_run(strict, run_dir, run_kind="image", device="cpu")
            self.assertEqual(read_json(run_dir / "run.json")["state"], "failed")

        synthetic = _tiny_gmm_config()
        synthetic.tracking.wandb.enabled = True
        with self.assertRaisesRegex(ValueError, "image runs only"):
            train_run(
                synthetic,
                Path(tempfile.gettempdir()) / "unused_synthetic_wandb_run",
                run_kind="synthetic",
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
