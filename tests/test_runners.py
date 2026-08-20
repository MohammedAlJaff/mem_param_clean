"""Focused tests for runner configuration, artifacts, training, and replay."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
)
from mem_param_src.runners.evaluation import (
    evaluate_image_run,
    evaluate_synthetic_run,
)
from mem_param_src.runners.synthetic_eval import main as synthetic_eval_main
from mem_param_src.runners.synthetic_train import main as synthetic_train_main
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
                "network.channel_mult=[1, 2]",
            ]
        )
        apply_overrides(config, overrides)

        self.assertEqual(config.training.n_training_steps, 7)
        self.assertIs(config.sampling.use_safe_division, False)
        self.assertEqual(config.framework.prediction_space, "v")
        self.assertEqual(config.network.channel_mult, (1, 2))
        with self.assertRaisesRegex(ValueError, "Unknown configuration field"):
            apply_overrides(config, {"training.unknown": 1})

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


if __name__ == "__main__":
    unittest.main()
