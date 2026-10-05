"""Focused tests for the one-million-step checkpoint evaluator."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "jobs" / "berzelius" / "eps_recovery_1m" / "eval"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVALUATOR = load_module("eps_recovery_evaluator", EVAL_DIR / "evaluate_checkpoints.py")
COMBINER = load_module("eps_recovery_combiner", EVAL_DIR / "combine_results.py")


class EpsRecoveryEvaluatorTest(unittest.TestCase):
    def test_sampling_profiles_define_the_five_requested_controls(self):
        profiles = EVALUATOR.load_profiles(EVAL_DIR / "sampling_profiles.tsv")
        settings = {
            profile["profile"]: (
                profile["eps_to_v"],
                profile["use_safe_div"],
                profile["safe_div_denominator_floor"],
                profile["heun_final_euler"],
            )
            for profile in profiles
        }
        self.assertEqual(
            settings,
            {
                "old_full_heun": ("via_x", True, 0.05, False),
                "direct_full_heun": ("direct", True, 0.05, False),
                "exact_full_heun": ("direct", False, 0.05, False),
                "direct_final_euler": ("direct", True, 0.05, True),
                "exact_final_euler": ("direct", False, 0.05, True),
            },
        )

    def test_common_checkpoint_selection_applies_limit_and_explicit_list(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = []
            for index, steps in enumerate(((0, 10, 20), (0, 10, 30), (0, 10, 40))):
                run = root / str(index)
                checkpoint_dir = run / "checkpoints"
                checkpoint_dir.mkdir(parents=True)
                for step in steps:
                    (checkpoint_dir / f"step_{step:06d}.pt").touch()
                runs.append(run)

            self.assertEqual(
                EVALUATOR.select_common_steps(runs, max_step=15),
                [0, 10],
            )
            self.assertEqual(
                EVALUATOR.select_common_steps(
                    runs,
                    max_step=15,
                    requested_steps=[10],
                ),
                [10],
            )
            with self.assertRaisesRegex(ValueError, "not common"):
                EVALUATOR.select_common_steps(
                    runs,
                    max_step=15,
                    requested_steps=[20],
                )

    def test_saved_heldout_selection_never_rebuilds_or_extends_data(self):
        reference = torch.arange(24).reshape(6, 1, 2, 2)
        selected = EVALUATOR.load_saved_heldout({"reference": reference}, 4)
        self.assertTrue(torch.equal(selected, reference[:4]))
        with self.assertRaisesRegex(ValueError, "saved artifact contains 6"):
            EVALUATOR.load_saved_heldout({"reference": reference}, 7)

    def test_replay_noise_is_saved_and_matches_the_training_prefix(self):
        config = SimpleNamespace(training=SimpleNamespace(seed=0))
        train = torch.zeros(3, 1, 2, 2)
        saved = EVALUATOR.make_eval_noise(config, train, 2)
        with tempfile.TemporaryDirectory() as temporary:
            first, path = EVALUATOR.load_or_build_eval_noise(
                temporary,
                config,
                train,
                saved,
                5,
            )
            second, second_path = EVALUATOR.load_or_build_eval_noise(
                temporary,
                config,
                train,
                saved,
                5,
            )
            self.assertEqual(first.shape, (5, 1, 2, 2))
            self.assertTrue(torch.equal(first[:2], saved))
            self.assertTrue(torch.equal(first, second))
            self.assertEqual(path, second_path)
            self.assertTrue(path.is_file())

    def test_raw_sample_stats_keep_overshoot_visible(self):
        samples = torch.tensor([-2.0, -1.0, 0.0, 2.0]).reshape(1, 1, 2, 2)
        stats = EVALUATOR.raw_sample_stats(samples)
        self.assertEqual(stats["finite_fraction"], 1.0)
        self.assertEqual(stats["raw_min"], -2.0)
        self.assertEqual(stats["raw_max"], 2.0)
        self.assertEqual(stats["outside_minus1_1_fraction"], 0.5)

    def test_identical_fid_statistics_have_zero_distance(self):
        features = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]])
        mean, covariance = EVALUATOR.covariance_stats(features)
        value = EVALUATOR.fid_from_stats(mean, covariance, mean, covariance)
        self.assertAlmostEqual(value, 0.0, places=8)

    def test_sample_batches_forward_the_sampling_controls(self):
        class FakeModel:
            def __init__(self):
                self.calls = []

            def eval(self):
                return self

            def sample(self, z, **kwargs):
                self.calls.append(kwargs)
                return z

        sampling = SimpleNamespace(
            n_steps=50,
            method="heun",
            start_time=0.001,
            end_time=0.999,
            use_safe_div=True,
            safe_div_denominator_floor=0.05,
            output="x_t_max",
            eps_to_v="direct",
            heun_final_euler=False,
            final_pixel_clamp=False,
        )
        model = FakeModel()
        samples = EVALUATOR.sample_in_batches(
            model,
            torch.zeros(5, 1, 2, 2),
            SimpleNamespace(sampling=sampling),
            device=torch.device("cpu"),
            batch_size=2,
        )
        self.assertEqual(samples.shape, (5, 1, 2, 2))
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(model.calls[0]["eps_to_v"], "direct")
        self.assertTrue(model.calls[0]["use_safe_div"])
        self.assertEqual(model.calls[0]["safe_div_denominator_floor"], 0.05)
        self.assertFalse(model.calls[0]["heun_final_euler"])


class EpsRecoveryCombinerTest(unittest.TestCase):
    def make_row(self, source, profile, fid, f_mem):
        return {
            "source_index": source,
            "source_label": f"source_{source}",
            "sampling_profile": profile,
            "step": 100,
            "status": "ok",
            "fid": fid,
            "f_mem": f_mem,
            "outside_minus1_1_fraction": 0.25,
        }

    def test_selection_keeps_only_the_six_as_launched_eps_profiles(self):
        rows = [
            self.make_row(source, profile, 1.0, 0.0)
            for source, profile in COMBINER.EPS_PROFILE_BY_SOURCE.items()
        ]
        rows[3]["status"] = "nonfinite"
        rows.append(self.make_row(0, "direct_full_heun", 2.0, 0.1))
        rows.append(self.make_row(6, "saved", 3.0, 0.2))
        selected = COMBINER.select_eps_config_rows(rows)
        self.assertEqual(len(selected), 6)
        self.assertEqual({row["source_index"] for row in selected}, set(range(6)))
        self.assertIn("nonfinite", {row["status"] for row in selected})


    def test_combiner_rejects_false_fid_10k_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            task_dir = output_root / "tasks" / "00_source"
            task_dir.mkdir(parents=True)
            row = self.make_row(0, "old_full_heun", 1.0, 0.0)
            row.update(
                {
                    "sample_count": 10_000,
                    "fid_reference_count": 1_024,
                    "sampling_signature": "test",
                }
            )
            (task_dir / "checkpoint_metrics.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "fewer than 10k"):
                COMBINER.combine(output_root)


if __name__ == "__main__":
    unittest.main()
