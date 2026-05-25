import importlib
import json
import math
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import torch
import torch.nn.functional as F

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

eval_dataset = importlib.import_module("eval_dataset")


class EvaluationAccumulatorTest(unittest.TestCase):
    def test_metrics_are_row_weighted(self):
        logits = torch.tensor([[0.0], [math.log(3.0)], [math.log(7.0)]])
        labels = torch.tensor([[0.25], [0.75], [0.5]])

        combined = eval_dataset.EvaluationAccumulator()
        combined.update(logits, labels)

        split = eval_dataset.EvaluationAccumulator()
        split.update(logits[:1], labels[:1])
        split.update(logits[1:], labels[1:])

        self.assertEqual(combined.rows, 3)
        self.assertEqual(split.rows, 3)
        for key, value in combined.metrics().items():
            self.assertAlmostEqual(value, split.metrics()[key], delta=1e-4)

        expected_bce = F.binary_cross_entropy_with_logits(
            logits.view(-1), labels.view(-1), reduction="sum"
        ).item() / 3
        self.assertAlmostEqual(combined.metrics()["bce_loss"], expected_bce)

    def test_calibration_bins_include_exact_one_in_last_bin(self):
        accumulator = eval_dataset.EvaluationAccumulator()
        logits = torch.tensor([[-1000.0], [0.0], [1000.0]])
        labels = torch.tensor([[0.0], [0.5], [1.0]])

        accumulator.update(logits, labels)
        bins = accumulator.calibration_bins()

        self.assertEqual(bins[0]["count"], 1)
        self.assertEqual(bins[5]["count"], 1)
        self.assertEqual(bins[9]["count"], 1)
        self.assertIsNone(bins[1]["pred_mean"])

    def test_probability_to_centipawn_clamps_extreme_values(self):
        probabilities = torch.tensor([0.0, 0.5, 1.0])

        cps = eval_dataset.probability_to_centipawn(probabilities)

        self.assertTrue(torch.isfinite(cps).all())
        self.assertAlmostEqual(cps[1].item(), 0.0)
        self.assertLess(cps[0].item(), -2000.0)
        self.assertGreater(cps[2].item(), 2000.0)

    def test_labels_must_be_probabilities(self):
        accumulator = eval_dataset.EvaluationAccumulator()

        with self.assertRaisesRegex(ValueError, "probabilities"):
            accumulator.update(torch.tensor([[0.0]]), torch.tensor([[1.5]]))


class EvaluationReportContractTest(unittest.TestCase):
    def test_selection_defaults_to_one_million_prefix_rows(self):
        selection = eval_dataset.resolve_selection(
            dataset_rows=2_000_000,
            rows=None,
            full=False,
            seed=7,
        )

        self.assertEqual(selection.evaluated_rows, 1_000_000)
        self.assertEqual(selection.selection, "deterministic-prefix")
        self.assertEqual(selection.seed, 7)

    def test_selection_rejects_rows_with_full(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            eval_dataset.resolve_selection(
                dataset_rows=10,
                rows=5,
                full=True,
                seed=0,
            )

    def test_resolve_dataset_path_supports_test_split(self):
        self.assertEqual(
            eval_dataset.resolve_dataset_path("test", None),
            Path("data/processed/test.chesseval"),
        )

    def test_missing_test_dataset_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pth"
            missing_test_path = Path(tmpdir) / "test.chesseval"
            checkpoint_path.write_bytes(b"placeholder")

            with mock.patch.object(
                eval_dataset,
                "checkpoint_path",
                return_value=checkpoint_path,
            ), mock.patch.object(
                eval_dataset,
                "TEST_DATA_PATH",
                missing_test_path,
            ):
                with self.assertRaises(FileNotFoundError) as context:
                    eval_dataset.run_eval_dataset("model", split="test")

        self.assertIn("test.chesseval not found", str(context.exception))

    def test_report_has_stable_top_level_shape_and_validation_warning(self):
        accumulator = eval_dataset.EvaluationAccumulator()
        accumulator.update(torch.tensor([[0.0]]), torch.tensor([[0.5]]))
        selection = eval_dataset.EvaluationSelection(
            dataset_rows=10,
            evaluated_rows=1,
            selection="deterministic-prefix",
            seed=0,
        )

        report = eval_dataset.build_report(
            model_name="example",
            checkpoint={"epoch": 3, "best_validation_loss": float("inf")},
            model_path=Path("artifacts/checkpoints/example.pth"),
            split="validation",
            dataset_path=Path("data/processed/validation.chesseval"),
            selection=selection,
            device=torch.device("cpu"),
            batch_size=4,
            duration_seconds=1.25,
            accumulator=accumulator,
        )

        self.assertEqual(
            list(report.keys()),
            [
                "schema_version",
                "model",
                "data",
                "run",
                "metrics",
                "calibration_bins",
                "warnings",
            ],
        )
        self.assertIsNone(report["model"]["checkpoint_best_validation_loss"])
        self.assertIn(
            "validation_split_is_model_selection_data",
            report["warnings"],
        )

    def test_test_report_has_no_validation_warning(self):
        accumulator = eval_dataset.EvaluationAccumulator()
        accumulator.update(torch.tensor([[0.0]]), torch.tensor([[0.5]]))
        selection = eval_dataset.EvaluationSelection(
            dataset_rows=1,
            evaluated_rows=1,
            selection="full",
            seed=0,
        )

        report = eval_dataset.build_report(
            model_name="example",
            checkpoint={},
            model_path=Path("artifacts/checkpoints/example.pth"),
            split="test",
            dataset_path=Path("data/processed/test.chesseval"),
            selection=selection,
            device=torch.device("cpu"),
            batch_size=1,
            duration_seconds=0.0,
            accumulator=accumulator,
        )

        self.assertEqual(report["warnings"], [])

    def test_write_report_emits_strict_json(self):
        accumulator = eval_dataset.EvaluationAccumulator()
        accumulator.update(torch.tensor([[0.0]]), torch.tensor([[0.5]]))
        selection = eval_dataset.EvaluationSelection(
            dataset_rows=1,
            evaluated_rows=1,
            selection="full",
            seed=0,
        )
        report = eval_dataset.build_report(
            model_name="example",
            checkpoint={},
            model_path=Path("artifacts/checkpoints/example.pth"),
            split="train",
            dataset_path=Path("data/processed/train.chesseval"),
            selection=selection,
            device=torch.device("cpu"),
            batch_size=1,
            duration_seconds=0.0,
            accumulator=accumulator,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            eval_dataset.write_report(report, output_path)
            loaded = json.loads(output_path.read_text())

        self.assertEqual(loaded["schema_version"], 1)
        self.assertEqual(loaded["warnings"], [])


if __name__ == "__main__":
    unittest.main()
