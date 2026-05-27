import importlib
import math
import sys
import unittest
from pathlib import Path

import torch

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

analysis = importlib.import_module("analyze_parallel_fusion")


class ParallelFusionDiagnosticUtilityTest(unittest.TestCase):
    def test_tensor_stats_report_mean_std_and_bounds(self):
        stats = analysis.TensorStats()

        stats.update(torch.tensor([[1.0, -2.0], [3.0, 0.0]]))
        report = stats.report()

        self.assertEqual(report["count"], 4)
        self.assertAlmostEqual(report["mean"], 0.5)
        self.assertAlmostEqual(report["mean_abs"], 1.5)
        self.assertAlmostEqual(report["rms"], math.sqrt(14.0 / 4.0))
        self.assertEqual(report["min"], -2.0)
        self.assertEqual(report["max"], 3.0)

    def test_bucket_accumulator_reports_parallel_delta(self):
        buckets = analysis.BucketAccumulator((0.0, 0.5, 1.0))

        buckets.update(
            torch.tensor([0.25, 0.75]),
            torch.tensor([0.4, 0.2]),
            torch.tensor([0.5, 0.1]),
        )
        report = buckets.report()

        self.assertEqual(report[0]["count"], 1)
        self.assertAlmostEqual(report[0]["parallel_minus_baseline_bce"], 0.1)
        self.assertEqual(report[1]["count"], 1)
        self.assertAlmostEqual(report[1]["parallel_minus_baseline_bce"], -0.1)

    def test_top_examples_handles_tied_loss_deltas(self):
        examples = analysis.TopExamples(limit=2)

        examples.update(
            start_index=10,
            labels=torch.tensor([[0.5], [0.5]]),
            material_diff=torch.tensor([0.0, 1.0]),
            baseline_logits=torch.tensor([[0.0], [0.0]]),
            parallel_logits=torch.tensor([[0.0], [0.0]]),
            baseline_losses=torch.tensor([0.3, 0.3]),
            parallel_losses=torch.tensor([0.3, 0.3]),
        )
        report = examples.report()

        self.assertEqual(len(report["parallel_worst"]), 2)
        self.assertEqual(len(report["parallel_best"]), 2)
        self.assertEqual(
            {item["dataset_index"] for item in report["parallel_worst"]},
            {10, 11},
        )

    def test_default_output_path_is_under_reports_dir(self):
        output = analysis.default_output_path(
            "example-model", "validation", 123)

        self.assertEqual(
            output,
            Path(
                "artifacts/reports/"
                "example-model.parallel-diagnostics.validation.123.json"
            ),
        )


if __name__ == "__main__":
    unittest.main()
