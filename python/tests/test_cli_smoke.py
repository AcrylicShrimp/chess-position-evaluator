import importlib
import sys
import unittest
from pathlib import Path

from typer.testing import CliRunner

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

cli_module = importlib.import_module("cli")


class CliSmokeTest(unittest.TestCase):
    def test_top_level_help_lists_primary_commands(self):
        result = CliRunner().invoke(cli_module.app, ["--help"])

        self.assertEqual(result.exit_code, 0, result.output)
        for command in [
            "train",
            "analyze-rank",
            "analyze-material-labels",
            "eval-dataset",
            "eval",
            "battle",
            "export-onnx",
        ]:
            self.assertIn(command, result.output)

    def test_eval_dataset_help_lists_test_split(self):
        result = CliRunner().invoke(cli_module.app, ["eval-dataset", "--help"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Dataset split: train,", result.output)
        self.assertIn("validation, or test", result.output)

    def test_train_help_lists_scheduler_options(self):
        result = CliRunner().invoke(cli_module.app, ["train", "--help"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--model-variant", result.output)
        self.assertIn("one-layer-edge-", result.output)
        self.assertIn("no-attention", result.output)
        self.assertIn("parallel-cnn-", result.output)
        self.assertIn("--scheduler", result.output)
        self.assertIn("warmup-cosine", result.output)
        self.assertIn("--warmup-epochs", result.output)

    def test_analyze_material_labels_help_lists_dataset_splits(self):
        result = CliRunner().invoke(
            cli_module.app,
            ["analyze-material-labels", "--help"],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Dataset split: train,", result.output)
        self.assertIn("validation, or test", result.output)
        self.assertIn("--full", result.output)

    def test_train_rejects_unknown_model_variant_with_allowed_values(self):
        result = CliRunner().invoke(
            cli_module.app,
            [
                "train",
                "example",
                "--epochs",
                "1",
                "--steps",
                "1",
                "--batch",
                "1",
                "--lr",
                "0.001",
                "--wd",
                "0.0001",
                "--model-variant",
                "unknown",
            ],
        )

        self.assertNotEqual(result.exit_code, 0, result.output)
        self.assertIn("Unsupported model variant 'unknown'", result.output)
        self.assertIn("one-layer-edge-gate", result.output)
        self.assertIn("stacked-edge-gate-ffn", result.output)
        self.assertIn("no-attention", result.output)
        self.assertIn("parallel-cnn-attn-fuse", result.output)


if __name__ == "__main__":
    unittest.main()
