import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import torch

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

analyze_rank_module = importlib.import_module("analyze_rank")
model_module = importlib.import_module("libs.model")


class AnalyzeRankHooksTest(unittest.TestCase):
    def test_register_hooks_matches_current_model_structure(self):
        model = model_module.ValueOnlyModel()
        model.eval()

        activations = analyze_rank_module.register_hooks(model)
        expected_names = {
            "initial_block",
            "board_attention",
            "value_head_conv",
            "value_head_mlp",
            *[f"block_{i}" for i in range(model_module.BLOCKS)],
        }

        self.assertEqual(set(activations), expected_names)

        with torch.no_grad():
            model(torch.zeros(2, 20, 8, 8))

        for name in expected_names:
            self.assertEqual(len(activations[name]), 1)

        self.assertEqual(
            activations["initial_block"][0].shape,
            torch.Size([2, model_module.CHANNELS, 8, 8]),
        )
        self.assertEqual(
            activations["block_0"][0].shape,
            torch.Size([2, model_module.CHANNELS, 8, 8]),
        )
        self.assertEqual(
            activations["board_attention"][0].shape,
            torch.Size([2, model_module.CHANNELS, 8, 8]),
        )
        self.assertEqual(
            activations["value_head_conv"][0].shape,
            torch.Size([2, 2, 8, 8]),
        )
        self.assertEqual(
            activations["value_head_mlp"][0].shape,
            torch.Size([2, 1]),
        )

    def test_register_hooks_supports_parallel_fusion_model_structure(self):
        model = model_module.ValueOnlyModel(
            model_variant=model_module.MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE,
        )
        model.eval()

        activations = analyze_rank_module.register_hooks(model)
        expected_names = {
            "initial_block",
            "fuse",
            "value_head_conv",
            "value_head_mlp",
            *[
                f"shared_block_{index}"
                for index in range(model_module.ATTENTION_AFTER_BLOCK)
            ],
            *[
                f"local_block_{index}"
                for index in range(
                    model_module.BLOCKS - model_module.ATTENTION_AFTER_BLOCK
                )
            ],
            *[
                f"global_block_{index}"
                for index in range(model_module.ATTENTION_LAYERS)
            ],
        }

        self.assertEqual(set(activations), expected_names)

        with torch.no_grad():
            model(torch.zeros(2, 20, 8, 8))

        for name in expected_names:
            self.assertEqual(len(activations[name]), 1)

        self.assertEqual(
            activations["initial_block"][0].shape,
            torch.Size([2, model_module.CHANNELS, 8, 8]),
        )
        self.assertEqual(
            activations["shared_block_0"][0].shape,
            torch.Size([2, model_module.CHANNELS, 8, 8]),
        )
        self.assertEqual(
            activations["local_block_0"][0].shape,
            torch.Size([2, model_module.CHANNELS, 8, 8]),
        )
        self.assertEqual(
            activations["global_block_0"][0].shape,
            torch.Size([2, model_module.CHANNELS, 8, 8]),
        )
        self.assertEqual(
            activations["fuse"][0].shape,
            torch.Size([2, model_module.CHANNELS, 8, 8]),
        )
        self.assertEqual(
            activations["value_head_conv"][0].shape,
            torch.Size([2, 2, 8, 8]),
        )
        self.assertEqual(
            activations["value_head_mlp"][0].shape,
            torch.Size([2, 1]),
        )

    def test_register_hooks_supports_kedge_late_evidence_model_structure(self):
        model = model_module.ValueOnlyModel(
            model_variant=(
                model_module.
                MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_LATEEVIDENCE_NO_MATERIAL
            ),
        )
        model.eval()

        activations = analyze_rank_module.register_hooks(model)
        expected_names = {
            "initial_block",
            "local_evidence",
            "global_evidence",
            "value_head_mlp",
            *[
                f"shared_block_{index}"
                for index in range(model_module.ATTENTION_AFTER_BLOCK)
            ],
            *[
                f"local_block_{index}"
                for index in range(
                    model_module.BLOCKS - model_module.ATTENTION_AFTER_BLOCK
                )
            ],
            *[
                f"global_block_{index}"
                for index in range(model_module.ATTENTION_LAYERS)
            ],
        }

        self.assertEqual(set(activations), expected_names)

        with torch.no_grad():
            model(torch.zeros(2, 20, 8, 8))

        for name in expected_names:
            self.assertEqual(len(activations[name]), 1)

        self.assertEqual(
            activations["initial_block"][0].shape,
            torch.Size([2, model_module.CHANNELS, 8, 8]),
        )
        self.assertEqual(
            activations["shared_block_0"][0].shape,
            torch.Size([2, model_module.CHANNELS, 8, 8]),
        )
        self.assertEqual(
            activations["local_block_0"][0].shape,
            torch.Size([2, model_module.CHANNELS, 8, 8]),
        )
        self.assertEqual(
            activations["global_block_0"][0].shape,
            torch.Size([2, model_module.CHANNELS, 8, 8]),
        )
        self.assertEqual(
            activations["local_evidence"][0].shape,
            torch.Size([2, 2, 8, 8]),
        )
        self.assertEqual(
            activations["global_evidence"][0].shape,
            torch.Size([2, 2, 8, 8]),
        )
        self.assertEqual(
            activations["value_head_mlp"][0].shape,
            torch.Size([2, 1]),
        )

    def test_validate_input_paths_reports_missing_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pth"
            dataset_path = Path(tmpdir) / "validation.chesseval"
            checkpoint_path.write_bytes(b"placeholder")

            with self.assertRaises(FileNotFoundError) as context:
                analyze_rank_module.validate_input_paths(
                    checkpoint_path, dataset_path)

        self.assertIn("validation.chesseval not found", str(context.exception))


if __name__ == "__main__":
    unittest.main()
