import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import torch

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

train_entry_module = importlib.import_module("train.entry")
model_module = importlib.import_module("libs.model")


class TrainEntryTest(unittest.TestCase):
    def test_resume_uses_checkpoint_model_variant_before_model_construction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pth"
            torch.save(
                {
                    "model_variant": model_module.MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE,
                },
                checkpoint_path,
            )

            resolved = train_entry_module._resolve_model_variant_for_training(
                checkpoint_path,
                model_module.MODEL_VARIANT_STACKED_EDGE_GATE_FFN,
                resume=True,
            )

        self.assertEqual(
            resolved,
            model_module.MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE,
        )

    def test_resume_missing_checkpoint_keeps_requested_model_variant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "missing.pth"

            resolved = train_entry_module._resolve_model_variant_for_training(
                checkpoint_path,
                model_module.MODEL_VARIANT_NO_ATTENTION,
                resume=True,
            )

        self.assertEqual(resolved, model_module.MODEL_VARIANT_NO_ATTENTION)

    def test_non_resume_keeps_requested_model_variant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pth"
            torch.save(
                {
                    "model_variant": model_module.MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE,
                },
                checkpoint_path,
            )

            resolved = train_entry_module._resolve_model_variant_for_training(
                checkpoint_path,
                model_module.MODEL_VARIANT_NO_ATTENTION,
                resume=False,
            )

        self.assertEqual(resolved, model_module.MODEL_VARIANT_NO_ATTENTION)


if __name__ == "__main__":
    unittest.main()
