import importlib
import sys
import unittest
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from libs import paths

analyze_rank = importlib.import_module("analyze_rank")
battle_entry = importlib.import_module("battle.entry")
eval_module = importlib.import_module("eval")
export_onnx = importlib.import_module("export_onnx")


class ArtifactContractTest(unittest.TestCase):
    def test_runtime_directory_contracts_are_canonical(self):
        self.assertEqual(
            paths.RAW_EVALUATIONS_PATH,
            Path("data/raw/lichess_db_eval.jsonl"),
        )
        self.assertEqual(
            paths.DUCKDB_TEMP_PATH,
            Path("data/interim/lichess_db_eval.duckdb.tmp"),
        )
        self.assertEqual(
            paths.TRAIN_DATA_PATH,
            Path("data/processed/train.chesseval"),
        )
        self.assertEqual(
            paths.VALIDATION_DATA_PATH,
            Path("data/processed/validation.chesseval"),
        )
        self.assertEqual(
            paths.checkpoint_path("experiment"),
            Path("artifacts/checkpoints/experiment.pth"),
        )
        self.assertEqual(
            paths.onnx_path("experiment"),
            Path("artifacts/onnx/experiment.onnx"),
        )

    def test_python_entrypoints_use_canonical_runtime_paths(self):
        self.assertEqual(
            battle_entry.resolve_checkpoint_path("explicit-model"),
            paths.checkpoint_path("explicit-model"),
        )
        self.assertEqual(analyze_rank.DATASET_PATH, paths.VALIDATION_DATA_PATH)
        self.assertEqual(export_onnx.ONNX_DIR, paths.ONNX_DIR)
        self.assertEqual(
            eval_module.checkpoint_path("explicit-model"),
            paths.checkpoint_path("explicit-model"),
        )


if __name__ == "__main__":
    unittest.main()
