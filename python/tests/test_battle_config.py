import importlib
import sys
import unittest
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

entry_module = importlib.import_module("battle.entry")


class BattleCheckpointConfigTest(unittest.TestCase):
    def test_explicit_model_name_resolves_under_checkpoints_dir(self):
        self.assertEqual(
            entry_module.resolve_checkpoint_path("explicit-model"),
            Path("models/checkpoints/explicit-model.pth"),
        )

    def test_model_name_is_required(self):
        with self.assertRaises(ValueError):
            entry_module.resolve_checkpoint_path("")


if __name__ == "__main__":
    unittest.main()
