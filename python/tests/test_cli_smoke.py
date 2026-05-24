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
        for command in ["train", "analyze-rank", "eval", "battle", "export-onnx"]:
            self.assertIn(command, result.output)


if __name__ == "__main__":
    unittest.main()
