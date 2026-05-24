import importlib
import sys
import unittest
from pathlib import Path

import torch

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

model_module = importlib.import_module("libs.model")


def make_known_material_board() -> torch.Tensor:
    board = torch.zeros(1, 20, 8, 8)
    board[:, 6, 0, 0] = 1.0
    board[:, 15, 0, 1] = 1.0
    return board


class CaptureValueHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("material_weights", model_module._MATERIAL_VALUES.clone())
        self.activation_shape = None
        self.material_diff = None

    def forward(
        self, x: torch.Tensor, material_diff: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.activation_shape = x.shape
        self.material_diff = material_diff.detach().cpu().clone()
        return torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)


class ModelMaterialFeatureTest(unittest.TestCase):
    def test_material_diff_uses_board_piece_channels(self):
        board = torch.zeros(2, 20, 8, 8)
        board[0, 6, 0, 0] = 1.0
        board[0, 15, 0, 1] = 1.0
        board[1, 10, 0, 0] = 1.0
        board[1, 13, 0, 1] = 1.0

        diff = model_module._material_diff_from_board(
            board, model_module._MATERIAL_VALUES
        )
        self.assertTrue(torch.equal(diff, torch.tensor([-4.0, 6.0])))

        feature = model_module._material_feature(
            board,
            model_module._MATERIAL_VALUES,
            material_scale=1.0,
        )
        expected = torch.tanh(torch.tensor([-4.0, 6.0]) / model_module._MATERIAL_ALPHA)
        self.assertTrue(torch.allclose(feature, expected.unsqueeze(1)))

    def test_material_diff_rejects_trunk_activation_shape(self):
        trunk_activation = torch.zeros(1, model_module.CHANNELS, 8, 8)

        with self.assertRaisesRegex(ValueError, r"\[B, 20, 8, 8\]"):
            model_module._material_diff_from_board(
                trunk_activation, model_module._MATERIAL_VALUES
            )

    def test_value_head_requires_explicit_material_diff(self):
        value_head = model_module.ValueHead()
        value_head.eval()
        trunk_activation = torch.zeros(1, model_module.CHANNELS, 8, 8)

        with self.assertRaisesRegex(ValueError, "explicit material_diff"):
            value_head(trunk_activation)

    def test_value_only_model_passes_board_material_diff_to_value_head(self):
        model = model_module.ValueOnlyModel()
        model.eval()
        capture_head = CaptureValueHead()
        model.value_head = capture_head

        with torch.no_grad():
            output = model(make_known_material_board())

        self.assertEqual(output.shape, torch.Size([1, 1]))
        self.assertEqual(
            capture_head.activation_shape,
            torch.Size([1, model_module.CHANNELS, 8, 8]),
        )
        self.assertTrue(torch.equal(capture_head.material_diff, torch.tensor([-4.0])))

    def test_explicit_material_diff_overrides_board_material(self):
        model = model_module.ValueOnlyModel()
        model.eval()
        capture_head = CaptureValueHead()
        model.value_head = capture_head

        with torch.no_grad():
            model(make_known_material_board(), material_diff=torch.tensor([7.0]))

        self.assertTrue(torch.equal(capture_head.material_diff, torch.tensor([7.0])))


if __name__ == "__main__":
    unittest.main()
