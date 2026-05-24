import importlib
import sys
import unittest
from pathlib import Path

import chess
import torch

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

encoding_module = importlib.import_module("libs.encoding")


class BoardEncodingContractTest(unittest.TestCase):
    def test_board2tensor_shape_dtype_and_side_to_move_plane(self):
        white_to_move = chess.Board("4k3/8/8/8/7q/8/8/1N2K3 w - - 0 1")
        black_to_move = chess.Board("4k3/8/8/8/7q/8/8/1N2K3 b - - 0 1")

        white_tensor = encoding_module.board2tensor(white_to_move)
        black_tensor = encoding_module.board2tensor(black_to_move)

        self.assertEqual(white_tensor.shape, torch.Size([20, 8, 8]))
        self.assertEqual(black_tensor.shape, torch.Size([20, 8, 8]))
        self.assertEqual(white_tensor.dtype, torch.float32)
        self.assertEqual(black_tensor.dtype, torch.float32)
        self.assertTrue(torch.equal(white_tensor[0], torch.zeros(8, 8)))
        self.assertTrue(torch.equal(black_tensor[0], torch.ones(8, 8)))

    def test_piece_channels_are_side_to_move_relative_for_white(self):
        board = chess.Board("4k3/8/8/8/7q/8/8/1N2K3 w - - 0 1")
        tensor = encoding_module.board2tensor(board)

        self.assertEqual(tensor[7, 0, 1].item(), 1.0)
        self.assertEqual(tensor[11, 0, 4].item(), 1.0)
        self.assertEqual(tensor[16, 3, 7].item(), 1.0)
        self.assertEqual(tensor[17, 7, 4].item(), 1.0)

    def test_piece_channels_are_side_to_move_relative_and_flipped_for_black(self):
        board = chess.Board("4k3/8/8/8/7q/8/8/1N2K3 b - - 0 1")
        tensor = encoding_module.board2tensor(board)

        self.assertEqual(tensor[10, 4, 7].item(), 1.0)
        self.assertEqual(tensor[11, 0, 4].item(), 1.0)
        self.assertEqual(tensor[13, 7, 1].item(), 1.0)
        self.assertEqual(tensor[17, 7, 4].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
