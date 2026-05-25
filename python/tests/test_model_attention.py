import importlib
import sys
import unittest
from pathlib import Path

import torch

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

model_module = importlib.import_module("libs.model")


class ModelAttentionTest(unittest.TestCase):
    def test_naive_board_self_attention_preserves_shape_and_finiteness(self):
        attention = model_module.NaiveBoardSelfAttention(
            model_module.CHANNELS, model_module.ATTENTION_DIM
        )
        attention.eval()
        x = torch.randn(2, model_module.CHANNELS, 8, 8)

        with torch.no_grad():
            y = attention(x)

        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())

    def test_naive_board_self_attention_is_identity_when_branch_is_zeroed(self):
        attention = model_module.NaiveBoardSelfAttention(
            model_module.CHANNELS, model_module.ATTENTION_DIM
        )
        attention.eval()
        for parameter in attention.parameters():
            torch.nn.init.zeros_(parameter)

        x = torch.randn(2, model_module.CHANNELS, 8, 8)

        with torch.no_grad():
            y = attention(x)

        self.assertTrue(torch.equal(y, x))

    def test_value_only_model_attention_placement(self):
        model = model_module.ValueOnlyModel()
        model.eval()

        self.assertEqual(len(model.blocks), model_module.BLOCKS)
        self.assertTrue(
            all(
                isinstance(block, model_module.GhostShuffleBlock)
                for block in model.blocks
            )
        )
        self.assertIsInstance(
            model.board_attention, model_module.NaiveBoardSelfAttention
        )

        call_order = []
        handles = []

        for index, block in enumerate(model.blocks):
            handles.append(
                block.register_forward_hook(
                    lambda _module, _input, _output, index=index: call_order.append(
                        f"block_{index}"
                    )
                )
            )

        handles.append(
            model.board_attention.register_forward_hook(
                lambda _module, _input, _output: call_order.append("board_attention")
            )
        )

        try:
            with torch.no_grad():
                output = model(torch.zeros(1, 20, 8, 8))
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(output.shape, torch.Size([1, 1]))
        self.assertEqual(
            call_order,
            [
                "block_0",
                "block_1",
                "block_2",
                "board_attention",
                "block_3",
                "block_4",
                "block_5",
            ],
        )


if __name__ == "__main__":
    unittest.main()
