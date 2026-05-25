import importlib
import sys
import unittest
from pathlib import Path

import torch

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

model_module = importlib.import_module("libs.model")


def square(row: int, file: int) -> int:
    return row * model_module.BOARD_ATTENTION_SIZE + file


def pair_index(query: int, key: int) -> int:
    return (
        query * model_module.BOARD_ATTENTION_SIZE * model_module.BOARD_ATTENTION_SIZE
        + key
    )


def relation_index(name: str) -> int:
    return model_module.BOARD_ATTENTION_RELATIONS.index(name)


class ModelAttentionTest(unittest.TestCase):
    def test_naive_board_self_attention_preserves_shape_and_finiteness(self):
        attention = model_module.NaiveBoardSelfAttention(
            model_module.CHANNELS,
            model_module.ATTENTION_HEADS,
            model_module.ATTENTION_HEAD_DIM,
        )
        attention.eval()
        x = torch.randn(2, model_module.CHANNELS, 8, 8)

        with torch.no_grad():
            y = attention(x)

        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())

    def test_naive_board_self_attention_is_identity_when_branch_is_zeroed(self):
        attention = model_module.NaiveBoardSelfAttention(
            model_module.CHANNELS,
            model_module.ATTENTION_HEADS,
            model_module.ATTENTION_HEAD_DIM,
        )
        attention.eval()
        for parameter in attention.parameters():
            torch.nn.init.zeros_(parameter)

        x = torch.randn(2, model_module.CHANNELS, 8, 8)

        with torch.no_grad():
            y = attention(x)

        self.assertTrue(torch.equal(y, x))

    def test_naive_board_self_attention_uses_four_heads(self):
        attention = model_module.NaiveBoardSelfAttention(
            model_module.CHANNELS,
            model_module.ATTENTION_HEADS,
            model_module.ATTENTION_HEAD_DIM,
        )

        self.assertEqual(attention.heads, 4)
        self.assertEqual(attention.head_dim, 16)
        self.assertEqual(attention.attn_dim, model_module.ATTENTION_DIM)
        self.assertEqual(attention.q_proj.out_channels, model_module.ATTENTION_DIM)
        self.assertEqual(attention.k_proj.out_channels, model_module.ATTENTION_DIM)
        self.assertEqual(attention.v_proj.out_channels, model_module.ATTENTION_DIM)
        self.assertEqual(attention.out_proj.in_channels, model_module.ATTENTION_DIM)
        self.assertEqual(attention.scale, model_module.ATTENTION_HEAD_DIM**-0.5)

    def test_attention_relation_bias_parameters_are_zero_initialized(self):
        attention = model_module.NaiveBoardSelfAttention(
            model_module.CHANNELS,
            model_module.ATTENTION_HEADS,
            model_module.ATTENTION_HEAD_DIM,
        )

        self.assertEqual(
            attention.rel_bias.shape,
            torch.Size(
                [
                    model_module.ATTENTION_HEADS,
                    len(model_module.BOARD_ATTENTION_RELATIONS),
                ]
            ),
        )
        self.assertEqual(
            attention.dist_bias.shape,
            torch.Size(
                [
                    model_module.ATTENTION_HEADS,
                    model_module.BOARD_ATTENTION_SIZE,
                ]
            ),
        )
        self.assertTrue(
            torch.equal(attention.rel_bias, torch.zeros_like(attention.rel_bias))
        )
        self.assertTrue(
            torch.equal(attention.dist_bias, torch.zeros_like(attention.dist_bias))
        )

    def test_attention_geometry_masks_match_chess_board_relations(self):
        attention = model_module.NaiveBoardSelfAttention(
            model_module.CHANNELS,
            model_module.ATTENTION_HEADS,
            model_module.ATTENTION_HEAD_DIM,
        )

        cases = [
            ("same_square", square(0, 0), square(0, 0)),
            ("same_rank", square(0, 0), square(0, 7)),
            ("same_file", square(0, 0), square(7, 0)),
            ("same_diagonal", square(0, 0), square(7, 7)),
            ("same_anti_diagonal", square(0, 7), square(7, 0)),
            ("knight_move", square(0, 0), square(2, 1)),
            ("king_adjacent", square(3, 3), square(4, 4)),
            ("our_pawn_attack_geometry", square(1, 1), square(2, 0)),
            ("their_pawn_attack_geometry", square(1, 1), square(0, 2)),
        ]

        for relation, query, key in cases:
            with self.subTest(relation=relation):
                self.assertEqual(
                    attention.relation_masks_flat[
                        relation_index(relation),
                        pair_index(query, key),
                    ].item(),
                    1.0,
                )

        same_square_pair = pair_index(square(0, 0), square(0, 0))
        for relation in [
            "same_rank",
            "same_file",
            "same_diagonal",
            "same_anti_diagonal",
        ]:
            with self.subTest(relation=relation):
                self.assertEqual(
                    attention.relation_masks_flat[
                        relation_index(relation),
                        same_square_pair,
                    ].item(),
                    0.0,
                )

    def test_attention_distance_index_uses_chebyshev_distance(self):
        attention = model_module.NaiveBoardSelfAttention(
            model_module.CHANNELS,
            model_module.ATTENTION_HEADS,
            model_module.ATTENTION_HEAD_DIM,
        )

        cases = [
            (square(0, 0), square(0, 0), 0),
            (square(0, 0), square(0, 7), 7),
            (square(3, 4), square(7, 2), 4),
            (square(6, 6), square(4, 5), 2),
        ]

        for query, key, expected in cases:
            with self.subTest(query=query, key=key):
                self.assertEqual(
                    attention.distance_index_flat[pair_index(query, key)].item(),
                    expected,
                )

    def test_attention_geometry_bias_composes_relations_and_distance(self):
        attention = model_module.NaiveBoardSelfAttention(
            model_module.CHANNELS,
            model_module.ATTENTION_HEADS,
            model_module.ATTENTION_HEAD_DIM,
        )
        with torch.no_grad():
            attention.rel_bias.zero_()
            attention.dist_bias.zero_()
            attention.rel_bias[0, relation_index("same_file")] = 2.0
            attention.rel_bias[0, relation_index("king_adjacent")] = 3.0
            attention.dist_bias[0, 1] = 0.5

        relation_bias = attention.rel_bias @ attention.relation_masks_flat
        distance_bias = attention.dist_bias.index_select(
            dim=1,
            index=attention.distance_index_flat,
        )
        bias = (relation_bias + distance_bias).reshape(
            attention.heads,
            model_module.BOARD_ATTENTION_SIZE * model_module.BOARD_ATTENTION_SIZE,
            model_module.BOARD_ATTENTION_SIZE * model_module.BOARD_ATTENTION_SIZE,
        )

        query = square(3, 3)
        key = square(4, 3)
        self.assertEqual(bias[0, query, key].item(), 5.5)
        self.assertEqual(bias[1, query, key].item(), 0.0)

    def test_zero_relation_bias_matches_unbiased_attention_logits(self):
        torch.manual_seed(0)
        attention = model_module.NaiveBoardSelfAttention(
            model_module.CHANNELS,
            model_module.ATTENTION_HEADS,
            model_module.ATTENTION_HEAD_DIM,
        )
        attention.eval()
        x = torch.randn(2, model_module.CHANNELS, 8, 8)

        with torch.no_grad():
            y = attention(x)

            batch, _channels, height, width = x.shape
            tokens = height * width
            q = attention.q_proj(x).reshape(
                batch, attention.heads, attention.head_dim, tokens
            ).transpose(2, 3)
            k = attention.k_proj(x).reshape(
                batch, attention.heads, attention.head_dim, tokens
            )
            v = attention.v_proj(x).reshape(
                batch, attention.heads, attention.head_dim, tokens
            ).transpose(2, 3)
            weights = torch.softmax(torch.matmul(q, k) * attention.scale, dim=-1)
            context = torch.matmul(weights, v)
            context = context.transpose(2, 3).reshape(
                batch,
                attention.attn_dim,
                height,
                width,
            )
            expected = x + attention.out_proj(context)

        self.assertTrue(torch.allclose(y, expected))

    def test_attention_rejects_non_8x8_inputs(self):
        attention = model_module.NaiveBoardSelfAttention(
            model_module.CHANNELS,
            model_module.ATTENTION_HEADS,
            model_module.ATTENTION_HEAD_DIM,
        )

        with self.assertRaisesRegex(ValueError, "8x8"):
            attention(torch.zeros(1, model_module.CHANNELS, 7, 8))

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
