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
        self.assertEqual(attention.q_proj.out_channels,
                         model_module.ATTENTION_DIM)
        self.assertEqual(attention.k_proj.out_channels,
                         model_module.ATTENTION_DIM)
        self.assertEqual(attention.v_proj.out_channels,
                         model_module.ATTENTION_DIM)
        self.assertEqual(attention.out_proj.in_channels,
                         model_module.ATTENTION_DIM)
        self.assertEqual(
            attention.scale, model_module.ATTENTION_HEAD_DIM**-0.5)

    def test_board_attention_stack_contract(self):
        stack = model_module.BoardAttentionStack(
            model_module.ATTENTION_LAYERS,
            model_module.CHANNELS,
            model_module.ATTENTION_HEADS,
            model_module.ATTENTION_HEAD_DIM,
            model_module.ATTENTION_FFN_HIDDEN,
        )

        self.assertEqual(len(stack.layers), 3)
        for layer in stack.layers:
            with self.subTest(layer=layer):
                self.assertIsInstance(
                    layer.attention,
                    model_module.NaiveBoardSelfAttention,
                )
                self.assertIsInstance(
                    layer.ffn, model_module.BoardAttentionFFN)

    def test_board_attention_stack_is_identity_when_branch_is_zeroed(self):
        stack = model_module.BoardAttentionStack(
            model_module.ATTENTION_LAYERS,
            model_module.CHANNELS,
            model_module.ATTENTION_HEADS,
            model_module.ATTENTION_HEAD_DIM,
            model_module.ATTENTION_FFN_HIDDEN,
        )
        stack.eval()
        for parameter in stack.parameters():
            torch.nn.init.zeros_(parameter)

        x = torch.randn(2, model_module.CHANNELS, 8, 8)

        with torch.no_grad():
            y = stack(x)

        self.assertTrue(torch.equal(y, x))

    def test_attention_edge_gate_parameters_are_zero_initialized(self):
        attention = model_module.NaiveBoardSelfAttention(
            model_module.CHANNELS,
            model_module.ATTENTION_HEADS,
            model_module.ATTENTION_HEAD_DIM,
        )

        expected_relation_shape = torch.Size(
            [
                model_module.ATTENTION_HEADS,
                len(model_module.BOARD_ATTENTION_RELATIONS),
                model_module.ATTENTION_HEAD_DIM,
            ]
        )
        expected_distance_shape = torch.Size(
            [
                model_module.ATTENTION_HEADS,
                model_module.BOARD_ATTENTION_SIZE,
                model_module.ATTENTION_HEAD_DIM,
            ]
        )

        self.assertEqual(attention.rel_gate_q.shape, expected_relation_shape)
        self.assertEqual(attention.rel_gate_k.shape, expected_relation_shape)
        self.assertEqual(attention.dist_gate_q.shape, expected_distance_shape)
        self.assertEqual(attention.dist_gate_k.shape, expected_distance_shape)

        for name in [
            "rel_gate_q",
            "rel_gate_k",
            "dist_gate_q",
            "dist_gate_k",
        ]:
            parameter = getattr(attention, name)
            with self.subTest(name=name):
                self.assertTrue(torch.equal(
                    parameter, torch.zeros_like(parameter)))

        self.assertNotIn("rel_bias", dict(attention.named_parameters()))
        self.assertNotIn("dist_bias", dict(attention.named_parameters()))

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
                    attention.distance_index_flat[pair_index(
                        query, key)].item(),
                    expected,
                )
                self.assertEqual(
                    attention.distance_masks_flat[
                        expected,
                        pair_index(query, key),
                    ].item(),
                    1.0,
                )

    def test_attention_edge_gate_composes_relations_and_distance(self):
        attention = model_module.NaiveBoardSelfAttention(
            model_module.CHANNELS,
            model_module.ATTENTION_HEADS,
            model_module.ATTENTION_HEAD_DIM,
        )
        with torch.no_grad():
            attention.rel_gate_q.zero_()
            attention.rel_gate_k.zero_()
            attention.dist_gate_q.zero_()
            attention.dist_gate_k.zero_()
            attention.rel_gate_q[0, relation_index("same_file"), 0] = 1.0
            attention.rel_gate_k[0, relation_index("king_adjacent"), 0] = 2.0
            attention.dist_gate_q[0, 1, 0] = 4.0
            attention.dist_gate_k[0, 1, 0] = 5.0

        tokens = model_module.BOARD_ATTENTION_SIZE * model_module.BOARD_ATTENTION_SIZE
        q = torch.zeros(1, attention.heads, tokens, attention.head_dim)
        k = torch.zeros(1, attention.heads, tokens, attention.head_dim)
        query = square(3, 3)
        key = square(4, 3)
        q[0, 0, query, 0] = 2.0
        k[0, 0, key, 0] = 3.0

        edge_gate_logits = attention._edge_gate_logits(
            q,
            k,
            tokens,
            torch.float32,
        )

        expected = (2.0 + 6.0 + 8.0 + 15.0) * attention.scale
        self.assertEqual(edge_gate_logits[0, 0, query, key].item(), expected)
        self.assertEqual(edge_gate_logits[0, 1, query, key].item(), 0.0)

    def test_k_only_edge_gate_omits_query_gate_parameters(self):
        attention = model_module.NaiveBoardSelfAttention(
            model_module.CHANNELS,
            model_module.ATTENTION_HEADS,
            model_module.ATTENTION_HEAD_DIM,
            edge_gate_mode=model_module.EDGE_GATE_K_ONLY,
        )
        tokens = model_module.BOARD_ATTENTION_SIZE * model_module.BOARD_ATTENTION_SIZE
        q_a = torch.randn(
            2,
            model_module.ATTENTION_HEADS,
            tokens,
            model_module.ATTENTION_HEAD_DIM,
        )
        q_b = torch.randn_like(q_a)
        k = torch.randn_like(q_a)
        with torch.no_grad():
            attention.rel_gate_k.fill_(0.25)
            attention.dist_gate_k.fill_(-0.125)

        logits_a = attention._edge_gate_logits(q_a, k, tokens, torch.float32)
        logits_b = attention._edge_gate_logits(q_b, k, tokens, torch.float32)

        self.assertIsNone(attention.rel_gate_q)
        self.assertIsNone(attention.dist_gate_q)
        self.assertNotIn("rel_gate_q", attention.state_dict())
        self.assertNotIn("dist_gate_q", attention.state_dict())
        self.assertIn("rel_gate_k", attention.state_dict())
        self.assertIn("dist_gate_k", attention.state_dict())
        self.assertTrue(torch.equal(logits_a, logits_b))

    def test_zero_edge_gate_matches_unbiased_attention_logits(self):
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
            weights = torch.softmax(torch.matmul(
                q, k) * attention.scale, dim=-1)
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
            model.board_attention,
            model_module.BoardAttentionStack,
        )
        self.assertEqual(
            len(model.board_attention.layers),
            model_module.ATTENTION_LAYERS,
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
                lambda _module, _input, _output: call_order.append(
                    "board_attention")
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

    def test_model_full_and_value_only_use_attention_stack(self):
        cases = [
            ("value_only", model_module.ValueOnlyModel()),
            ("model_full", model_module.ModelFull()),
        ]

        for name, model in cases:
            with self.subTest(name=name):
                self.assertIsInstance(
                    model.board_attention,
                    model_module.BoardAttentionStack,
                )
                self.assertEqual(
                    len(model.board_attention.layers),
                    model_module.ATTENTION_LAYERS,
                )

    def test_value_only_model_parameter_count_matches_stacked_attention_plan(self):
        model = model_module.ValueOnlyModel()

        self.assertEqual(sum(p.numel() for p in model.parameters()), 457685)

    def test_no_attention_variant_uses_identity_board_attention(self):
        model = model_module.ValueOnlyModel(
            model_variant=model_module.MODEL_VARIANT_NO_ATTENTION,
        )

        self.assertIsInstance(
            model.board_attention,
            model_module.IdentityBoardAttention,
        )
        self.assertEqual(model.model_variant,
                         model_module.MODEL_VARIANT_NO_ATTENTION)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 155861)

    def test_one_layer_edge_gate_variant_uses_single_attention_layer(self):
        model = model_module.ValueOnlyModel(
            model_variant=model_module.MODEL_VARIANT_ONE_LAYER_EDGE_GATE,
        )

        self.assertIsInstance(
            model.board_attention,
            model_module.NaiveBoardSelfAttention,
        )
        self.assertEqual(
            model.model_variant,
            model_module.MODEL_VARIANT_ONE_LAYER_EDGE_GATE,
        )
        self.assertEqual(sum(p.numel() for p in model.parameters()), 223573)

    def test_parallel_cnn_attention_fuse_variant_topology(self):
        model = model_module.ValueOnlyModel(
            model_variant=model_module.MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE,
        )
        model.eval()

        self.assertIsInstance(
            model.trunk, model_module.ParallelCnnAttentionTrunk)
        self.assertEqual(len(model.trunk.shared_blocks), 3)
        self.assertEqual(len(model.trunk.local_blocks), 3)
        self.assertTrue(
            all(
                isinstance(block, model_module.GhostShuffleBlock)
                for block in model.trunk.shared_blocks
            )
        )
        self.assertTrue(
            all(
                isinstance(block, model_module.GhostShuffleBlock)
                for block in model.trunk.local_blocks
            )
        )
        self.assertIsInstance(model.trunk.global_blocks,
                              model_module.BoardAttentionStack)
        self.assertEqual(len(model.trunk.global_blocks.layers), 3)
        for layer in model.trunk.global_blocks.layers:
            self.assertEqual(layer.ffn.net[0].out_channels, 64)
        self.assertIsInstance(
            model.value_head, model_module.MaterialFeatureValueHead)

        fuse_conv = model.trunk.fuse[0]
        self.assertIsInstance(fuse_conv, torch.nn.Conv2d)
        self.assertEqual(fuse_conv.in_channels, model_module.CHANNELS * 2)
        self.assertEqual(fuse_conv.out_channels, model_module.CHANNELS)
        self.assertEqual(fuse_conv.kernel_size, (1, 1))
        self.assertFalse(hasattr(model, "blocks"))
        self.assertFalse(hasattr(model, "board_attention"))

        with torch.no_grad():
            output = model(torch.zeros(1, 20, 8, 8))

        self.assertEqual(output.shape, torch.Size([1, 1]))
        self.assertEqual(
            sum(p.numel() for p in model.parameters()),
            589397,
        )

    def test_parallel_cnn_attention_aligned_add_variant_topology(self):
        model = model_module.ValueOnlyModel(
            model_variant=model_module.MODEL_VARIANT_PARALLEL_CNN_ATTN_ALIGNED_ADD,
        )
        model.eval()

        self.assertIsInstance(
            model.trunk, model_module.ParallelCnnAttentionAlignedAddTrunk)
        self.assertEqual(len(model.trunk.shared_blocks), 3)
        self.assertEqual(len(model.trunk.local_blocks), 3)
        self.assertIsInstance(model.trunk.global_blocks,
                              model_module.BoardAttentionStack)
        self.assertEqual(len(model.trunk.global_blocks.layers), 3)
        self.assertIsInstance(
            model.value_head, model_module.MaterialFeatureValueHead)

        for aligner in (model.trunk.local_aligner, model.trunk.global_aligner):
            self.assertIsInstance(aligner[0], torch.nn.BatchNorm2d)
            self.assertIsInstance(aligner[1], torch.nn.Conv2d)
            self.assertEqual(aligner[1].in_channels, model_module.CHANNELS)
            self.assertEqual(aligner[1].out_channels, model_module.CHANNELS)
            self.assertEqual(aligner[1].kernel_size, (1, 1))
            self.assertFalse(aligner[1].bias)

        self.assertIsInstance(model.trunk.fuse[0], torch.nn.BatchNorm2d)
        self.assertIsInstance(model.trunk.fuse[1], torch.nn.Hardswish)
        self.assertFalse(hasattr(model, "blocks"))
        self.assertFalse(hasattr(model, "board_attention"))

        with torch.no_grad():
            output = model(torch.zeros(1, 20, 8, 8))

        self.assertEqual(output.shape, torch.Size([1, 1]))
        self.assertEqual(
            sum(p.numel() for p in model.parameters()),
            590421,
        )

    def test_parallel_cnn_attention_fuse_no_material_variant_topology(self):
        model = model_module.ValueOnlyModel(
            model_variant=(
                model_module.MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE_NO_MATERIAL
            ),
        )
        model.eval()

        self.assertIsInstance(
            model.trunk, model_module.ParallelCnnAttentionTrunk)
        self.assertIsInstance(model.value_head, model_module.ResidualValueHead)
        self.assertFalse(hasattr(model, "blocks"))
        self.assertFalse(hasattr(model, "board_attention"))

        with torch.no_grad():
            output = model(torch.zeros(1, 20, 8, 8))

        self.assertEqual(output.shape, torch.Size([1, 1]))
        self.assertEqual(
            sum(p.numel() for p in model.parameters()),
            589205,
        )

    def test_parallel_cnn_attention_kedge_fuse_no_material_variant_topology(self):
        model = model_module.ValueOnlyModel(
            model_variant=(
                model_module.
                MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_FUSE_NO_MATERIAL
            ),
        )
        model.eval()

        self.assertIsInstance(
            model.trunk, model_module.ParallelCnnAttentionTrunk)
        self.assertIsInstance(model.value_head, model_module.ResidualValueHead)
        self.assertFalse(hasattr(model, "blocks"))
        self.assertFalse(hasattr(model, "board_attention"))

        fuse_conv = model.trunk.fuse[0]
        self.assertIsInstance(fuse_conv, torch.nn.Conv2d)
        self.assertEqual(fuse_conv.in_channels, model_module.CHANNELS * 2)
        self.assertEqual(fuse_conv.out_channels, model_module.CHANNELS)
        self.assertEqual(fuse_conv.kernel_size, (1, 1))

        for layer in model.trunk.global_blocks.layers:
            self.assertEqual(layer.attention.edge_gate_mode,
                             model_module.EDGE_GATE_K_ONLY)
            self.assertIsNone(layer.attention.rel_gate_q)
            self.assertIsNone(layer.attention.dist_gate_q)
            self.assertIsNotNone(layer.attention.rel_gate_k)
            self.assertIsNotNone(layer.attention.dist_gate_k)

        with torch.no_grad():
            output = model(torch.zeros(1, 20, 8, 8))

        self.assertEqual(output.shape, torch.Size([1, 1]))
        self.assertEqual(
            sum(p.numel() for p in model.parameters()),
            585941,
        )

    def test_parallel_cnn_attention_kedge_late_evidence_no_material_topology(self):
        model = model_module.ValueOnlyModel(
            model_variant=(
                model_module.
                MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_LATEEVIDENCE_NO_MATERIAL
            ),
        )
        model.eval()

        self.assertIsInstance(
            model.trunk,
            model_module.ParallelCnnAttentionKEdgeLateEvidenceTrunk,
        )
        self.assertIsInstance(model.value_head,
                              model_module.LateEvidenceValueHead)
        self.assertEqual(len(model.trunk.shared_blocks), 3)
        self.assertEqual(len(model.trunk.local_blocks), 3)
        self.assertIsInstance(model.trunk.global_blocks,
                              model_module.BoardAttentionStack)
        self.assertEqual(len(model.trunk.global_blocks.layers), 3)
        self.assertFalse(hasattr(model.trunk, "fuse"))
        self.assertFalse(hasattr(model, "blocks"))
        self.assertFalse(hasattr(model, "board_attention"))

        for layer in model.trunk.global_blocks.layers:
            self.assertEqual(layer.attention.edge_gate_mode,
                             model_module.EDGE_GATE_K_ONLY)
            self.assertIsNone(layer.attention.rel_gate_q)
            self.assertIsNone(layer.attention.dist_gate_q)
            self.assertIsNotNone(layer.attention.rel_gate_k)
            self.assertIsNotNone(layer.attention.dist_gate_k)
            self.assertEqual(layer.ffn.net[0].out_channels, 64)

        for evidence in (model.trunk.local_evidence,
                         model.trunk.global_evidence):
            evidence_conv = evidence[0]
            self.assertIsInstance(evidence_conv, torch.nn.Conv2d)
            self.assertEqual(evidence_conv.in_channels, model_module.CHANNELS)
            self.assertEqual(evidence_conv.out_channels, 2)
            self.assertEqual(evidence_conv.kernel_size, (1, 1))
            self.assertIsInstance(evidence[1], torch.nn.BatchNorm2d)
            self.assertIsInstance(evidence[2], torch.nn.Hardswish)

        with torch.no_grad():
            trunk_output = model.trunk(
                torch.zeros(1, model_module.CHANNELS, 8, 8)
            )
            output = model(torch.zeros(1, 20, 8, 8))

        self.assertEqual(trunk_output.shape, torch.Size([1, 4, 8, 8]))
        self.assertEqual(output.shape, torch.Size([1, 1]))
        self.assertEqual(
            sum(p.numel() for p in model.parameters()),
            463065,
        )

    def test_funnel_cnn_attention_variant_topology(self):
        model = model_module.ValueOnlyModel(
            model_variant=model_module.MODEL_VARIANT_FUNNEL_CNN_ATTENTION,
        )
        model.eval()

        self.assertIsInstance(
            model.trunk, model_module.FunnelCnnAttentionTrunk)
        self.assertIsInstance(model.value_head, model_module.ValueHead)
        self.assertFalse(hasattr(model, "blocks"))
        self.assertFalse(hasattr(model, "board_attention"))

        initial_conv = model.initial_block[0]
        self.assertEqual(initial_conv.in_channels, 24)
        self.assertEqual(
            initial_conv.out_channels, model_module.FUNNEL_INITIAL_CHANNELS)
        self.assertEqual(len(model.trunk.wide_blocks), 2)
        self.assertEqual(len(model.trunk.mid_blocks), 2)
        self.assertEqual(len(model.trunk.narrow_blocks), 2)

        wide_projection = model.trunk.compress_wide_to_mid.net[0]
        self.assertEqual(
            wide_projection.in_channels,
            model_module.FUNNEL_INITIAL_CHANNELS,
        )
        self.assertEqual(
            wide_projection.out_channels,
            model_module.FUNNEL_MID_CHANNELS,
        )

        mid_projection = model.trunk.compress_mid_to_attention.net[0]
        self.assertEqual(
            mid_projection.in_channels,
            model_module.FUNNEL_MID_CHANNELS,
        )
        self.assertEqual(
            mid_projection.out_channels,
            model_module.FUNNEL_ATTENTION_CHANNELS,
        )

        self.assertEqual(
            len(model.trunk.attention_blocks.layers),
            model_module.FUNNEL_ATTENTION_LAYERS,
        )
        for layer in model.trunk.attention_blocks.layers:
            self.assertEqual(
                layer.attention.channels,
                model_module.FUNNEL_ATTENTION_CHANNELS,
            )
            self.assertEqual(
                layer.attention.edge_gate_mode,
                model_module.EDGE_GATE_QK,
            )
            self.assertEqual(layer.attention.q_proj.out_channels, 64)
            self.assertEqual(layer.attention.out_proj.in_channels, 64)
            self.assertEqual(layer.ffn.net[0].out_channels, 64)

        self.assertEqual(
            model.value_head.conv[0].in_channels,
            model_module.FUNNEL_ATTENTION_CHANNELS,
        )

        with torch.no_grad():
            trunk_output = model.trunk(
                torch.zeros(
                    1,
                    model_module.FUNNEL_INITIAL_CHANNELS,
                    8,
                    8,
                )
            )
            output = model(torch.zeros(1, 20, 8, 8))

        self.assertEqual(
            trunk_output.shape,
            torch.Size([1, model_module.FUNNEL_ATTENTION_CHANNELS, 8, 8]),
        )
        self.assertEqual(output.shape, torch.Size([1, 1]))
        self.assertEqual(
            sum(p.numel() for p in model.parameters()),
            474197,
        )

    def test_funnel_interleaved_attention_variant_topology(self):
        model = model_module.ValueOnlyModel(
            model_variant=(
                model_module.MODEL_VARIANT_FUNNEL_INTERLEAVED_ATTENTION
            ),
        )
        model.eval()

        self.assertIsInstance(
            model.trunk, model_module.FunnelInterleavedAttentionTrunk)
        self.assertIsInstance(model.value_head, model_module.ValueHead)
        self.assertFalse(hasattr(model, "blocks"))
        self.assertFalse(hasattr(model, "board_attention"))

        initial_conv = model.initial_block[0]
        self.assertEqual(initial_conv.in_channels, 24)
        self.assertEqual(
            initial_conv.out_channels, model_module.FUNNEL_INITIAL_CHANNELS)
        self.assertEqual(len(model.trunk.wide_blocks), 2)
        self.assertEqual(len(model.trunk.mid_blocks), 2)
        self.assertEqual(len(model.trunk.narrow_blocks), 2)
        self.assertEqual(
            len(model.trunk.interleaved_blocks),
            model_module.FUNNEL_INTERLEAVED_STAGES,
        )

        for block in model.trunk.interleaved_blocks:
            self.assertIsInstance(
                block, model_module.AttentionCnnRefreshBlock)
            self.assertIsInstance(
                block.attention, model_module.BoardAttentionBlock)
            self.assertIsInstance(
                block.refresh, model_module.GhostShuffleBlock)
            self.assertEqual(
                block.attention.attention.channels,
                model_module.FUNNEL_ATTENTION_CHANNELS,
            )
            self.assertEqual(
                block.attention.attention.edge_gate_mode,
                model_module.EDGE_GATE_QK,
            )
            self.assertEqual(block.attention.attention.q_proj.out_channels, 64)
            self.assertEqual(block.attention.attention.out_proj.in_channels, 64)
            self.assertEqual(block.attention.ffn.net[0].out_channels, 64)

        self.assertEqual(
            model.value_head.conv[0].in_channels,
            model_module.FUNNEL_ATTENTION_CHANNELS,
        )

        with torch.no_grad():
            trunk_output = model.trunk(
                torch.zeros(
                    1,
                    model_module.FUNNEL_INITIAL_CHANNELS,
                    8,
                    8,
                )
            )
            output = model(torch.zeros(1, 20, 8, 8))

        self.assertEqual(
            trunk_output.shape,
            torch.Size([1, model_module.FUNNEL_ATTENTION_CHANNELS, 8, 8]),
        )
        self.assertEqual(output.shape, torch.Size([1, 1]))
        self.assertEqual(
            sum(p.numel() for p in model.parameters()),
            336509,
        )

    def test_supported_model_variants_include_parallel_fusion(self):
        self.assertIn(
            model_module.MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE,
            model_module.SUPPORTED_MODEL_VARIANTS,
        )
        self.assertIn(
            model_module.MODEL_VARIANT_PARALLEL_CNN_ATTN_ALIGNED_ADD,
            model_module.SUPPORTED_MODEL_VARIANTS,
        )
        self.assertIn(
            model_module.MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE_NO_MATERIAL,
            model_module.SUPPORTED_MODEL_VARIANTS,
        )
        self.assertIn(
            (
                model_module.
                MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_FUSE_NO_MATERIAL
            ),
            model_module.SUPPORTED_MODEL_VARIANTS,
        )
        self.assertIn(
            (
                model_module.
                MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_LATEEVIDENCE_NO_MATERIAL
            ),
            model_module.SUPPORTED_MODEL_VARIANTS,
        )
        self.assertIn(
            model_module.MODEL_VARIANT_FUNNEL_CNN_ATTENTION,
            model_module.SUPPORTED_MODEL_VARIANTS,
        )
        self.assertIn(
            model_module.MODEL_VARIANT_FUNNEL_INTERLEAVED_ATTENTION,
            model_module.SUPPORTED_MODEL_VARIANTS,
        )

    def test_checkpoint_without_variant_defaults_to_current_model_variant(self):
        self.assertEqual(
            model_module.model_variant_from_checkpoint({}),
            model_module.MODEL_VARIANT_STACKED_EDGE_GATE_FFN,
        )


if __name__ == "__main__":
    unittest.main()
