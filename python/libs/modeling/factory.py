import torch

from libs.modeling.attention import (
    BoardAttentionStack,
    IdentityBoardAttention,
    NaiveBoardSelfAttention,
)
from libs.modeling.constants import (
    ATTENTION_AFTER_BLOCK,
    ATTENTION_FFN_HIDDEN,
    ATTENTION_HEAD_DIM,
    ATTENTION_HEADS,
    ATTENTION_LAYERS,
    BLOCKS,
    CHANNELS,
    EDGE_GATE_K_ONLY,
    FUNNEL_ATTENTION_CHANNELS,
    FUNNEL_INITIAL_CHANNELS,
    MODEL_VARIANT_FUNNEL_CNN_ATTENTION,
    MODEL_VARIANT_FUNNEL_DEPTH_REFRESH_ATTENTION,
    MODEL_VARIANT_FUNNEL_INTERLEAVED_ATTENTION,
    MODEL_VARIANT_NO_ATTENTION,
    MODEL_VARIANT_ONE_LAYER_EDGE_GATE,
    MODEL_VARIANT_PARALLEL_CNN_ATTN_ALIGNED_ADD,
    MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE,
    MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE_NO_MATERIAL,
    MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_FUSE_NO_MATERIAL,
    MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_LATEEVIDENCE_NO_MATERIAL,
    MODEL_VARIANT_STACKED_EDGE_GATE_FFN,
)
from libs.modeling.heads import (
    LateEvidenceValueHead,
    MaterialFeatureValueHead,
    PolicyHead,
    ResidualValueHead,
    ValueHead,
)
from libs.modeling.material import _MATERIAL_VALUES, _material_diff_from_board
from libs.modeling.primitives import AddCoords, GhostShuffleBlock
from libs.modeling.registry import get_model_variant_spec
from libs.modeling.trunks import (
    FunnelCnnAttentionTrunk,
    FunnelDepthRefreshAttentionTrunk,
    FunnelInterleavedAttentionTrunk,
    ParallelCnnAttentionAlignedAddTrunk,
    ParallelCnnAttentionKEdgeLateEvidenceTrunk,
    ParallelCnnAttentionTrunk,
    _run_trunk_blocks,
)


def build_board_attention(model_variant: str) -> torch.nn.Module:
    get_model_variant_spec(model_variant)

    if model_variant == MODEL_VARIANT_STACKED_EDGE_GATE_FFN:
        return BoardAttentionStack(
            ATTENTION_LAYERS,
            CHANNELS,
            ATTENTION_HEADS,
            ATTENTION_HEAD_DIM,
            ATTENTION_FFN_HIDDEN,
        )

    if model_variant == MODEL_VARIANT_ONE_LAYER_EDGE_GATE:
        return NaiveBoardSelfAttention(
            CHANNELS,
            ATTENTION_HEADS,
            ATTENTION_HEAD_DIM,
        )

    if model_variant == MODEL_VARIANT_NO_ATTENTION:
        return IdentityBoardAttention()

    allowed = ", ".join(
        [
            MODEL_VARIANT_STACKED_EDGE_GATE_FFN,
            MODEL_VARIANT_ONE_LAYER_EDGE_GATE,
            MODEL_VARIANT_NO_ATTENTION,
        ]
    )
    raise ValueError(
        f"unsupported board attention variant {model_variant!r}; "
        f"expected one of: {allowed}"
    )


def model_variant_from_checkpoint(checkpoint: dict) -> str:
    return checkpoint.get("model_variant", MODEL_VARIANT_STACKED_EDGE_GATE_FFN)


class ModelFull(torch.nn.Module):
    def __init__(self, model_variant: str = MODEL_VARIANT_STACKED_EDGE_GATE_FFN):
        super().__init__()
        get_model_variant_spec(model_variant)
        self.model_variant = model_variant
        self.add_coords = AddCoords(8, 8)
        self.initial_block = torch.nn.Sequential(
            torch.nn.Conv2d(24, CHANNELS, kernel_size=3,
                            padding=1, bias=False),
            torch.nn.BatchNorm2d(CHANNELS),
            torch.nn.Hardswish(inplace=True),
        )

        self.blocks = torch.nn.Sequential(
            *[GhostShuffleBlock(CHANNELS, ratio=4) for _ in range(BLOCKS)],
        )
        self.board_attention = build_board_attention(model_variant)

        self.value_head = ValueHead()
        self.policy_head = PolicyHead()

    def forward(
        self, x: torch.Tensor, material_diff: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if material_diff is None:
            material_diff = _material_diff_from_board(
                x, self.value_head.material_weights
            )

        out = self.add_coords(x)
        out = self.initial_block(out)
        out = _run_trunk_blocks(out, self.blocks, self.board_attention)

        value = self.value_head(out, material_diff)
        policy = self.policy_head(out)

        return value, policy

    def forward_value(
        self, x: torch.Tensor, material_diff: torch.Tensor | None = None
    ) -> torch.Tensor:
        if material_diff is None:
            material_diff = _material_diff_from_board(
                x, self.value_head.material_weights
            )

        out = self.add_coords(x)
        out = self.initial_block(out)
        out = _run_trunk_blocks(out, self.blocks, self.board_attention)

        return self.value_head(out, material_diff)

    def forward_policy(self, x: torch.Tensor) -> torch.Tensor:
        out = self.add_coords(x)
        out = self.initial_block(out)
        out = _run_trunk_blocks(out, self.blocks, self.board_attention)

        return self.policy_head(out)


class ValueOnlyModel(torch.nn.Module):
    def __init__(self, model_variant: str = MODEL_VARIANT_STACKED_EDGE_GATE_FFN):
        super().__init__()
        get_model_variant_spec(model_variant)
        self.model_variant = model_variant
        self.register_buffer(
            "material_weights",
            _MATERIAL_VALUES.clone(),
            persistent=False,
        )

        self.add_coords = AddCoords(8, 8)
        initial_channels = (
            FUNNEL_INITIAL_CHANNELS
            if model_variant
            in {
                MODEL_VARIANT_FUNNEL_CNN_ATTENTION,
                MODEL_VARIANT_FUNNEL_INTERLEAVED_ATTENTION,
                MODEL_VARIANT_FUNNEL_DEPTH_REFRESH_ATTENTION,
            }
            else CHANNELS
        )
        self.initial_block = torch.nn.Sequential(
            torch.nn.Conv2d(24, initial_channels, kernel_size=3,
                            padding=1, bias=False),
            torch.nn.BatchNorm2d(initial_channels),
            torch.nn.Hardswish(inplace=True),
        )

        if model_variant == MODEL_VARIANT_FUNNEL_CNN_ATTENTION:
            self.trunk = FunnelCnnAttentionTrunk()
            self.value_head = ValueHead(channels=FUNNEL_ATTENTION_CHANNELS)
        elif model_variant == MODEL_VARIANT_FUNNEL_INTERLEAVED_ATTENTION:
            self.trunk = FunnelInterleavedAttentionTrunk()
            self.value_head = ValueHead(channels=FUNNEL_ATTENTION_CHANNELS)
        elif model_variant == MODEL_VARIANT_FUNNEL_DEPTH_REFRESH_ATTENTION:
            self.trunk = FunnelDepthRefreshAttentionTrunk()
            self.value_head = ValueHead(channels=FUNNEL_ATTENTION_CHANNELS)
        elif model_variant == MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE:
            self.trunk = ParallelCnnAttentionTrunk()
            self.value_head = MaterialFeatureValueHead()
        elif model_variant == MODEL_VARIANT_PARALLEL_CNN_ATTN_ALIGNED_ADD:
            self.trunk = ParallelCnnAttentionAlignedAddTrunk()
            self.value_head = MaterialFeatureValueHead()
        elif model_variant == MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE_NO_MATERIAL:
            self.trunk = ParallelCnnAttentionTrunk()
            self.value_head = ResidualValueHead()
        elif (
            model_variant
            == MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_FUSE_NO_MATERIAL
        ):
            self.trunk = ParallelCnnAttentionTrunk(
                edge_gate_mode=EDGE_GATE_K_ONLY)
            self.value_head = ResidualValueHead()
        elif (
            model_variant
            == MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_LATEEVIDENCE_NO_MATERIAL
        ):
            self.trunk = ParallelCnnAttentionKEdgeLateEvidenceTrunk()
            self.value_head = LateEvidenceValueHead()
        else:
            self.blocks = torch.nn.Sequential(
                *[GhostShuffleBlock(CHANNELS, ratio=4) for _ in range(BLOCKS)],
            )
            self.board_attention = build_board_attention(model_variant)
            self.value_head = ValueHead()

    def forward(
        self, x: torch.Tensor, material_diff: torch.Tensor | None = None
    ) -> torch.Tensor:
        if (
            material_diff is None
            and self.model_variant
            not in {
                MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE_NO_MATERIAL,
                MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_FUSE_NO_MATERIAL,
                MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_LATEEVIDENCE_NO_MATERIAL,
            }
        ):
            material_diff = _material_diff_from_board(
                x, self.material_weights
            )

        out = self.add_coords(x)
        out = self.initial_block(out)

        if self.model_variant in {
            MODEL_VARIANT_FUNNEL_CNN_ATTENTION,
            MODEL_VARIANT_FUNNEL_INTERLEAVED_ATTENTION,
            MODEL_VARIANT_FUNNEL_DEPTH_REFRESH_ATTENTION,
        }:
            out = self.trunk(out)
            return self.value_head(out, material_diff)

        if self.model_variant in {
            MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE_NO_MATERIAL,
            MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_FUSE_NO_MATERIAL,
            MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_LATEEVIDENCE_NO_MATERIAL,
        }:
            out = self.trunk(out)
            return self.value_head(out)

        if self.model_variant in {
            MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE,
            MODEL_VARIANT_PARALLEL_CNN_ATTN_ALIGNED_ADD,
        }:
            out = self.trunk(out)
            return self.value_head(out, material_diff)

        out = _run_trunk_blocks(out, self.blocks, self.board_attention)
        return self.value_head(out, material_diff)
