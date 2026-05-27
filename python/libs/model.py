import math

import torch

from libs.movement import TOTAL_MOVES

# Order matches chess.PIECE_TYPES (pawn, knight, bishop, rook, queen, king)
_MATERIAL_VALUES = torch.tensor(
    [1.0, 3.0, 3.0, 5.0, 9.0, 0.0], dtype=torch.float32)
_MATERIAL_ALPHA = 5.0
_MATERIAL_LOGIT_SCALE = math.log(10.0) / 4.0
_MATERIAL_FEATURE_COUNT = 3


CHANNELS = 256
BLOCKS = 6
ATTENTION_AFTER_BLOCK = 3
MODEL_VARIANT_STACKED_EDGE_GATE_FFN = "stacked-edge-gate-ffn"
MODEL_VARIANT_ONE_LAYER_EDGE_GATE = "one-layer-edge-gate"
MODEL_VARIANT_NO_ATTENTION = "no-attention"
MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE = "parallel-cnn-attn-fuse"
MODEL_VARIANT_PARALLEL_CNN_ATTN_ALIGNED_ADD = "parallel-cnn-attn-aligned-add"
MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE_NO_MATERIAL = (
    "parallel-cnn-attn-fuse-no-material"
)
MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_FUSE_NO_MATERIAL = (
    "parallel-cnn-attn-kedge-fuse-no-material"
)
MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_LATEEVIDENCE_NO_MATERIAL = (
    "parallel-cnn-attn-kedge-lateevidence-no-material"
)
MODEL_VARIANT_FUNNEL_CNN_ATTENTION = "funnel-cnn224-160-128-attn6-edgegate"
SUPPORTED_MODEL_VARIANTS = (
    MODEL_VARIANT_STACKED_EDGE_GATE_FFN,
    MODEL_VARIANT_ONE_LAYER_EDGE_GATE,
    MODEL_VARIANT_NO_ATTENTION,
    MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE,
    MODEL_VARIANT_PARALLEL_CNN_ATTN_ALIGNED_ADD,
    MODEL_VARIANT_PARALLEL_CNN_ATTN_FUSE_NO_MATERIAL,
    MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_FUSE_NO_MATERIAL,
    MODEL_VARIANT_PARALLEL_CNN_ATTN_KEDGE_LATEEVIDENCE_NO_MATERIAL,
    MODEL_VARIANT_FUNNEL_CNN_ATTENTION,
)
EDGE_GATE_QK = "qk"
EDGE_GATE_Q_ONLY = "q-only"
EDGE_GATE_K_ONLY = "k-only"
ATTENTION_HEADS = 4
ATTENTION_HEAD_DIM = 16
ATTENTION_DIM = ATTENTION_HEADS * ATTENTION_HEAD_DIM
ATTENTION_LAYERS = 3
ATTENTION_FFN_HIDDEN = 64
FUNNEL_INITIAL_CHANNELS = 224
FUNNEL_MID_CHANNELS = 160
FUNNEL_ATTENTION_CHANNELS = 128
FUNNEL_BLOCKS_PER_STAGE = 2
FUNNEL_ATTENTION_LAYERS = 6
BOARD_ATTENTION_SIZE = 8
BOARD_ATTENTION_RELATIONS = (
    "same_square",
    "same_rank",
    "same_file",
    "same_diagonal",
    "same_anti_diagonal",
    "knight_move",
    "king_adjacent",
    "our_pawn_attack_geometry",
    "their_pawn_attack_geometry",
)


def _board_attention_geometry() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(BOARD_ATTENTION_SIZE),
            torch.arange(BOARD_ATTENTION_SIZE),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)

    query_rows = coords[:, 0].view(-1, 1)
    query_files = coords[:, 1].view(-1, 1)
    key_rows = coords[:, 0].view(1, -1)
    key_files = coords[:, 1].view(1, -1)

    dr = key_rows - query_rows
    df = key_files - query_files
    abs_dr = dr.abs()
    abs_df = df.abs()
    distance = torch.maximum(abs_dr, abs_df).long()

    relation_masks = torch.stack(
        [
            (dr == 0) & (df == 0),
            (dr == 0) & (df != 0),
            (df == 0) & (dr != 0),
            (dr == df) & (dr != 0),
            (dr == -df) & (dr != 0),
            ((abs_dr == 1) & (abs_df == 2))
            | ((abs_dr == 2) & (abs_df == 1)),
            distance == 1,
            (dr == 1) & (abs_df == 1),
            (dr == -1) & (abs_df == 1),
        ],
        dim=0,
    )

    return (
        relation_masks.reshape(len(BOARD_ATTENTION_RELATIONS), -1).float(),
        distance.reshape(-1),
        torch.nn.functional.one_hot(
            distance.reshape(-1),
            num_classes=BOARD_ATTENTION_SIZE,
        ).transpose(0, 1).float(),
    )


def _material_diff_from_board(
    board: torch.Tensor,
    material_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Compute raw material difference (our material - enemy material).

    Args:
        board: Board tensor shaped [B, 20, 8, 8] before coords are added.
        material_weights: Piece values shaped [6].

    Returns:
        Tensor of shape [B].
    """
    if not torch.jit.is_tracing() and (board.dim() != 4 or board.shape[1] != 20):
        raise ValueError(
            "material diff requires a board tensor shaped [B, 20, 8, 8]"
        )

    # Channels: 0-4 meta, 5 en-passant, 6-11 ours, 12-17 theirs, 18-19 heatmaps.
    our_pieces = board[:, 6:12]
    enemy_pieces = board[:, 12:18]
    weights = material_weights.to(dtype=board.dtype, device=board.device).view(
        1, 6, 1, 1
    )
    our_score = (our_pieces * weights).sum(dim=(1, 2, 3))
    enemy_score = (enemy_pieces * weights).sum(dim=(1, 2, 3))
    return our_score - enemy_score


def _material_feature(
    x: torch.Tensor,
    material_weights: torch.Tensor,
    material_scale: float,
    material_diff: torch.Tensor | None = None,
    alpha: float = _MATERIAL_ALPHA,
) -> torch.Tensor:
    """
    Compute (or inject) a normalized material difference feature.

    Args:
        x: Board tensor shaped [B, 20, 8, 8] when material_diff is not provided.
        material_weights: Piece values shaped [6].
        material_scale: Scalar multiplier.
        material_diff: Optional precomputed raw material diff (my - enemy),
            shaped [B] or [B, 1]. If None, it is derived from x.
        alpha: Normalization factor for tanh.

    Returns:
        Tensor of shape [B, 1] to concatenate after flattening.
    """
    batch = x.shape[0]

    if material_diff is None:
        diff = _material_diff_from_board(x, material_weights)
    else:
        diff = material_diff.to(device=x.device, dtype=x.dtype).reshape(batch)

    normalized = torch.tanh(diff / alpha)
    scaled = material_scale * normalized
    return scaled.unsqueeze(1)


def _material_prior_logit(
    material_diff: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    diff = material_diff.to(device=device, dtype=dtype).reshape(-1, 1)
    return diff * _MATERIAL_LOGIT_SCALE


def _material_input_features(
    x: torch.Tensor,
    material_diff: torch.Tensor,
    alpha: float = _MATERIAL_ALPHA,
) -> torch.Tensor:
    batch = x.shape[0]
    diff = material_diff.to(device=x.device, dtype=x.dtype).reshape(batch)
    signed_material = torch.tanh(diff / alpha)
    abs_material = torch.tanh(diff.abs() / alpha)
    material_prior_prob = torch.sigmoid(diff * _MATERIAL_LOGIT_SCALE)
    return torch.stack(
        (signed_material, abs_material, material_prior_prob),
        dim=1,
    )


class AddCoords(torch.nn.Module):
    def __init__(self, height, width):
        super().__init__()
        y_coords = (
            2.0
            * torch.arange(height).unsqueeze(1).expand(height, width)
            / (height - 1.0)
            - 1.0
        )
        x_coords = (
            2.0 * torch.arange(width).unsqueeze(0).expand(height,
                                                          width) / (width - 1.0)
            - 1.0
        )

        d1_coords = x_coords + y_coords
        d1_coords = d1_coords / d1_coords.abs().max()

        d2_coords = x_coords - y_coords
        d2_coords = d2_coords / d2_coords.abs().max()

        coords = torch.stack(
            (y_coords, x_coords, d1_coords, d2_coords), dim=0
        ).unsqueeze(0)

        self.register_buffer("coords", coords)

    def forward(self, x):
        batch_size = x.shape[0]
        coords_batch = self.coords.expand(batch_size, -1, -1, -1)
        coords_batch = coords_batch.to(dtype=x.dtype, device=x.device)
        return torch.cat([x, coords_batch], dim=1)


class CoordinateAttention(torch.nn.Module):
    def __init__(self, channels: int, reduction: int):
        super().__init__()
        reduced = max(8, channels // reduction)
        self.reduce = torch.nn.Conv2d(
            channels, reduced, kernel_size=1, bias=True)
        self.bn = torch.nn.BatchNorm2d(reduced)
        self.act = torch.nn.Hardswish(inplace=True)
        self.attn_h = torch.nn.Conv2d(
            reduced, channels, kernel_size=1, bias=True)
        self.attn_w = torch.nn.Conv2d(
            reduced, channels, kernel_size=1, bias=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _b, _c, h, w = x.shape

        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.reduce(y)
        y = self.bn(y)
        y = self.act(y)

        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)

        a_h = self.sigmoid(self.attn_h(y_h))
        a_w = self.sigmoid(self.attn_w(y_w))

        return x * a_h * a_w


class NaiveBoardSelfAttention(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        heads: int,
        head_dim: int,
        edge_gate_mode: str = EDGE_GATE_QK,
    ):
        super().__init__()
        if heads <= 0:
            raise ValueError("board attention requires at least one head")
        if head_dim <= 0:
            raise ValueError(
                "board attention requires a positive head dimension")
        if edge_gate_mode not in {
            EDGE_GATE_QK,
            EDGE_GATE_Q_ONLY,
            EDGE_GATE_K_ONLY,
        }:
            raise ValueError(
                f"unsupported edge gate mode {edge_gate_mode!r}"
            )

        self.channels = channels
        self.heads = heads
        self.head_dim = head_dim
        self.attn_dim = heads * head_dim
        self.edge_gate_mode = edge_gate_mode
        self.scale = head_dim**-0.5
        (
            relation_masks_flat,
            distance_index_flat,
            distance_masks_flat,
        ) = _board_attention_geometry()

        self.q_proj = torch.nn.Conv2d(
            channels, self.attn_dim, kernel_size=1, bias=False
        )
        self.k_proj = torch.nn.Conv2d(
            channels, self.attn_dim, kernel_size=1, bias=False
        )
        self.v_proj = torch.nn.Conv2d(
            channels, self.attn_dim, kernel_size=1, bias=False
        )
        self.out_proj = torch.nn.Conv2d(
            self.attn_dim, channels, kernel_size=1, bias=False
        )
        if edge_gate_mode in {EDGE_GATE_QK, EDGE_GATE_Q_ONLY}:
            self.rel_gate_q = torch.nn.Parameter(
                torch.zeros(heads, len(BOARD_ATTENTION_RELATIONS), head_dim)
            )
            self.dist_gate_q = torch.nn.Parameter(
                torch.zeros(heads, BOARD_ATTENTION_SIZE, head_dim)
            )
        else:
            self.register_parameter("rel_gate_q", None)
            self.register_parameter("dist_gate_q", None)

        if edge_gate_mode in {EDGE_GATE_QK, EDGE_GATE_K_ONLY}:
            self.rel_gate_k = torch.nn.Parameter(
                torch.zeros(heads, len(BOARD_ATTENTION_RELATIONS), head_dim)
            )
            self.dist_gate_k = torch.nn.Parameter(
                torch.zeros(heads, BOARD_ATTENTION_SIZE, head_dim)
            )
        else:
            self.register_parameter("rel_gate_k", None)
            self.register_parameter("dist_gate_k", None)
        self.register_buffer("relation_masks_flat", relation_masks_flat)
        self.register_buffer("distance_index_flat", distance_index_flat)
        self.register_buffer("distance_masks_flat", distance_masks_flat)

    def _edge_gate_logits(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        tokens: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        relation_masks = self.relation_masks_flat.reshape(
            len(BOARD_ATTENTION_RELATIONS), tokens, tokens
        ).to(dtype=dtype, device=q.device)
        distance_masks = self.distance_masks_flat.reshape(
            BOARD_ATTENTION_SIZE, tokens, tokens
        ).to(dtype=dtype, device=q.device)

        edge_logits = None

        if self.edge_gate_mode in {EDGE_GATE_QK, EDGE_GATE_Q_ONLY}:
            edge_q_vectors = torch.einsum(
                "rij,hrd->hijd",
                relation_masks,
                self.rel_gate_q.to(dtype=dtype),
            ) + torch.einsum(
                "mij,hmd->hijd",
                distance_masks,
                self.dist_gate_q.to(dtype=dtype),
            )
            edge_logits = torch.einsum(
                "bhid,hijd->bhij",
                q.to(dtype=dtype),
                edge_q_vectors,
            )

        if self.edge_gate_mode in {EDGE_GATE_QK, EDGE_GATE_K_ONLY}:
            edge_k_vectors = torch.einsum(
                "rij,hrd->hijd",
                relation_masks,
                self.rel_gate_k.to(dtype=dtype),
            ) + torch.einsum(
                "mij,hmd->hijd",
                distance_masks,
                self.dist_gate_k.to(dtype=dtype),
            )
            edge_k_logits = torch.einsum(
                "bhjd,hijd->bhij",
                k.to(dtype=dtype),
                edge_k_vectors,
            )
            if edge_logits is None:
                edge_logits = edge_k_logits
            else:
                edge_logits = edge_logits + edge_k_logits

        return edge_logits * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing() and (x.dim() != 4 or x.shape[1] != self.channels):
            raise ValueError(
                f"board attention requires [B, {self.channels}, H, W] input"
            )

        batch, _channels, height, width = x.shape
        if not torch.jit.is_tracing() and (
            height != BOARD_ATTENTION_SIZE or width != BOARD_ATTENTION_SIZE
        ):
            raise ValueError(
                "board attention edge gate requires 8x8 spatial input"
            )

        tokens = height * width

        q = self.q_proj(x).reshape(
            batch, self.heads, self.head_dim, tokens
        ).transpose(2, 3)
        k = self.k_proj(x).reshape(batch, self.heads, self.head_dim, tokens)
        k_tokens = k.transpose(2, 3)
        v = self.v_proj(x).reshape(
            batch, self.heads, self.head_dim, tokens
        ).transpose(2, 3)

        logits = torch.matmul(q, k) * self.scale
        edge_gate_logits = self._edge_gate_logits(
            q,
            k_tokens,
            tokens,
            logits.dtype,
        )

        weights = torch.softmax(logits + edge_gate_logits, dim=-1)
        context = torch.matmul(weights, v)
        context = context.transpose(2, 3).reshape(
            batch, self.attn_dim, height, width
        )

        return x + self.out_proj(context)


class BoardAttentionFFN(torch.nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        if hidden <= 0:
            raise ValueError(
                "board attention FFN requires a positive hidden size")

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(hidden),
            torch.nn.Hardswish(inplace=True),
            torch.nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class BoardAttentionBlock(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        heads: int,
        head_dim: int,
        ffn_hidden: int,
        edge_gate_mode: str = EDGE_GATE_QK,
    ):
        super().__init__()
        self.attention = NaiveBoardSelfAttention(
            channels,
            heads,
            head_dim,
            edge_gate_mode=edge_gate_mode,
        )
        self.ffn = BoardAttentionFFN(channels, ffn_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        return self.ffn(x)


class BoardAttentionStack(torch.nn.Module):
    def __init__(
        self,
        layers: int,
        channels: int,
        heads: int,
        head_dim: int,
        ffn_hidden: int,
        edge_gate_mode: str = EDGE_GATE_QK,
    ):
        super().__init__()
        if layers <= 0:
            raise ValueError(
                "board attention stack requires at least one layer")

        self.layers = torch.nn.ModuleList(
            [
                BoardAttentionBlock(
                    channels,
                    heads,
                    head_dim,
                    ffn_hidden,
                    edge_gate_mode=edge_gate_mode,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class IdentityBoardAttention(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ParallelCnnAttentionTrunk(torch.nn.Module):
    def __init__(self, edge_gate_mode: str = EDGE_GATE_QK):
        super().__init__()
        self.shared_blocks = torch.nn.Sequential(
            *[
                GhostShuffleBlock(CHANNELS, ratio=4)
                for _ in range(ATTENTION_AFTER_BLOCK)
            ]
        )
        self.local_blocks = torch.nn.Sequential(
            *[
                GhostShuffleBlock(CHANNELS, ratio=4)
                for _ in range(BLOCKS - ATTENTION_AFTER_BLOCK)
            ]
        )
        self.global_blocks = BoardAttentionStack(
            ATTENTION_LAYERS,
            CHANNELS,
            ATTENTION_HEADS,
            ATTENTION_HEAD_DIM,
            ATTENTION_FFN_HIDDEN,
            edge_gate_mode=edge_gate_mode,
        )
        self.fuse = torch.nn.Sequential(
            torch.nn.Conv2d(CHANNELS * 2, CHANNELS, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(CHANNELS),
            torch.nn.Hardswish(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_blocks(x)
        local = self.local_blocks(shared)
        global_ = self.global_blocks(shared)
        return self.fuse(torch.cat([local, global_], dim=1))


class ParallelCnnAttentionAlignedAddTrunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_blocks = torch.nn.Sequential(
            *[
                GhostShuffleBlock(CHANNELS, ratio=4)
                for _ in range(ATTENTION_AFTER_BLOCK)
            ]
        )
        self.local_blocks = torch.nn.Sequential(
            *[
                GhostShuffleBlock(CHANNELS, ratio=4)
                for _ in range(BLOCKS - ATTENTION_AFTER_BLOCK)
            ]
        )
        self.global_blocks = BoardAttentionStack(
            ATTENTION_LAYERS,
            CHANNELS,
            ATTENTION_HEADS,
            ATTENTION_HEAD_DIM,
            ATTENTION_FFN_HIDDEN,
        )
        self.local_aligner = torch.nn.Sequential(
            torch.nn.BatchNorm2d(CHANNELS),
            torch.nn.Conv2d(CHANNELS, CHANNELS, kernel_size=1, bias=False),
        )
        self.global_aligner = torch.nn.Sequential(
            torch.nn.BatchNorm2d(CHANNELS),
            torch.nn.Conv2d(CHANNELS, CHANNELS, kernel_size=1, bias=False),
        )
        self.fuse = torch.nn.Sequential(
            torch.nn.BatchNorm2d(CHANNELS),
            torch.nn.Hardswish(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_blocks(x)
        local = self.local_blocks(shared)
        global_ = self.global_blocks(shared)
        local_aligned = self.local_aligner(local)
        global_aligned = self.global_aligner(global_)
        return self.fuse(local_aligned + global_aligned)


class ParallelCnnAttentionKEdgeLateEvidenceTrunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_blocks = torch.nn.Sequential(
            *[
                GhostShuffleBlock(CHANNELS, ratio=4)
                for _ in range(ATTENTION_AFTER_BLOCK)
            ]
        )
        self.local_blocks = torch.nn.Sequential(
            *[
                GhostShuffleBlock(CHANNELS, ratio=4)
                for _ in range(BLOCKS - ATTENTION_AFTER_BLOCK)
            ]
        )
        self.global_blocks = BoardAttentionStack(
            ATTENTION_LAYERS,
            CHANNELS,
            ATTENTION_HEADS,
            ATTENTION_HEAD_DIM,
            ATTENTION_FFN_HIDDEN,
            edge_gate_mode=EDGE_GATE_K_ONLY,
        )
        self.local_evidence = torch.nn.Sequential(
            torch.nn.Conv2d(CHANNELS, 2, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(2),
            torch.nn.Hardswish(inplace=True),
        )
        self.global_evidence = torch.nn.Sequential(
            torch.nn.Conv2d(CHANNELS, 2, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(2),
            torch.nn.Hardswish(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_blocks(x)
        local = self.local_blocks(shared)
        global_ = self.global_blocks(shared)
        local_evidence = self.local_evidence(local)
        global_evidence = self.global_evidence(global_)
        return torch.cat([local_evidence, global_evidence], dim=1)


class ChannelProjection(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Hardswish(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FunnelCnnAttentionTrunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wide_blocks = torch.nn.Sequential(
            *[
                GhostShuffleBlock(FUNNEL_INITIAL_CHANNELS, ratio=4)
                for _ in range(FUNNEL_BLOCKS_PER_STAGE)
            ]
        )
        self.compress_wide_to_mid = ChannelProjection(
            FUNNEL_INITIAL_CHANNELS,
            FUNNEL_MID_CHANNELS,
        )
        self.mid_blocks = torch.nn.Sequential(
            *[
                GhostShuffleBlock(FUNNEL_MID_CHANNELS, ratio=4)
                for _ in range(FUNNEL_BLOCKS_PER_STAGE)
            ]
        )
        self.compress_mid_to_attention = ChannelProjection(
            FUNNEL_MID_CHANNELS,
            FUNNEL_ATTENTION_CHANNELS,
        )
        self.narrow_blocks = torch.nn.Sequential(
            *[
                GhostShuffleBlock(FUNNEL_ATTENTION_CHANNELS, ratio=4)
                for _ in range(FUNNEL_BLOCKS_PER_STAGE)
            ]
        )
        self.attention_blocks = BoardAttentionStack(
            FUNNEL_ATTENTION_LAYERS,
            FUNNEL_ATTENTION_CHANNELS,
            ATTENTION_HEADS,
            ATTENTION_HEAD_DIM,
            ATTENTION_FFN_HIDDEN,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wide_blocks(x)
        x = self.compress_wide_to_mid(x)
        x = self.mid_blocks(x)
        x = self.compress_mid_to_attention(x)
        x = self.narrow_blocks(x)
        return self.attention_blocks(x)


def build_board_attention(model_variant: str) -> torch.nn.Module:
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


class GhostModule(torch.nn.Module):
    def __init__(
        self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True
    ):
        super().__init__()
        self.oup = oup
        init_channels = max(8, oup // ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False
            ),
            torch.nn.BatchNorm2d(init_channels),
            torch.nn.Hardswish(
                inplace=True) if relu else torch.nn.Sequential(),
        )
        self.cheap_operation = torch.nn.Sequential(
            torch.nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=False,
            ),
            torch.nn.BatchNorm2d(new_channels),
            torch.nn.Hardswish(
                inplace=True) if relu else torch.nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.oup, :, :]


class ChannelShuffle(torch.nn.Module):
    def __init__(self, groups: int = 2):
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        x = x.reshape(batch, self.groups, channels //
                      self.groups, height, width)
        x = x.transpose(1, 2).contiguous()
        return x.reshape(batch, channels, height, width)


class GhostShuffleBlock(torch.nn.Module):
    def __init__(self, channels: int, ratio: int = 2, dw_kernel_size: int = 3):
        super().__init__()

        mid_channels = channels // 2

        self.branch2 = torch.nn.Sequential(
            GhostModule(
                mid_channels, mid_channels, kernel_size=1, ratio=ratio, relu=True
            ),
            torch.nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=dw_kernel_size // 2,
                groups=mid_channels,
                bias=False,
            ),
            torch.nn.BatchNorm2d(mid_channels),
            CoordinateAttention(mid_channels, 16),
            GhostModule(
                mid_channels, mid_channels, kernel_size=1, ratio=ratio, relu=False
            ),
            torch.nn.Hardswish(inplace=True),
        )

        self.shuffle = ChannelShuffle(groups=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        x2 = self.branch2(x2)
        out = torch.cat((x1, x2), dim=1)
        return self.shuffle(out)


def _run_trunk_blocks(
    x: torch.Tensor,
    blocks: torch.nn.Sequential,
    board_attention: torch.nn.Module,
) -> torch.Tensor:
    for index, block in enumerate(blocks):
        x = block(x)
        if index + 1 == ATTENTION_AFTER_BLOCK:
            x = board_attention(x)
    return x


class ValueHead(torch.nn.Module):
    def __init__(self, channels: int = CHANNELS):
        super().__init__()
        self.channels = channels
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(2),
            torch.nn.Hardswish(inplace=True),
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * 8 * 8 + 1, 64),
            torch.nn.Hardswish(inplace=True),
            torch.nn.Linear(64, 1),
        )

        self.register_buffer("material_weights", _MATERIAL_VALUES)

    def forward(
        self, x: torch.Tensor, material_diff: torch.Tensor | None = None
    ) -> torch.Tensor:
        if material_diff is None:
            raise ValueError("ValueHead requires explicit material_diff")

        material_feature = _material_feature(
            x=x,
            material_weights=self.material_weights,
            material_scale=1.0,
            material_diff=material_diff,
        )

        out = self.conv(x)
        out = out.flatten(start_dim=1)
        out = torch.cat([out, material_feature], dim=1)
        return self.mlp(out)


class ResidualValueHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(CHANNELS, 2, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(2),
            torch.nn.Hardswish(inplace=True),
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * 8 * 8, 64),
            torch.nn.Hardswish(inplace=True),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = out.flatten(start_dim=1)
        return self.mlp(out)


class MaterialFeatureValueHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(CHANNELS, 2, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(2),
            torch.nn.Hardswish(inplace=True),
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * 8 * 8 + _MATERIAL_FEATURE_COUNT, 64),
            torch.nn.Hardswish(inplace=True),
            torch.nn.Linear(64, 1),
        )

    def forward(
        self, x: torch.Tensor, material_diff: torch.Tensor | None = None
    ) -> torch.Tensor:
        if material_diff is None:
            raise ValueError(
                "MaterialFeatureValueHead requires explicit material_diff")

        material_features = _material_input_features(x, material_diff)
        out = self.conv(x)
        out = out.flatten(start_dim=1)
        out = torch.cat([out, material_features], dim=1)
        return self.mlp(out)


class LateEvidenceValueHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4 * 8 * 8, 64),
            torch.nn.Hardswish(inplace=True),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing() and (x.dim() != 4 or x.shape[1] != 4):
            raise ValueError(
                "late evidence value head requires [B, 4, 8, 8] input"
            )
        out = x.flatten(start_dim=1)
        return self.mlp(out)


class PolicyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(CHANNELS, 16, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.Hardswish(inplace=True),
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(16 * 8 * 8, TOTAL_MOVES),
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.flatten(start_dim=1)
        return self.mlp(out)


class ModelFull(torch.nn.Module):
    def __init__(self, model_variant: str = MODEL_VARIANT_STACKED_EDGE_GATE_FFN):
        super().__init__()
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
        self.model_variant = model_variant
        self.register_buffer(
            "material_weights",
            _MATERIAL_VALUES.clone(),
            persistent=False,
        )

        self.add_coords = AddCoords(8, 8)
        initial_channels = (
            FUNNEL_INITIAL_CHANNELS
            if model_variant == MODEL_VARIANT_FUNNEL_CNN_ATTENTION
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
