import torch

from libs.modeling.constants import (
    BOARD_ATTENTION_RELATIONS,
    BOARD_ATTENTION_SIZE,
    EDGE_GATE_K_ONLY,
    EDGE_GATE_Q_ONLY,
    EDGE_GATE_QK,
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
