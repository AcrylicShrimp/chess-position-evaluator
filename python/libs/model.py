import torch
import torch.nn.functional as F

from libs.movement import TOTAL_MOVES

# Order matches chess.PIECE_TYPES (pawn, knight, bishop, rook, queen, king)
_MATERIAL_VALUES = torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0], dtype=torch.float32)
_MATERIAL_ALPHA = 5.0


CHANNELS = 256
BLOCKS = 6


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
        x: Board tensor shaped [B, 20, 8, 8] (before coords are added).
        material_weights: Piece values shaped [6].
        material_scale: Learnable scalar gate.
        material_diff: Optional precomputed raw material diff (my - enemy),
            shaped [B] or [B, 1]. If None, it is derived from x.
        alpha: Normalization factor for tanh.

    Returns:
        Tensor of shape [B, 1] to concatenate after flattening.
    """
    batch = x.shape[0]

    if material_diff is None:
        # Channels: 0-4 meta, 5 en-passant, 6-11 ours, 12-17 theirs, 18-19 heatmaps.
        our_pieces = x[:, 6:12]
        enemy_pieces = x[:, 12:18]
        weights = material_weights.to(dtype=x.dtype, device=x.device).view(1, 6, 1, 1)
        our_score = (our_pieces * weights).sum(dim=(1, 2, 3))
        enemy_score = (enemy_pieces * weights).sum(dim=(1, 2, 3))
        diff = our_score - enemy_score
    else:
        diff = material_diff.to(device=x.device, dtype=x.dtype).reshape(batch)

    normalized = torch.tanh(diff / alpha)
    scaled = material_scale * normalized
    return scaled.unsqueeze(1)


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
            2.0 * torch.arange(width).unsqueeze(0).expand(height, width) / (width - 1.0)
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
        self.reduce = torch.nn.Conv2d(channels, reduced, kernel_size=1, bias=True)
        self.bn = torch.nn.BatchNorm2d(reduced)
        self.act = torch.nn.Hardswish(inplace=True)
        self.attn_h = torch.nn.Conv2d(reduced, channels, kernel_size=1, bias=True)
        self.attn_w = torch.nn.Conv2d(reduced, channels, kernel_size=1, bias=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _b, _c, h, w = x.shape

        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.reduce(y)
        y = self.bn(y)
        y = self.act(y)

        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)

        a_h = self.sigmoid(self.attn_h(y_h))
        a_w = self.sigmoid(self.attn_w(y_w))

        return x * a_h * a_w


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
            torch.nn.Hardswish(inplace=True) if relu else torch.nn.Sequential(),
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
            torch.nn.Hardswish(inplace=True) if relu else torch.nn.Sequential(),
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
        x = x.reshape(batch, self.groups, channels // self.groups, height, width)
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


class ValueHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(CHANNELS, 2, kernel_size=1, bias=False),
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
    def __init__(self):
        super().__init__()
        self.add_coords = AddCoords(8, 8)
        self.initial_block = torch.nn.Sequential(
            torch.nn.Conv2d(24, CHANNELS, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(CHANNELS),
            torch.nn.Hardswish(inplace=True),
        )

        self.blocks = torch.nn.Sequential(
            *[GhostShuffleBlock(CHANNELS, ratio=4) for _ in range(BLOCKS)],
        )

        self.value_head = ValueHead()
        self.policy_head = PolicyHead()

    def forward(
        self, x: torch.Tensor, material_diff: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.add_coords(x)
        out = self.initial_block(out)
        out = self.blocks(out)

        value = self.value_head(out, material_diff)
        policy = self.policy_head(out)

        return value, policy

    def forward_value(
        self, x: torch.Tensor, material_diff: torch.Tensor | None = None
    ) -> torch.Tensor:
        out = self.add_coords(x)
        out = self.initial_block(out)
        out = self.blocks(out)

        return self.value_head(out, material_diff)

    def forward_policy(self, x: torch.Tensor) -> torch.Tensor:
        out = self.add_coords(x)
        out = self.initial_block(out)
        out = self.blocks(out)

        return self.policy_head(out)


class ValueOnlyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.add_coords = AddCoords(8, 8)
        self.initial_block = torch.nn.Sequential(
            torch.nn.Conv2d(24, CHANNELS, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(CHANNELS),
            torch.nn.Hardswish(inplace=True),
        )

        self.blocks = torch.nn.Sequential(
            *[GhostShuffleBlock(CHANNELS, ratio=4) for _ in range(BLOCKS)],
        )

        self.value_head = ValueHead()

    def forward(
        self, x: torch.Tensor, material_diff: torch.Tensor | None = None
    ) -> torch.Tensor:
        out = self.add_coords(x)
        out = self.initial_block(out)
        out = self.blocks(out)

        return self.value_head(out, material_diff)
