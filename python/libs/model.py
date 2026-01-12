import torch
import torch.nn.functional as F

from libs.movement import TOTAL_MOVES

# Order matches chess.PIECE_TYPES (pawn, knight, bishop, rook, queen, king)
_MATERIAL_VALUES = torch.tensor(
    [1.0, 3.0, 3.0, 5.0, 9.0, 0.0], dtype=torch.float32)
_MATERIAL_ALPHA = 5.0


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
        x: Board tensor shaped [B, 18, 8, 8] (before coords are added).
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
        # Channels: 0-4 meta, 5 en-passant, 6-11 ours, 12-17 theirs.
        our_pieces = x[:, 6:12]
        enemy_pieces = x[:, 12:18]
        weights = material_weights.to(
            dtype=x.dtype, device=x.device).view(1, 6, 1, 1)
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


class DepthwiseSeparableConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)

        return out


class CoordinateAttention(torch.nn.Module):
    def __init__(self, channels: int, reduction: int = 32):
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


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dsc1 = DepthwiseSeparableConv(channels, channels)
        self.relu1 = torch.nn.Hardswish(inplace=True)
        self.dsc2 = DepthwiseSeparableConv(channels, channels)
        self.ca = CoordinateAttention(channels)
        self.relu2 = torch.nn.Hardswish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dsc1(x)
        out = self.relu1(out)
        out = self.dsc2(out)
        out = self.ca(out)
        out += x
        out = self.relu2(out)

        return out


class ModelFull(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.add_coords = AddCoords(8, 8)
        self.initial_block = torch.nn.Sequential(
            torch.nn.Conv2d(22, 48, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(48),
            torch.nn.Hardswish(inplace=True),
        )

        self.residual_blocks = torch.nn.Sequential(
            *[ResidualBlock(48) for _ in range(8)],
        )

        self.register_buffer("material_weights", _MATERIAL_VALUES)

        self.value_conv = torch.nn.Sequential(
            torch.nn.Conv2d(48, 2, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(2),
            torch.nn.Hardswish(inplace=True),
            torch.nn.Flatten(),
        )
        self.value_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * 8 * 8 + 1, 64),
            torch.nn.Hardswish(inplace=True),
            torch.nn.Linear(64, 1),
        )
        self.policy_head = torch.nn.Sequential(
            torch.nn.Conv2d(48, 16, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.Hardswish(inplace=True),
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 8 * 8, TOTAL_MOVES),
        )

    def forward(
        self, x: torch.Tensor, material_diff: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        material_feature = _material_feature(
            x=x,
            material_weights=self.material_weights,
            material_scale=1.0,
            material_diff=material_diff,
        )

        out = self.add_coords(x)
        out = self.initial_block(out)
        out = self.residual_blocks(out)

        value_flat = self.value_conv(out)
        value = self.value_mlp(
            torch.cat([value_flat, material_feature], dim=1))

        return value, self.policy_head(out)

    def forward_eval(
        self, x: torch.Tensor, material_diff: torch.Tensor | None = None
    ) -> torch.Tensor:
        material_feature = _material_feature(
            x=x,
            material_weights=self.material_weights,
            material_scale=1.0,
            material_diff=material_diff,
        )

        out = self.add_coords(x)
        out = self.initial_block(out)
        out = self.residual_blocks(out)

        value_flat = self.value_conv(out)
        return self.value_mlp(torch.cat([value_flat, material_feature], dim=1))

    def forward_policy(self, x: torch.Tensor) -> torch.Tensor:
        out = self.add_coords(x)
        out = self.initial_block(out)
        out = self.residual_blocks(out)

        return self.policy_head(out)


class EvalOnlyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.add_coords = AddCoords(8, 8)
        self.initial_block = torch.nn.Sequential(
            torch.nn.Conv2d(22, 48, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(48),
            torch.nn.Hardswish(inplace=True),
        )

        self.residual_blocks = torch.nn.Sequential(
            *[ResidualBlock(48) for _ in range(8)],
        )

        self.register_buffer("material_weights", _MATERIAL_VALUES)

        self.value_conv = torch.nn.Sequential(
            torch.nn.Conv2d(48, 2, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(2),
            torch.nn.Hardswish(inplace=True),
            torch.nn.Flatten(),
        )
        self.value_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * 8 * 8 + 1, 64),
            torch.nn.Hardswish(inplace=True),
            torch.nn.Linear(64, 1),
        )

    def forward(
        self, x: torch.Tensor, material_diff: torch.Tensor | None = None
    ) -> torch.Tensor:
        material_feature = _material_feature(
            x=x,
            material_weights=self.material_weights,
            material_scale=1.0,
            material_diff=material_diff,
        )

        out = self.add_coords(x)
        out = self.initial_block(out)
        out = self.residual_blocks(out)

        value_flat = self.value_conv(out)
        return self.value_mlp(torch.cat([value_flat, material_feature], dim=1))
