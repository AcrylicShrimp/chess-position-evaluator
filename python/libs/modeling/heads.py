import torch

from libs.modeling.constants import CHANNELS
from libs.modeling.material import (
    _MATERIAL_FEATURE_COUNT,
    _MATERIAL_VALUES,
    _material_feature,
    _material_input_features,
)
from libs.movement import TOTAL_MOVES


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
