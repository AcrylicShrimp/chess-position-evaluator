import torch

from libs.movement import TOTAL_MOVES


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


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dsc1 = DepthwiseSeparableConv(channels, channels)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.dsc2 = DepthwiseSeparableConv(channels, channels)
        self.relu2 = torch.nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dsc1(x)
        out = self.relu1(out)
        out = self.dsc2(out)
        out += x
        out = self.relu2(out)

        return out


class ModelFull(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_block = torch.nn.Sequential(
            torch.nn.Conv2d(18, 64, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
        )

        self.residual_blocks = torch.nn.Sequential(
            *[ResidualBlock(64) for _ in range(4)],
        )

        self.value_head = torch.nn.Sequential(
            torch.nn.Conv2d(64, 2, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(),
            torch.nn.Linear(2 * 8 * 8, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1),
        )
        self.policy_head = torch.nn.Sequential(
            torch.nn.Conv2d(64, 16, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 8 * 8, TOTAL_MOVES),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.initial_block(x)
        out = self.residual_blocks(out)

        return (self.value_head(out), self.policy_head(out))

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        out = self.initial_block(x)
        out = self.residual_blocks(out)

        return self.value_head(out)

    def forward_policy(self, x: torch.Tensor) -> torch.Tensor:
        out = self.initial_block(x)
        out = self.residual_blocks(out)

        return self.policy_head(out)


class EvalOnlyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_block = torch.nn.Sequential(
            torch.nn.Conv2d(18, 64, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
        )

        self.residual_blocks = torch.nn.Sequential(
            *[ResidualBlock(64) for _ in range(4)],
        )

        self.value_head = torch.nn.Sequential(
            torch.nn.Conv2d(64, 2, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(),
            torch.nn.Linear(2 * 8 * 8, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.initial_block(x)
        out = self.residual_blocks(out)

        return self.value_head(out)
