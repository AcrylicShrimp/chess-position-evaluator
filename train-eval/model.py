import torch


class SqueezeExcitationBlock(torch.nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 8):
        super().__init__()
        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channels, channels // reduction_ratio, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channels // reduction_ratio, channels, bias=False),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        out = self.global_avg_pool(x)
        out = out.view(b, c)
        out = self.fc(out)
        out = out.view(b, c, 1, 1)
        return x * out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(channels),
        )
        self.se = SqueezeExcitationBlock(channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out = self.se(out)
        out += x
        return self.relu(out)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_block = torch.nn.Sequential(
            torch.nn.Conv2d(20, 128, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
        )

        self.residual_blocks = torch.nn.Sequential(
            *[ResidualBlock(128) for _ in range(6)],
        )

        self.value_head = torch.nn.Sequential(
            torch.nn.Conv2d(128, 32, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 8 * 8, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.initial_block(x)
        out = self.residual_blocks(out)

        return self.value_head(out)
