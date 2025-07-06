import torch


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
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out += x
        return self.relu(out)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_block = torch.nn.Sequential(
            torch.nn.Conv2d(20, 256, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
        )

        self.residual_blocks = torch.nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
        )

        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))

        head_input_size = 256

        self.cp_head = torch.nn.Sequential(
            torch.nn.Linear(head_input_size, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 1),
        )
        self.mate_head = torch.nn.Sequential(
            torch.nn.Linear(head_input_size, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.initial_block(x)
        out = self.residual_blocks(out)
        out = self.gap(out)
        out = torch.flatten(out, 1)

        cp_out = self.cp_head(out)
        mate_out = self.mate_head(out)

        return torch.cat([cp_out, mate_out], dim=1)
