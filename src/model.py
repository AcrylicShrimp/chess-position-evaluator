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
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(20, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
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
            torch.nn.Conv2d(256, 32, kernel_size=1),
        )

        self.cp_head = torch.nn.Sequential(
            torch.nn.Linear(32 * 8 * 8, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 1),
        )
        self.mate_head = torch.nn.Sequential(
            torch.nn.Linear(32 * 8 * 8, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 3),
            torch.nn.Tanh(),
        )

    def forward_cp(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out = out.reshape(-1, 32 * 8 * 8)
        return self.cp_head(out)

    def forward_mate(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out = out.reshape(-1, 32 * 8 * 8)
        return self.mate_head(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.forward_cp(x), self.forward_mate(x)], dim=1)
