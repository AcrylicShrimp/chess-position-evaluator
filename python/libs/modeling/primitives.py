import torch


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
