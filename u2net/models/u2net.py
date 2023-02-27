import torch
import torch.nn as nn
import torch.nn.functional as F


def auto_pad(kernel_size, padding=None, dilation=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1  # actual kernel-size
    if padding is None:
        padding = kernel_size // 2  # auto-pad
    return padding


class ConvNormActivation(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, padding=None, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=auto_pad(kernel_size=kernel_size, padding=padding, dilation=dilation),
            dilation=dilation
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Interpolate(nn.Module):
    """ Wrapper class for `nn.functional.interpolate()` """

    def __init__(self, scale_factor: int, mode: str = 'bilinear', align_corners: bool = None) -> None:
        super().__init__()
        self.fn = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fn(input=x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

        return x


# RSU-7
class RSU7(nn.Module):
    def __init__(self, in_channels=3, mid_channels=12, out_channels=3):
        super().__init__()

        self.conv_in = ConvNormActivation(in_channels, out_channels)

        self.conv1 = ConvNormActivation(out_channels, mid_channels)
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv2 = ConvNormActivation(mid_channels, mid_channels)
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv3 = ConvNormActivation(mid_channels, mid_channels)
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4 = ConvNormActivation(mid_channels, mid_channels)
        self.down4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv5 = ConvNormActivation(mid_channels, mid_channels)
        self.down5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv6 = ConvNormActivation(mid_channels, mid_channels)

        self.conv7 = ConvNormActivation(mid_channels, mid_channels, dilation=2)

        self.conv6d = ConvNormActivation(mid_channels * 2, mid_channels)
        self.conv5d = ConvNormActivation(mid_channels * 2, mid_channels)
        self.conv4d = ConvNormActivation(mid_channels * 2, mid_channels)
        self.conv3d = ConvNormActivation(mid_channels * 2, mid_channels)
        self.conv2d = ConvNormActivation(mid_channels * 2, mid_channels)
        self.conv1d = ConvNormActivation(mid_channels * 2, out_channels)

    def forward(self, x):
        hx = x
        hxin = self.conv_in(hx)

        hx1 = self.conv1(hxin)
        hx = self.down1(hx1)

        hx2 = self.conv2(hx)
        hx = self.down2(hx2)

        hx3 = self.conv3(hx)
        hx = self.down3(hx3)

        hx4 = self.conv4(hx)
        hx = self.down4(hx4)

        hx5 = self.conv5(hx)
        hx = self.down5(hx5)

        hx6 = self.conv6(hx)

        hx7 = self.conv7(hx6)

        hx6d = self.conv6d(torch.cat((hx7, hx6), 1))
        hx6dup = F.interpolate(hx6d, scale_factor=2)

        hx5d = self.conv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = F.interpolate(hx5d, scale_factor=2)

        hx4d = self.conv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, scale_factor=2)

        hx3d = self.conv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, scale_factor=2)

        hx2d = self.conv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, scale_factor=2)

        hx1d = self.conv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


# RSU-6
class RSU6(nn.Module):  # UNet06DRES(nn.Module):
    def __init__(self, in_channels=3, mid_ch=12, out_ch=3):
        super().__init__()

        self.conv_in = ConvNormActivation(in_channels, out_ch)

        self.conv1 = ConvNormActivation(out_ch, mid_ch)
        self.down1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv2 = ConvNormActivation(mid_ch, mid_ch)
        self.down2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3 = ConvNormActivation(mid_ch, mid_ch)
        self.down3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv4 = ConvNormActivation(mid_ch, mid_ch)
        self.down4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv5 = ConvNormActivation(mid_ch, mid_ch)

        self.conv6 = ConvNormActivation(mid_ch, mid_ch, dilation=2)

        self.conv5d = ConvNormActivation(mid_ch * 2, mid_ch)
        self.conv4d = ConvNormActivation(mid_ch * 2, mid_ch)
        self.conv3d = ConvNormActivation(mid_ch * 2, mid_ch)
        self.conv2d = ConvNormActivation(mid_ch * 2, mid_ch)
        self.conv1d = ConvNormActivation(mid_ch * 2, out_ch)

    def forward(self, x):
        hx = x

        hxin = self.conv_in(hx)

        hx1 = self.conv1(hxin)
        hx = self.down1(hx1)

        hx2 = self.conv2(hx)
        hx = self.down2(hx2)

        hx3 = self.conv3(hx)
        hx = self.down3(hx3)

        hx4 = self.conv4(hx)
        hx = self.down4(hx4)

        hx5 = self.conv5(hx)

        hx6 = self.conv6(hx5)

        hx5d = self.conv5d(torch.cat((hx6, hx5), 1))
        hx5dup = F.interpolate(hx5d, scale_factor=2)

        hx4d = self.conv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, scale_factor=2)

        hx3d = self.conv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, scale_factor=2)

        hx2d = self.conv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, scale_factor=2)

        hx1d = self.conv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


# RSU-5
class RSU5(nn.Module):  # UNet05DRES(nn.Module):
    def __init__(self, in_channels=3, mid_ch=12, out_ch=3):
        super().__init__()

        self.conv_in = ConvNormActivation(in_channels, out_ch, dilation=1)

        self.conv1 = ConvNormActivation(out_ch, mid_ch, dilation=1)
        self.down1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv2 = ConvNormActivation(mid_ch, mid_ch, dilation=1)
        self.down2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3 = ConvNormActivation(mid_ch, mid_ch, dilation=1)
        self.down3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv4 = ConvNormActivation(mid_ch, mid_ch, dilation=1)

        self.conv5 = ConvNormActivation(mid_ch, mid_ch, dilation=2)

        self.conv4d = ConvNormActivation(mid_ch * 2, mid_ch, dilation=1)
        self.conv3d = ConvNormActivation(mid_ch * 2, mid_ch, dilation=1)
        self.conv2d = ConvNormActivation(mid_ch * 2, mid_ch, dilation=1)
        self.conv1d = ConvNormActivation(mid_ch * 2, out_ch, dilation=1)

    def forward(self, x):
        hx = x

        hxin = self.conv_in(hx)

        hx1 = self.conv1(hxin)
        hx = self.down1(hx1)

        hx2 = self.conv2(hx)
        hx = self.down2(hx2)

        hx3 = self.conv3(hx)
        hx = self.down3(hx3)

        hx4 = self.conv4(hx)

        hx5 = self.conv5(hx4)

        hx4d = self.conv4d(torch.cat((hx5, hx4), 1))
        hx4dup = F.interpolate(hx4d, scale_factor=2)

        hx3d = self.conv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, scale_factor=2)

        hx2d = self.conv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, scale_factor=2)

        hx1d = self.conv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


# RSU-4
class RSU4(nn.Module):
    def __init__(self, in_channels=3, mid_ch=12, out_ch=3):
        super().__init__()

        self.conv_in = ConvNormActivation(in_channels, out_ch, dilation=1)

        self.conv1 = ConvNormActivation(out_ch, mid_ch, dilation=1)
        self.down1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv2 = ConvNormActivation(mid_ch, mid_ch, dilation=1)
        self.down2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3 = ConvNormActivation(mid_ch, mid_ch, dilation=1)

        self.conv4 = ConvNormActivation(mid_ch, mid_ch, dilation=2)

        self.conv3d = ConvNormActivation(mid_ch * 2, mid_ch, dilation=1)
        self.conv2d = ConvNormActivation(mid_ch * 2, mid_ch, dilation=1)
        self.conv1d = ConvNormActivation(mid_ch * 2, out_ch, dilation=1)

    def forward(self, x):
        hx = x

        hxin = self.conv_in(hx)

        hx1 = self.conv1(hxin)
        hx = self.down1(hx1)

        hx2 = self.conv2(hx)
        hx = self.down2(hx2)

        hx3 = self.conv3(hx)

        hx4 = self.conv4(hx3)

        hx3d = self.conv3d(torch.cat((hx4, hx3), 1))
        hx3dup = F.interpolate(hx3d, scale_factor=2)

        hx2d = self.conv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, scale_factor=2)

        hx1d = self.conv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


# RSU-4F
class RSU4F(nn.Module):
    def __init__(self, in_channels=3, mid_ch=12, out_ch=3):
        super().__init__()

        self.conv_in = ConvNormActivation(in_channels, out_ch, dilation=1)

        self.conv1 = ConvNormActivation(out_ch, mid_ch, dilation=1)
        self.conv2 = ConvNormActivation(mid_ch, mid_ch, dilation=2)
        self.conv3 = ConvNormActivation(mid_ch, mid_ch, dilation=4)

        self.conv4 = ConvNormActivation(mid_ch, mid_ch, dilation=8)

        self.conv3d = ConvNormActivation(mid_ch * 2, mid_ch, dilation=4)
        self.conv2d = ConvNormActivation(mid_ch * 2, mid_ch, dilation=2)
        self.conv1d = ConvNormActivation(mid_ch * 2, out_ch, dilation=1)

    def forward(self, x):
        hx = x

        hxin = self.conv_in(hx)

        hx1 = self.conv1(hxin)
        hx2 = self.conv2(hx1)
        hx3 = self.conv3(hx2)

        hx4 = self.conv4(hx3)

        hx3d = self.conv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.conv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.conv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


# U^2-Net
class U2NET(nn.Module):
    def __init__(self, in_channels=3, out_ch=1):
        super().__init__()

        self.stage1 = RSU7(in_channels, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = F.interpolate(hx6, scale_factor=2)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = F.interpolate(hx5d, scale_factor=2)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, scale_factor=2)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, scale_factor=2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, scale_factor=2)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = F.interpolate(d2, scale_factor=2)

        d3 = self.side3(hx3d)
        d3 = F.interpolate(d3, scale_factor=4)

        d4 = self.side4(hx4d)
        d4 = F.interpolate(d4, scale_factor=8)

        d5 = self.side5(hx5d)
        d5 = F.interpolate(d5, scale_factor=16)

        d6 = self.side6(hx6)
        d6 = F.interpolate(d6, scale_factor=32)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(
            d4), torch.sigmoid(d5), torch.sigmoid(d6)


# U^2-Net small
class U2NETP(nn.Module):
    def __init__(self, in_channels=3, out_ch=1):
        super().__init__()

        self.stage1 = RSU7(in_channels, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(64, 16, 64)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(64, 16, 64)

        # decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = F.interpolate(hx6, scale_factor=2)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = F.interpolate(hx5d, scale_factor=2)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, scale_factor=2)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, scale_factor=2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, scale_factor=2)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = F.interpolate(d2, scale_factor=2)

        d3 = self.side3(hx3d)
        d3 = F.interpolate(d3, scale_factor=4)

        d4 = self.side4(hx4d)
        d4 = F.interpolate(d4, scale_factor=8)

        d5 = self.side5(hx5d)
        d5 = F.interpolate(d5, scale_factor=16)

        d6 = self.side6(hx6)
        d6 = F.interpolate(d6, scale_factor=32)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)


if __name__ == '__main__':
    model = U2NETP(3, 1)
    x = torch.randn(1, 3, 640, 640)
    print(model(x))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
