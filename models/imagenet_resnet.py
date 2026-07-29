"""论文版 ImageNet ResNet-18/34/50/101/152 网络结构。"""
import torch
from torch import nn


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Conv2d:
    """创建不带偏置的3x3卷积。"""

    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Conv2d:
    """创建projection shortcut使用的不带偏置1x1卷积。"""

    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    """ResNet-18/34使用的两层残差块。"""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算ReLU(F(x) + shortcut(x))。"""

        identity = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)

        if self.downsample is not None:
            identity = self.downsample(x)

        output = output + identity
        return self.relu(output)
