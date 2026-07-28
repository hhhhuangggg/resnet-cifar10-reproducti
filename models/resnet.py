"""CIFAR-10 ResNet 模型定义。

本模块后续负责：
1. 实现带残差连接的 ResidualBlock。
2. 实现论文中的 Option A shortcut。
3. 实现 CIFAR-10 版本的 ResNet。
4. 支持 ResNet-20、32、44、56。
5. 只定义模型结构，不处理数据和训练。
"""
import torch
from torch.nn import functional as F
from torch import nn


class OptionAShortcut(nn.Module):
    """论文CIFAR实验使用的无参数Option A shortcut。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> None:
        super().__init__()

        for name, value in (
            ("in_channels", in_channels),
            ("out_channels", out_channels),
            ("stride", stride),
        ):
            if not isinstance(value, int):
                raise TypeError(f"{name}必须是整数")
            if value <= 0:
                raise ValueError(f"{name}必须大于0")

        is_identity = in_channels == out_channels and stride == 1
        is_stage_transition = (
            out_channels == 2 * in_channels and stride == 2
        )
        if not (is_identity or is_stage_transition):
            raise ValueError(
                "Option A只支持通道不变且stride=1，"
                "或通道翻倍且stride=2"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行恒等映射，或隔点采样后对称补零。"""

        if self.stride == 1:
            return x

        x = x[:, :, ::2, ::2]
        channel_padding = (self.out_channels - self.in_channels) // 2
        return F.pad(
            x,
            (0, 0, 0, 0, channel_padding, channel_padding),
        )


class ResidualBlock(nn.Module):
    """包含两层卷积和Option A捷径的基础残差块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.shortcut = OptionAShortcut(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
        )
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算ReLU(F(x) + shortcut(x))。"""

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu2(out)
        return out


class ResNet(nn.Module):
    """用于CIFAR图像分类的论文版ResNet。"""

    def __init__(
        self,
        n: int,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if not isinstance(n, int):
            raise TypeError("n必须是整数")
        if n <= 0:
            raise ValueError("n必须大于0")
        if not isinstance(num_classes, int):
            raise TypeError("num_classes必须是整数")
        if num_classes <= 0:
            raise ValueError("num_classes必须大于0")

        self.n = n
        self.depth = 6 * n + 2
        self.num_classes = num_classes
        self.in_channels = 16

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(16, n, 1)
        self.stage2 = self._make_stage(32, n, 2)
        self.stage3 = self._make_stage(64, n, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self._initialize_weights()

    def _make_stage(
        self,
        out_channels: int,
        num_blocks: int,
        first_stride: int,
    ) -> nn.Sequential:
        """创建一个由ResidualBlock组成的残差阶段。"""

        blocks = [
            ResidualBlock(
                self.in_channels,
                out_channels,
                first_stride,
            )
        ]
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            blocks.append(
                ResidualBlock(
                    self.in_channels,
                    out_channels,
                    1,
                )
            )
        return nn.Sequential(*blocks)

    def _initialize_weights(self) -> None:
        """使用与PlainNet相同的卷积和BatchNorm初始化。"""

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行完整ResNet前向传播并输出分类logits。"""

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
