"""CIFAR-10 PlainNet 模型定义。

本模块后续负责：
1. 实现不带残差连接的 PlainBlock。
2. 实现 CIFAR-10 版本的 PlainNet。
3. 支持 PlainNet-20、32、44、56。
4. 只定义模型结构，不处理数据和训练。
"""
import torch
from torch import nn


class PlainBlock(nn.Module):
    """不包含残差连接的两层卷积模块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
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
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
    
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x

class PlainNet(nn.Module):
    """用于CIFAR图像分类的PlainNet。"""

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
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(
            out_channels=16,
            num_blocks=n,
            first_stride=1,
        )
        self.stage2 = self._make_stage(
            out_channels=32,
            num_blocks=n,
            first_stride=2,
        )
        self.stage3 = self._make_stage(
            out_channels=64,
            num_blocks=n,
            first_stride=2,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            in_features=64,
            out_features=num_classes,
        )
        self._initialize_weights()
    def _make_stage(
        self,
        out_channels: int,
        num_blocks: int,
        first_stride: int,
    ) -> nn.Sequential:
        """创建由多个PlainBlock组成的一个网络阶段。"""

        blocks = [
            PlainBlock(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=first_stride,
            )
        ]

        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            blocks.append(
                PlainBlock(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=1,
                )
            )

        return nn.Sequential(*blocks)
    def _initialize_weights(self) -> None:
        """按照适合ReLU网络的规则初始化模型参数。"""

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
        """执行PlainNet的完整前向传播。"""

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x