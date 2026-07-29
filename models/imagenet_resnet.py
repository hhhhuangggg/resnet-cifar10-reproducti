"""论文版 ImageNet ResNet-18/34/50/101/152 网络结构。"""
from collections.abc import Sequence

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


class Bottleneck(nn.Module):
    """ResNet-50/101/152使用的三层瓶颈残差块。"""

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        expanded_channels = channels * self.expansion
        self.conv1 = conv1x1(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = conv1x1(channels, expanded_channels)
        self.bn3 = nn.BatchNorm2d(expanded_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算论文版Bottleneck的残差映射。"""

        identity = x

        output = self.relu(self.bn1(self.conv1(x)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))

        if self.downsample is not None:
            identity = self.downsample(x)

        output = output + identity
        return self.relu(output)


BlockType = type[BasicBlock] | type[Bottleneck]


class ImageNetResNet(nn.Module):
    """由四个残差阶段组成的论文版ImageNet ResNet。"""

    def __init__(
        self,
        block: BlockType,
        layers: Sequence[int],
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, int)
        ):
            raise TypeError("num_classes必须是整数")
        if num_classes <= 0:
            raise ValueError("num_classes必须大于0")
        if len(layers) != 4 or any(
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for count in layers
        ):
            raise ValueError("layers必须包含四个正整数")

        self.block = block
        self.layers_config = tuple(layers)
        self.num_classes = num_classes
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.layer1 = self._make_layer(
            block,
            channels=64,
            blocks=layers[0],
            stride=1,
        )
        self.layer2 = self._make_layer(
            block,
            channels=128,
            blocks=layers[1],
            stride=2,
        )
        self.layer3 = self._make_layer(
            block,
            channels=256,
            blocks=layers[2],
            stride=2,
        )
        self.layer4 = self._make_layer(
            block,
            channels=512,
            blocks=layers[3],
            stride=2,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            512 * block.expansion,
            num_classes,
        )
        self._initialize_weights()

    def _make_layer(
        self,
        block: BlockType,
        channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """创建一个残差阶段并在第一个block按需projection。"""

        output_channels = channels * block.expansion
        downsample = None
        if stride != 1 or self.in_channels != output_channels:
            downsample = nn.Sequential(
                conv1x1(
                    self.in_channels,
                    output_channels,
                    stride,
                ),
                nn.BatchNorm2d(output_channels),
            )

        modules: list[nn.Module] = [
            block(
                self.in_channels,
                channels,
                stride,
                downsample,
            )
        ]
        self.in_channels = output_channels
        modules.extend(
            block(self.in_channels, channels)
            for _ in range(1, blocks)
        )
        return nn.Sequential(*modules)

    def _initialize_weights(self) -> None:
        """初始化卷积和BatchNorm参数。"""

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
        """输出每张ImageNet图片的分类logits。"""

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet18(num_classes: int = 1000) -> ImageNetResNet:
    """创建ImageNet ResNet-18。"""

    return ImageNetResNet(
        BasicBlock,
        (2, 2, 2, 2),
        num_classes,
    )


def resnet34(num_classes: int = 1000) -> ImageNetResNet:
    """创建ImageNet ResNet-34。"""

    return ImageNetResNet(
        BasicBlock,
        (3, 4, 6, 3),
        num_classes,
    )


def resnet50(num_classes: int = 1000) -> ImageNetResNet:
    """创建ImageNet ResNet-50。"""

    return ImageNetResNet(
        Bottleneck,
        (3, 4, 6, 3),
        num_classes,
    )


def resnet101(num_classes: int = 1000) -> ImageNetResNet:
    """创建ImageNet ResNet-101。"""

    return ImageNetResNet(
        Bottleneck,
        (3, 4, 23, 3),
        num_classes,
    )


def resnet152(num_classes: int = 1000) -> ImageNetResNet:
    """创建ImageNet ResNet-152。"""

    return ImageNetResNet(
        Bottleneck,
        (3, 8, 36, 3),
        num_classes,
    )
