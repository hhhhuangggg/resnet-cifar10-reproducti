"""ImageNet ResNet-18/34/50/101/152 的结构测试。"""
import torch
from torch import nn

from models.imagenet_resnet import BasicBlock, Bottleneck, conv1x1


def test_basic_block_identity_preserves_shape() -> None:
    """删除恒等shortcut或错误改变通道时，本测试应当失败。"""

    block = BasicBlock(64, 64)
    block.eval()
    x = torch.randn(2, 64, 16, 16)

    with torch.no_grad():
        output = block(x)

    assert BasicBlock.expansion == 1
    assert output.shape == (2, 64, 16, 16)
    assert block.downsample is None


def test_basic_block_projection_changes_shape() -> None:
    """阶段切换未同时改变空间和通道尺寸时，本测试应当失败。"""

    downsample = nn.Sequential(
        conv1x1(64, 128, stride=2),
        nn.BatchNorm2d(128),
    )
    block = BasicBlock(64, 128, stride=2, downsample=downsample)
    block.eval()
    x = torch.randn(2, 64, 16, 16)

    with torch.no_grad():
        output = block(x)

    assert output.shape == (2, 128, 8, 8)
    assert block.conv1.kernel_size == (3, 3)
    assert block.conv1.stride == (2, 2)
    assert block.conv2.stride == (1, 1)


def test_bottleneck_uses_paper_stride_placement() -> None:
    """stride离开论文规定的第一个1x1卷积时，本测试应当失败。"""

    downsample = nn.Sequential(
        conv1x1(256, 512, stride=2),
        nn.BatchNorm2d(512),
    )
    block = Bottleneck(256, 128, stride=2, downsample=downsample)
    block.eval()
    x = torch.randn(2, 256, 16, 16)

    with torch.no_grad():
        output = block(x)

    assert Bottleneck.expansion == 4
    assert block.conv1.stride == (2, 2)
    assert block.conv2.stride == (1, 1)
    assert block.conv3.out_channels == 512
    assert output.shape == (2, 512, 8, 8)
