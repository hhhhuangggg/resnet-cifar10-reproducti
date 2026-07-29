"""ImageNet ResNet-18/34/50/101/152 的结构测试。"""
from collections.abc import Callable

import pytest
import torch
import torchvision.models as tv_models
from torch import nn

from models.imagenet_resnet import (
    BasicBlock,
    Bottleneck,
    ImageNetResNet,
    conv1x1,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)


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


@pytest.mark.parametrize(
    ("factory", "block_type", "lengths"),
    [
        (resnet18, BasicBlock, (2, 2, 2, 2)),
        (resnet34, BasicBlock, (3, 4, 6, 3)),
        (resnet50, Bottleneck, (3, 4, 6, 3)),
        (resnet101, Bottleneck, (3, 4, 23, 3)),
        (resnet152, Bottleneck, (3, 8, 36, 3)),
    ],
)
def test_factories_build_expected_stages(
    factory: Callable[..., ImageNetResNet],
    block_type: type[nn.Module],
    lengths: tuple[int, int, int, int],
) -> None:
    """工厂函数使用错误block或阶段数量时，本测试应当失败。"""

    model = factory()

    assert isinstance(model.layer1[0], block_type)
    assert tuple(
        len(stage)
        for stage in (
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
    ) == lengths
    assert model.conv1.kernel_size == (7, 7)
    assert model.conv1.stride == (2, 2)
    assert model.maxpool.kernel_size == 3
    assert model.fc.out_features == 1000


@pytest.mark.parametrize("num_classes", [0, -1, 10.5, True])
def test_factory_rejects_invalid_num_classes(num_classes: object) -> None:
    """非法分类数被接受时，本测试应当失败。"""

    with pytest.raises((TypeError, ValueError)):
        resnet18(num_classes=num_classes)


@pytest.mark.parametrize(
    ("ours", "reference", "expected_parameters"),
    [
        (resnet18, tv_models.resnet18, 11_689_512),
        (resnet34, tv_models.resnet34, 21_797_672),
        (resnet50, tv_models.resnet50, 25_557_032),
        (resnet101, tv_models.resnet101, 44_549_160),
        (resnet152, tv_models.resnet152, 60_192_808),
    ],
)
def test_parameter_counts_match_torchvision(
    ours: Callable[..., ImageNetResNet],
    reference: Callable[..., nn.Module],
    expected_parameters: int,
) -> None:
    """遗漏卷积、projection或分类层时，本测试应当失败。"""

    our_model = ours()
    reference_model = reference(weights=None)

    our_parameters = sum(
        parameter.numel()
        for parameter in our_model.parameters()
    )
    reference_parameters = sum(
        parameter.numel()
        for parameter in reference_model.parameters()
    )

    assert our_parameters == expected_parameters
    assert our_parameters == reference_parameters


@pytest.mark.parametrize(
    "factory",
    [resnet18, resnet34, resnet50, resnet101, resnet152],
)
def test_imagenet_models_meta_forward(
    factory: Callable[..., ImageNetResNet],
) -> None:
    """尺寸传播或分类输出错误时，本测试应当失败。"""

    model = factory(num_classes=17).to(device="meta")
    x = torch.empty(1, 3, 224, 224, device="meta")

    output = model(x)

    assert output.shape == (1, 17)
    assert output.device.type == "meta"
