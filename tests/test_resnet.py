"""CIFAR-10 ResNet 网络结构测试。"""

import importlib.util

import pytest
import torch

import models.resnet as resnet
from models import create_model


def test_resnet_module_exists() -> None:
    """模型包应当提供独立的ResNet模块。"""

    assert importlib.util.find_spec("models.resnet") is not None


def test_option_a_shortcut_class_exists() -> None:
    """ResNet模块应当提供论文Option A shortcut。"""

    assert hasattr(resnet, "OptionAShortcut")


def test_option_a_identity_returns_unchanged_values() -> None:
    """通道和尺寸不变时，Option A应当执行恒等映射。"""

    shortcut = resnet.OptionAShortcut(16, 16, 1)
    x = torch.randn(2, 16, 32, 32)

    y = shortcut(x)

    assert torch.equal(y, x)


def test_option_a_downsamples_and_pads_channels_symmetrically() -> None:
    """通道翻倍时，Option A应隔点采样并在两侧补零。"""

    shortcut = resnet.OptionAShortcut(16, 32, 2)
    x = torch.arange(
        1 * 16 * 4 * 4,
        dtype=torch.float32,
    ).reshape(1, 16, 4, 4)

    y = shortcut(x)

    assert y.shape == (1, 32, 2, 2)
    assert torch.count_nonzero(y[:, :8]) == 0
    assert torch.equal(y[:, 8:24], x[:, :, ::2, ::2])
    assert torch.count_nonzero(y[:, 24:]) == 0


def test_option_a_has_no_trainable_parameters() -> None:
    """Option A只进行取样和补零，不应引入模型参数。"""

    shortcut = resnet.OptionAShortcut(16, 32, 2)

    assert sum(parameter.numel() for parameter in shortcut.parameters()) == 0


@pytest.mark.parametrize(
    ("arguments", "exception_type"),
    [
        ((0, 16, 1), ValueError),
        ((16, 0, 1), ValueError),
        ((16, 16, 0), ValueError),
        ((16.0, 16, 1), TypeError),
        ((16, 16, 2), ValueError),
        ((16, 32, 1), ValueError),
        ((16, 48, 2), ValueError),
        ((16, 32, 3), ValueError),
    ],
)
def test_option_a_rejects_invalid_arguments(
    arguments: tuple[object, object, object],
    exception_type: type[Exception],
) -> None:
    """Option A应当拒绝非法类型、数值和通道步幅组合。"""

    with pytest.raises(exception_type):
        resnet.OptionAShortcut(*arguments)


def test_residual_block_class_exists() -> None:
    """ResNet模块应当提供ResidualBlock类。"""

    assert hasattr(resnet, "ResidualBlock")


def test_residual_block_preserves_shape() -> None:
    """普通残差块应保持通道数和空间尺寸。"""

    block = resnet.ResidualBlock(16, 16, 1)
    x = torch.randn(2, 16, 32, 32)

    assert block(x).shape == (2, 16, 32, 32)


def test_residual_block_downsamples_once() -> None:
    """阶段转换残差块应使通道翻倍、空间尺寸减半。"""

    block = resnet.ResidualBlock(16, 32, 2)
    x = torch.randn(2, 16, 32, 32)

    assert block(x).shape == (2, 32, 16, 16)


def test_residual_block_convolution_configuration() -> None:
    """残差分支应包含论文规定的两层3×3无偏置卷积。"""

    block = resnet.ResidualBlock(16, 32, 2)

    assert block.conv1.kernel_size == (3, 3)
    assert block.conv1.stride == (2, 2)
    assert block.conv1.padding == (1, 1)
    assert block.conv1.bias is None
    assert block.conv2.kernel_size == (3, 3)
    assert block.conv2.stride == (1, 1)
    assert block.conv2.padding == (1, 1)
    assert block.conv2.bias is None


def test_residual_block_adds_shortcut_before_final_relu() -> None:
    """残差分支为0时，输出应等于ReLU(shortcut(x))。"""

    block = resnet.ResidualBlock(16, 32, 2).eval()
    torch.nn.init.zeros_(block.conv1.weight)
    torch.nn.init.zeros_(block.conv2.weight)
    x = torch.randn(2, 16, 8, 8)

    with torch.no_grad():
        actual = block(x)
        expected = torch.relu(block.shortcut(x))

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("arguments", "exception_type"),
    [
        ((0, 16, 1), ValueError),
        ((16, 0, 1), ValueError),
        ((16.0, 16, 1), TypeError),
        ((16, 16, 2), ValueError),
    ],
)
def test_residual_block_rejects_invalid_arguments(
    arguments: tuple[object, object, object],
    exception_type: type[Exception],
) -> None:
    """ResidualBlock应沿用Option A的合法通道与步幅规则。"""

    with pytest.raises(exception_type):
        resnet.ResidualBlock(*arguments)


def test_resnet_class_exists() -> None:
    """ResNet模块应当提供完整ResNet类。"""

    assert hasattr(resnet, "ResNet")


@pytest.mark.parametrize(
    ("n", "num_classes", "exception_type"),
    [
        ((0), 10, ValueError),
        ((-1), 10, ValueError),
        ((3.5), 10, TypeError),
        ((3), 0, ValueError),
        ((3), 10.0, TypeError),
    ],
)
def test_resnet_rejects_invalid_arguments(
    n: object,
    num_classes: object,
    exception_type: type[Exception],
) -> None:
    """ResNet应拒绝非法的n和类别数。"""

    with pytest.raises(exception_type):
        resnet.ResNet(n=n, num_classes=num_classes)


@pytest.mark.parametrize("n", [3, 5, 7, 9])
def test_resnet_has_n_blocks_per_stage(n: int) -> None:
    """ResNet的三个阶段都应包含n个ResidualBlock。"""

    model = resnet.ResNet(n=n)

    assert len(model.stage1) == n
    assert len(model.stage2) == n
    assert len(model.stage3) == n
    assert all(
        isinstance(block, resnet.ResidualBlock)
        for stage in (model.stage1, model.stage2, model.stage3)
        for block in stage
    )


def test_resnet_stage_shapes() -> None:
    """三个残差阶段应产生论文规定的特征图尺寸。"""

    model = resnet.ResNet(n=3).eval()
    x = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        x = model.stem(x)
        assert x.shape == (2, 16, 32, 32)
        x = model.stage1(x)
        assert x.shape == (2, 16, 32, 32)
        x = model.stage2(x)
        assert x.shape == (2, 32, 16, 16)
        x = model.stage3(x)
        assert x.shape == (2, 64, 8, 8)
        x = model.avgpool(x)
        assert x.shape == (2, 64, 1, 1)


def test_resnet20_output_shape() -> None:
    """ResNet-20应为每张CIFAR图片输出10个分类分数。"""

    model = resnet.ResNet(n=3).eval()
    x = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        y = model(x)

    assert model.depth == 20
    assert y.shape == (2, 10)


def test_resnet_runs_weight_initialization() -> None:
    """创建ResNet时应执行权重初始化。"""

    class TrackingResNet(resnet.ResNet):
        def _initialize_weights(self) -> None:
            self.initialization_was_called = True

    model = TrackingResNet(n=3)

    assert model.initialization_was_called is True


def test_resnet_batchnorm_initialization() -> None:
    """全部BatchNorm应初始化为weight=1、bias=0。"""

    model = resnet.ResNet(n=3)
    batchnorms = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.BatchNorm2d)
    ]

    assert batchnorms
    for module in batchnorms:
        assert torch.all(module.weight == 1)
        assert torch.all(module.bias == 0)


@pytest.mark.parametrize(
    ("name", "expected_depth"),
    [
        ("resnet20", 20),
        ("resnet32", 32),
        ("resnet44", 44),
        ("resnet56", 56),
    ],
)
def test_create_resnet_models(
    name: str,
    expected_depth: int,
) -> None:
    """统一入口应创建对应深度的ResNet。"""

    model = create_model(name)

    assert isinstance(model, resnet.ResNet)
    assert model.depth == expected_depth


@pytest.mark.parametrize(
    "name",
    ["resnet20", "resnet32", "resnet44", "resnet56"],
)
def test_resnet_output_shapes(name: str) -> None:
    """四种ResNet都应输出每张图片的10个分类分数。"""

    model = create_model(name).eval()
    x = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 10)


def test_resnet_custom_num_classes() -> None:
    """num_classes应控制ResNet最终输出维度。"""

    model = create_model("resnet20", num_classes=100).eval()
    x = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        y = model(x)

    assert model.fc.out_features == 100
    assert y.shape == (2, 100)


@pytest.mark.parametrize(
    ("name", "expected_depth"),
    [
        ("resnet20", 20),
        ("resnet32", 32),
        ("resnet44", 44),
        ("resnet56", 56),
    ],
)
def test_resnet_weighted_layer_depth(
    name: str,
    expected_depth: int,
) -> None:
    """卷积层与全连接层总数应等于ResNet标称深度。"""

    model = create_model(name)
    actual_depth = sum(
        isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
        for module in model.modules()
    )

    assert actual_depth == expected_depth


@pytest.mark.parametrize("depth", [20, 32, 44, 56])
def test_plainnet_and_resnet_have_equal_parameter_counts(
    depth: int,
) -> None:
    """同深度两种网络应只差shortcut连接，不差参数规模。"""

    plain = create_model(f"plain{depth}")
    residual = create_model(f"resnet{depth}")

    plain_count = sum(parameter.numel() for parameter in plain.parameters())
    residual_count = sum(
        parameter.numel() for parameter in residual.parameters()
    )

    assert residual_count == plain_count
