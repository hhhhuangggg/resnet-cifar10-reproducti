"""PlainNet 和 ResNet 的结构测试。

后续需要验证：
1. 八个模型都能成功创建。
2. 输入 [2, 3, 32, 32] 时输出为 [2, 10]。
3. 网络深度分别为 20、32、44、56。
4. 三个阶段的空间尺寸正确。
5. Option A shortcut 不含可训练参数。
6. 同深度 PlainNet 和 ResNet 可以公平比较。
"""
import torch
import pytest
import models.plainnet as plainnet
from models import create_model
def test_plain_block_class_exists() -> None:
    """PlainNet模块应当提供PlainBlock类。"""

    assert hasattr(plainnet, "PlainBlock")

def test_plain_block_first_path_downsamples() -> None:
    """第一组Conv-BN-ReLU应当改变通道并让尺寸减半。"""

    block = plainnet.PlainBlock(
        in_channels=16,
        out_channels=32,
        stride=2,
    )
    x = torch.randn(2, 16, 32, 32)

    y = block.conv1(x)
    y = block.bn1(y)
    y = block.relu1(y)

    assert y.shape == (2, 32, 16, 16)

def test_plain_block_downsamples_once() -> None:
    """完整PlainBlock应当只执行一次空间降采样。"""

    block = plainnet.PlainBlock(
        in_channels=16,
        out_channels=32,
        stride=2,
    )
    x = torch.randn(2, 16, 32, 32)

    y = block(x)

    assert y.shape == (2, 32, 16, 16)

    
def test_plain_block_preserves_shape() -> None:
    """stride为1时，PlainBlock应当保持张量形状。"""

    block = plainnet.PlainBlock(
        in_channels=16,
        out_channels=16,
        stride=1,
    )
    x = torch.randn(2, 16, 32, 32)

    y = block(x)

    assert y.shape == (2, 16, 32, 32)
@pytest.mark.parametrize(
    ("arguments", "exception_type"),
    [
        ((0, 16, 1), ValueError),
        ((16, 0, 1), ValueError),
        ((16, 16, 0), ValueError),
        ((16.0, 16, 1), TypeError),
    ],
)
def test_plain_block_rejects_invalid_arguments(
    arguments: tuple[object, object, object],
    exception_type: type[Exception],
) -> None:
    """PlainBlock应当拒绝类型或数值不合法的参数。"""

    with pytest.raises(exception_type):
        plainnet.PlainBlock(*arguments)
def test_plainnet_stem_shape() -> None:
    """PlainNet的stem应当把RGB输入转换成16通道特征。"""

    model = plainnet.PlainNet(n=3)
    x = torch.randn(2, 3, 32, 32)

    y = model.stem(x)

    assert y.shape == (2, 16, 32, 32)

@pytest.mark.parametrize(
    ("n", "num_classes", "exception_type"),
    [
        (0, 10, ValueError),
        (-1, 10, ValueError),
        (3.5, 10, TypeError),
        (3, 0, ValueError),
        (3, 10.0, TypeError),
    ],
)
def test_plainnet_rejects_invalid_arguments(
    n: object,
    num_classes: object,
    exception_type: type[Exception],
) -> None:
    """PlainNet应当拒绝非法的n和类别数。"""

    with pytest.raises(exception_type):
        plainnet.PlainNet(
            n=n,
            num_classes=num_classes,
        )
@pytest.mark.parametrize("n", [3, 5, 7, 9])
def test_plainnet_has_n_blocks_per_stage(n: int) -> None:
    """PlainNet的三个阶段都应当包含n个PlainBlock。"""

    model = plainnet.PlainNet(n=n)

    assert len(model.stage1) == n
    assert len(model.stage2) == n
    assert len(model.stage3) == n
def test_plainnet_stage_shapes() -> None:
    """PlainNet的三个阶段应当产生论文规定的特征图形状。"""

    model = plainnet.PlainNet(n=3)
    x = torch.randn(2, 3, 32, 32)

    x = model.stem(x)
    assert x.shape == (2, 16, 32, 32)

    x = model.stage1(x)
    assert x.shape == (2, 16, 32, 32)

    x = model.stage2(x)
    assert x.shape == (2, 32, 16, 16)

    x = model.stage3(x)
    assert x.shape == (2, 64, 8, 8)
def test_plainnet20_output_shape() -> None:
    """PlainNet-20应当为每张CIFAR图片输出10个分类分数。"""

    model = plainnet.PlainNet(
        n=3,
        num_classes=10,
    )
    x = torch.randn(2, 3, 32, 32)

    y = model(x)

    assert y.shape == (2, 10)
def test_plainnet_runs_weight_initialization() -> None:
    """创建PlainNet时应当执行权重初始化步骤。"""

    class TrackingPlainNet(plainnet.PlainNet):
        def _initialize_weights(self) -> None:
            self.initialization_was_called = True

    model = TrackingPlainNet(n=3)

    assert model.initialization_was_called is True
def test_plainnet_batchnorm_initialization() -> None:
    """所有BatchNorm层都应当初始化为weight=1、bias=0。"""

    model = plainnet.PlainNet(n=3)

    batchnorm_count = 0

    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            batchnorm_count += 1

            assert torch.all(module.weight == 1)
            assert torch.all(module.bias == 0)

    assert batchnorm_count > 0
@pytest.mark.parametrize(
    ("name", "expected_depth"),
    [
        ("plain20", 20),
        ("plain32", 32),
        ("plain44", 44),
        ("plain56", 56),
    ],
)
def test_create_plainnet_models(
    name: str,
    expected_depth: int,
) -> None:
    """统一入口应当创建对应深度的PlainNet。"""

    model = create_model(name)

    assert isinstance(model, plainnet.PlainNet)
    assert model.depth == expected_depth
@pytest.mark.parametrize(
    "name",
    [
        "plain20",
        "plain32",
        "plain44",
        "plain56",
    ],
)
def test_plainnet_output_shapes(name: str) -> None:
    """四种PlainNet都应当输出每张图片的10个分类分数。"""

    model = create_model(name)
    model.eval()

    x = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 10)
def test_plainnet_custom_num_classes() -> None:
    """num_classes应当控制PlainNet的最终输出维度。"""

    model = create_model(
        "plain20",
        num_classes=100,
    )
    model.eval()

    x = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        y = model(x)

    assert model.fc.out_features == 100
    assert y.shape == (2, 100)
@pytest.mark.parametrize(
    ("name", "expected_depth"),
    [
        ("plain20", 20),
        ("plain32", 32),
        ("plain44", 44),
        ("plain56", 56),
    ],
)
def test_plainnet_weighted_layer_depth(
    name: str,
    expected_depth: int,
) -> None:
    """卷积层和全连接层的总数应当等于模型标称深度。"""

    model = create_model(name)

    actual_depth = sum(
        isinstance(
            module,
            (torch.nn.Conv2d, torch.nn.Linear),
        )
        for module in model.modules()
    )

    assert actual_depth == expected_depth