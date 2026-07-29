# ImageNet ResNet Architectures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 手写论文版 ImageNet ResNet-18/34/50/101/152，并在不使用训练GPU的条件下验证结构、输出形状和参数量。

**Architecture:** 新增独立的 `models/imagenet_resnet.py`，不改造现有 CIFAR-10 `models/resnet.py`。模型由 `BasicBlock`、`Bottleneck`、`ImageNetResNet` 和五个工厂函数组成；测试使用 CPU 构造和 meta device 前向传播，并用 Torchvision 仅核对参数量和配置。

**Tech Stack:** Python 3.11、PyTorch 2.5.1、Torchvision 0.20.1、pytest 8.3.3、Git Worktree。

## Global Constraints

- 所有改动只发生在 `E:\文档\resnet-imagenet结构` 的 `imagenet-resnet-architectures` 分支。
- 不修改 CIFAR-10 正式训练参数、数据代码、训练循环或已有模型语义。
- 不下载 ImageNet、不下载预训练权重、不训练 ImageNet。
- 实施期间不使用 CUDA；完整形状验证使用 PyTorch meta device。
- 默认 ImageNet 输入语义为 `[N, 3, 224, 224]`，默认类别数为 1000。
- 原始论文优先：Bottleneck 阶段切换的 stride 置于第一个 `1x1` 卷积；学习文档明确说明 Torchvision v1.5 将 stride 置于 `3x3` 卷积。
- 每项功能遵循红-绿 TDD，并在任务完成后单独提交。

---

## File Map

- Create: `models/imagenet_resnet.py`
  - 只负责 ImageNet ResNet block、主干和五个工厂函数。
- Create: `tests/test_imagenet_resnet.py`
  - 只负责 ImageNet结构、配置、meta前向和Torchvision参数量对照。
- Create: `docs/阶段九-ImageNet-ResNet结构学习总结.md`
  - 解释两种block、projection shortcut、五种深度和CIFAR/ImageNet差异。
- Preserve: `models/resnet.py`
  - 继续只负责 CIFAR-10 ResNet，不修改。
- Preserve: `models/__init__.py`
  - 继续保留 CIFAR-10 `create_model` 入口，不把两套同名模型混入现有训练CLI。

---

### Task 1: BasicBlock 与卷积辅助函数

**Files:**
- Create: `models/imagenet_resnet.py`
- Create: `tests/test_imagenet_resnet.py`

**Interfaces:**
- Produces: `conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d`
- Produces: `conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d`
- Produces: `BasicBlock(in_channels: int, channels: int, stride: int = 1, downsample: nn.Module | None = None)`
- Produces: `BasicBlock.expansion == 1`

- [ ] **Step 1: 写BasicBlock失败测试**

```python
import torch
from torch import nn

from models.imagenet_resnet import BasicBlock, conv1x1


def test_basic_block_identity_preserves_shape() -> None:
    block = BasicBlock(64, 64)
    block.eval()
    x = torch.randn(2, 64, 16, 16)

    with torch.no_grad():
        output = block(x)

    assert BasicBlock.expansion == 1
    assert output.shape == (2, 64, 16, 16)
    assert block.downsample is None


def test_basic_block_projection_changes_shape() -> None:
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
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run:

```bat
E:\Anaconda3\envs\resnet-paper\python.exe -m pytest tests\test_imagenet_resnet.py -q
```

Expected: collection error containing `No module named 'models.imagenet_resnet'`.

- [ ] **Step 3: 实现卷积辅助函数与BasicBlock**

```python
"""论文版 ImageNet ResNet-18/34/50/101/152 网络结构。"""
from collections.abc import Callable, Sequence

import torch
from torch import nn


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Conv2d:
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
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
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
```

- [ ] **Step 4: 运行BasicBlock测试并确认通过**

Run:

```bat
E:\Anaconda3\envs\resnet-paper\python.exe -m pytest tests\test_imagenet_resnet.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: 提交BasicBlock**

```bat
git add models\imagenet_resnet.py tests\test_imagenet_resnet.py
git commit -m "feat: add ImageNet ResNet basic block"
```

---

### Task 2: 论文版 Bottleneck

**Files:**
- Modify: `models/imagenet_resnet.py`
- Modify: `tests/test_imagenet_resnet.py`

**Interfaces:**
- Consumes: `conv1x1`, `conv3x3`
- Produces: `Bottleneck(in_channels: int, channels: int, stride: int = 1, downsample: nn.Module | None = None)`
- Produces: `Bottleneck.expansion == 4`
- Guarantees: 阶段切换stride位于第一个 `1x1` 卷积，`3x3` 卷积stride恒为1。

- [ ] **Step 1: 写Bottleneck失败测试**

```python
from models.imagenet_resnet import Bottleneck


def test_bottleneck_uses_paper_stride_placement() -> None:
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
```

- [ ] **Step 2: 运行新增测试并确认失败**

Run:

```bat
E:\Anaconda3\envs\resnet-paper\python.exe -m pytest tests\test_imagenet_resnet.py::test_bottleneck_uses_paper_stride_placement -q
```

Expected: import error containing `cannot import name 'Bottleneck'`.

- [ ] **Step 3: 实现Bottleneck**

```python
class Bottleneck(nn.Module):
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
        identity = x

        output = self.relu(self.bn1(self.conv1(x)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))

        if self.downsample is not None:
            identity = self.downsample(x)

        output = output + identity
        return self.relu(output)
```

- [ ] **Step 4: 运行Block测试并确认全部通过**

Run:

```bat
E:\Anaconda3\envs\resnet-paper\python.exe -m pytest tests\test_imagenet_resnet.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: 提交Bottleneck**

```bat
git add models\imagenet_resnet.py tests\test_imagenet_resnet.py
git commit -m "feat: add paper ImageNet bottleneck"
```

---

### Task 3: ImageNetResNet主干和五个工厂函数

**Files:**
- Modify: `models/imagenet_resnet.py`
- Modify: `tests/test_imagenet_resnet.py`

**Interfaces:**
- Consumes: `BasicBlock`, `Bottleneck`, `conv1x1`
- Produces: `ImageNetResNet(block: type[BasicBlock] | type[Bottleneck], layers: Sequence[int], num_classes: int = 1000)`
- Produces: `resnet18(num_classes: int = 1000) -> ImageNetResNet`
- Produces: `resnet34(num_classes: int = 1000) -> ImageNetResNet`
- Produces: `resnet50(num_classes: int = 1000) -> ImageNetResNet`
- Produces: `resnet101(num_classes: int = 1000) -> ImageNetResNet`
- Produces: `resnet152(num_classes: int = 1000) -> ImageNetResNet`

- [ ] **Step 1: 写配置、stem和输入校验失败测试**

```python
from collections.abc import Callable

import pytest

from models.imagenet_resnet import (
    ImageNetResNet,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)


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
    model = factory()

    assert isinstance(model.layer1[0], block_type)
    assert tuple(
        len(stage)
        for stage in (model.layer1, model.layer2, model.layer3, model.layer4)
    ) == lengths
    assert model.conv1.kernel_size == (7, 7)
    assert model.conv1.stride == (2, 2)
    assert model.maxpool.kernel_size == 3
    assert model.fc.out_features == 1000


@pytest.mark.parametrize("num_classes", [0, -1, 10.5, True])
def test_factory_rejects_invalid_num_classes(num_classes: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        resnet18(num_classes=num_classes)
```

- [ ] **Step 2: 运行配置测试并确认失败**

Run:

```bat
E:\Anaconda3\envs\resnet-paper\python.exe -m pytest tests\test_imagenet_resnet.py -q
```

Expected: import error for `ImageNetResNet` or factory functions.

- [ ] **Step 3: 实现ImageNetResNet、初始化和工厂函数**

```python
BlockType = type[BasicBlock] | type[Bottleneck]


class ImageNetResNet(nn.Module):
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
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._initialize_weights()

    def _make_layer(
        self,
        block: BlockType,
        channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        output_channels = channels * block.expansion
        downsample = None
        if stride != 1 or self.in_channels != output_channels:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, output_channels, stride),
                nn.BatchNorm2d(output_channels),
            )

        modules: list[nn.Module] = [
            block(self.in_channels, channels, stride, downsample)
        ]
        self.in_channels = output_channels
        modules.extend(
            block(self.in_channels, channels)
            for _ in range(1, blocks)
        )
        return nn.Sequential(*modules)

    def _initialize_weights(self) -> None:
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
    return ImageNetResNet(BasicBlock, (2, 2, 2, 2), num_classes)


def resnet34(num_classes: int = 1000) -> ImageNetResNet:
    return ImageNetResNet(BasicBlock, (3, 4, 6, 3), num_classes)


def resnet50(num_classes: int = 1000) -> ImageNetResNet:
    return ImageNetResNet(Bottleneck, (3, 4, 6, 3), num_classes)


def resnet101(num_classes: int = 1000) -> ImageNetResNet:
    return ImageNetResNet(Bottleneck, (3, 4, 23, 3), num_classes)


def resnet152(num_classes: int = 1000) -> ImageNetResNet:
    return ImageNetResNet(Bottleneck, (3, 8, 36, 3), num_classes)
```

- [ ] **Step 4: 运行配置测试并确认通过**

Run:

```bat
E:\Anaconda3\envs\resnet-paper\python.exe -m pytest tests\test_imagenet_resnet.py -q
```

Expected: all current ImageNet tests pass.

- [ ] **Step 5: 提交主干与工厂函数**

```bat
git add models\imagenet_resnet.py tests\test_imagenet_resnet.py
git commit -m "feat: build ImageNet ResNet family"
```

---

### Task 4: 参数量、meta前向与CIFAR回归验证

**Files:**
- Modify: `tests/test_imagenet_resnet.py`

**Interfaces:**
- Consumes: 五个ImageNet工厂函数。
- Verifies: Torchvision `weights=None` 参数量参照。
- Verifies: meta输入 `[1, 3, 224, 224]` 输出 `[1, num_classes]`，不占用CPU卷积或CUDA。
- Verifies: 现有CIFAR测试保持通过。

- [ ] **Step 1: 写参数量与meta前向测试**

```python
import torchvision.models as tv_models


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
    our_model = ours()
    reference_model = reference(weights=None)

    our_parameters = sum(p.numel() for p in our_model.parameters())
    reference_parameters = sum(
        p.numel() for p in reference_model.parameters()
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
    model = factory(num_classes=17).to(device="meta")
    x = torch.empty(1, 3, 224, 224, device="meta")

    output = model(x)

    assert output.shape == (1, 17)
    assert output.device.type == "meta"
```

- [ ] **Step 2: 运行新增测试并确认真实结果**

Run:

```bat
E:\Anaconda3\envs\resnet-paper\python.exe -m pytest tests\test_imagenet_resnet.py -q
```

Expected: all ImageNet tests pass without CUDA allocation. If a declared parameter count differs, stop and inspect the layer responsible instead of changing the expected value.

- [ ] **Step 3: 运行CIFAR模型回归测试**

Run:

```bat
E:\Anaconda3\envs\resnet-paper\python.exe -m pytest tests\test_models.py tests\test_resnet.py -q
```

Expected: all existing CIFAR model tests pass.

- [ ] **Step 4: 运行完整测试套件**

Run:

```bat
set PYTEST_TMPDIR=%TEMP%\imagenet-resnet-final
E:\Anaconda3\envs\resnet-paper\python.exe -m pytest -q --basetemp "%PYTEST_TMPDIR%"
```

Expected: all tests pass; no CUDA process is created.

- [ ] **Step 5: 提交验证测试**

```bat
git add tests\test_imagenet_resnet.py
git commit -m "test: verify ImageNet ResNet structures"
```

---

### Task 5: 学习总结与最终验证

**Files:**
- Create: `docs/阶段九-ImageNet-ResNet结构学习总结.md`

**Interfaces:**
- Consumes: 最终实现与测试结果。
- Produces: 面向复现学习的中文结构说明和验证命令。

- [ ] **Step 1: 编写学习总结**

文档必须逐项解释：

```text
1. ImageNet stem为何使用7x7 stride=2和MaxPool；
2. BasicBlock与Bottleneck的数据流；
3. expansion=1与expansion=4的含义；
4. projection shortcut何时出现；
5. [2,2,2,2]等配置如何得到模型深度；
6. 18/34为什么使用BasicBlock；
7. 50/101/152为什么使用Bottleneck；
8. CIFAR ResNet与ImageNet ResNet的结构差异；
9. 原论文Bottleneck与Torchvision v1.5的stride位置差异；
10. weights=None、预训练权重和“只构建结构”的区别；
11. 参数量对照结果；
12. 如何在不训练ImageNet时验证结构。
```

文档加入以下层数计算示例：

```text
ResNet-18 = stem 1层 + 8个BasicBlock×2层 + FC 1层 = 18层
ResNet-50 = stem 1层 + 16个Bottleneck×3层 + FC 1层 = 50层
ResNet-152 = stem 1层 + 50个Bottleneck×3层 + FC 1层 = 152层
```

- [ ] **Step 2: 检查帮助示例不使用GPU或下载权重**

文档中的结构创建示例必须使用：

```python
from models.imagenet_resnet import resnet50

model = resnet50(num_classes=1000)
print(sum(parameter.numel() for parameter in model.parameters()))
```

不得包含 `.cuda()`、ImageNet下载命令或 `weights=...` 预训练下载调用。

- [ ] **Step 3: 最终静态和测试验证**

Run:

```bat
E:\Anaconda3\envs\resnet-paper\python.exe -m py_compile models\imagenet_resnet.py tests\test_imagenet_resnet.py
git diff --check
E:\Anaconda3\envs\resnet-paper\python.exe -m pytest tests\test_imagenet_resnet.py tests\test_models.py tests\test_resnet.py -q
```

Expected: compilation succeeds, diff check is clean, all selected tests pass.

- [ ] **Step 4: 提交学习总结**

```bat
git add docs\阶段九-ImageNet-ResNet结构学习总结.md
git commit -m "docs: explain ImageNet ResNet structures"
```

- [ ] **Step 5: 核对分支隔离**

Run:

```bat
git status --short
git branch --show-current
git log --oneline -6
```

Expected:

```text
working tree clean
imagenet-resnet-architectures
```

不合并、不推送，等待用户确认后再决定分支处理方式。
