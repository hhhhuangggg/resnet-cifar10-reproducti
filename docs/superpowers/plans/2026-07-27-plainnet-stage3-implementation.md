# CIFAR-10 PlainNet 第三阶段实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现并验证 CIFAR-10 PlainNet-20/32/44/56 网络结构，不加载数据集或执行训练。

**Architecture:** 在 `models/plainnet.py` 中独立实现无 shortcut 的 `PlainBlock` 和通用 `PlainNet`，由 `n` 决定深度；在 `models/__init__.py` 中通过现有模型注册表创建四种 PlainNet；在 `tests/test_models.py` 中分层验证模块形状、阶段形状、深度、参数检查和统一入口。

**Tech Stack:** Python 3.11、PyTorch 2.5.1、pytest 8.3.3、CUDA 12.1。

## Global Constraints

- 只实现 PlainNet-20/32/44/56，不实现 ResNet。
- 不加载 CIFAR-10，不计算 loss，不反向传播，不训练。
- PlainBlock 固定为两组 `Conv2d → BatchNorm2d → ReLU`。
- PlainBlock 不包含 shortcut，不执行 `F(x) + x`。
- 卷积使用 `kernel_size=3`、`padding=1`、`bias=False`。
- 只有每个阶段的第一个模块可以使用 `stride=2`。
- 最终输出为 logits，不在模型中添加 Softmax。
- 每个任务由用户亲手完成；助手先讲解，再给出小段代码和验证命令。
- 当前 `.git` 目录为空，暂不执行计划中的 Git 提交；每个测试通过点作为阶段记录点。

---

## 文件结构

- Modify: `models/plainnet.py`
  - 定义 `PlainBlock`、`PlainNet` 和权重初始化。
- Modify: `models/__init__.py`
  - 让 `create_model()` 创建四种 PlainNet。
- Modify: `tests/test_models.py`
  - 添加模块和完整网络结构测试。
- Reference: `docs/superpowers/specs/2026-07-27-plainnet-stage3-design.md`
  - 保存本阶段的结构与验收标准。

---

### Task 1: 建立 PlainBlock 类外壳

**Files:**
- Modify: `models/plainnet.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `torch.Tensor`、`torch.nn.Module`
- Produces: `PlainBlock(in_channels: int, out_channels: int, stride: int = 1)`

- [ ] **Step 1: 添加 PlainBlock 应当存在的测试**

```python
import models.plainnet as plainnet


def test_plain_block_class_exists() -> None:
    assert hasattr(plainnet, "PlainBlock")
```

- [ ] **Step 2: 运行导入测试并确认失败**

Run:

```powershell
python -m pytest tests/test_models.py -q
```

Expected: FAIL，显示 `assert False`，因为模块中还没有 `PlainBlock`。

- [ ] **Step 3: 在 plainnet.py 添加导入和类外壳**

```python
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
```

- [ ] **Step 4: 重新运行存在性测试**

Run:

```powershell
python -m pytest tests/test_models.py -q
```

Expected: PASS；此时只证明类已经存在，还不判断真实卷积结果。

---

### Task 2: 实现 PlainBlock 第一组 Conv-BN-ReLU

**Files:**
- Modify: `models/plainnet.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `[N, in_channels, H, W]`
- Produces: 第一组运算后的 `[N, out_channels, H', W']`

- [ ] **Step 1: 添加第一组运算的形状测试**

```python
import torch
from torch import nn


def test_plain_block_first_path_downsamples() -> None:
    block = PlainBlock(16, 32, stride=2)
    x = torch.randn(2, 16, 32, 32)

    y = block.relu1(block.bn1(block.conv1(x)))

    assert y.shape == (2, 32, 16, 16)
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```powershell
python -m pytest tests/test_models.py::test_plain_block_first_path_downsamples -q
```

Expected: FAIL，提示 `PlainBlock` 没有 `conv1`。

- [ ] **Step 3: 添加第一组层**

在 `PlainBlock.__init__()` 中添加：

```python
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
```

- [ ] **Step 4: 运行测试并确认通过**

Run:

```powershell
python -m pytest tests/test_models.py::test_plain_block_first_path_downsamples -q
```

Expected: PASS，形状为 `[2, 32, 16, 16]`。

---

### Task 3: 完成 PlainBlock 第二组运算与 forward

**Files:**
- Modify: `models/plainnet.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `[N, in_channels, H, W]`
- Produces: `[N, out_channels, H/stride, W/stride]`

- [ ] **Step 1: 添加完整模块形状测试**

```python
def test_plain_block_preserves_shape() -> None:
    block = PlainBlock(16, 16, stride=1)
    x = torch.randn(2, 16, 32, 32)

    y = block(x)

    assert y.shape == (2, 16, 32, 32)


def test_plain_block_downsamples_once() -> None:
    block = PlainBlock(16, 32, stride=2)
    x = torch.randn(2, 16, 32, 32)

    y = block(x)

    assert y.shape == (2, 32, 16, 16)
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```powershell
python -m pytest tests/test_models.py::test_plain_block_preserves_shape tests/test_models.py::test_plain_block_downsamples_once -q
```

Expected: 至少降采样测试失败，因为当前 `forward()` 仍返回原输入。

- [ ] **Step 3: 添加第二组层**

在 `PlainBlock.__init__()` 中添加：

```python
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
```

- [ ] **Step 4: 实现 forward**

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu2(x)

    return x
```

- [ ] **Step 5: 运行模块测试并确认通过**

Run:

```powershell
python -m pytest tests/test_models.py -q
```

Expected: 两个完整模块形状测试均通过。

---

### Task 4: 添加 PlainBlock 参数检查

**Files:**
- Modify: `models/plainnet.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `in_channels`、`out_channels`、`stride`
- Produces: 合法 PlainBlock 或明确异常

- [ ] **Step 1: 添加错误参数测试**

```python
import pytest


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
    with pytest.raises(exception_type):
        PlainBlock(*arguments)
```

- [ ] **Step 2: 运行并确认失败**

Run:

```powershell
python -m pytest tests/test_models.py::test_plain_block_rejects_invalid_arguments -q
```

Expected: FAIL，因为当前没有主动校验。

- [ ] **Step 3: 在创建卷积前添加校验**

```python
for name, value in (
    ("in_channels", in_channels),
    ("out_channels", out_channels),
    ("stride", stride),
):
    if not isinstance(value, int):
        raise TypeError(f"{name}必须是整数")
    if value <= 0:
        raise ValueError(f"{name}必须大于0")
```

- [ ] **Step 4: 运行并确认通过**

Run:

```powershell
python -m pytest tests/test_models.py::test_plain_block_rejects_invalid_arguments -q
```

Expected: PASS。

---

### Task 5: 建立 PlainNet 类与 stem

**Files:**
- Modify: `models/plainnet.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `PlainNet(n: int, num_classes: int = 10)`
- Produces: 包含 stem 的模型对象

- [ ] **Step 1: 添加 stem 输出形状测试**

```python
from models.plainnet import PlainNet


def test_plainnet_stem_shape() -> None:
    model = PlainNet(n=3)
    x = torch.randn(2, 3, 32, 32)

    y = model.stem(x)

    assert y.shape == (2, 16, 32, 32)
```

- [ ] **Step 2: 运行并确认失败**

Run:

```powershell
python -m pytest tests/test_models.py::test_plainnet_stem_shape -q
```

Expected: FAIL，提示无法导入 `PlainNet`。

- [ ] **Step 3: 添加 PlainNet 类、参数检查和 stem**

```python
class PlainNet(nn.Module):
    """论文中用于CIFAR图像分类的PlainNet。"""

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
```

- [ ] **Step 4: 运行并确认通过**

Run:

```powershell
python -m pytest tests/test_models.py::test_plainnet_stem_shape -q
```

Expected: PASS。

---

### Task 6: 实现 `_make_stage()` 与三个阶段

**Files:**
- Modify: `models/plainnet.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `_make_stage(out_channels: int, num_blocks: int, first_stride: int)`
- Produces: `nn.Sequential` 阶段

- [ ] **Step 1: 添加阶段模块数量测试**

```python
@pytest.mark.parametrize("n", [3, 5, 7, 9])
def test_plainnet_has_n_blocks_per_stage(n: int) -> None:
    model = PlainNet(n=n)

    assert len(model.stage1) == n
    assert len(model.stage2) == n
    assert len(model.stage3) == n
```

- [ ] **Step 2: 运行并确认失败**

Run:

```powershell
python -m pytest tests/test_models.py::test_plainnet_has_n_blocks_per_stage -q
```

Expected: FAIL，提示没有 `stage1`。

- [ ] **Step 3: 添加 `_make_stage()`**

```python
def _make_stage(
    self,
    out_channels: int,
    num_blocks: int,
    first_stride: int,
) -> nn.Sequential:
    blocks = [
        PlainBlock(
            self.in_channels,
            out_channels,
            stride=first_stride,
        )
    ]
    self.in_channels = out_channels

    for _ in range(1, num_blocks):
        blocks.append(
            PlainBlock(
                self.in_channels,
                out_channels,
                stride=1,
            )
        )

    return nn.Sequential(*blocks)
```

- [ ] **Step 4: 在 `__init__()` 中创建三个阶段**

在 stem 后添加：

```python
self.stage1 = self._make_stage(16, n, first_stride=1)
self.stage2 = self._make_stage(32, n, first_stride=2)
self.stage3 = self._make_stage(64, n, first_stride=2)
```

- [ ] **Step 5: 运行并确认通过**

Run:

```powershell
python -m pytest tests/test_models.py::test_plainnet_has_n_blocks_per_stage -q
```

Expected: 四组参数均 PASS。

---

### Task 7: 验证三个阶段的中间形状

**Files:**
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `model.stem`、`stage1`、`stage2`、`stage3`
- Produces: 论文规定的中间张量形状

- [ ] **Step 1: 添加逐层形状测试**

```python
def test_plainnet_stage_shapes() -> None:
    model = PlainNet(n=3)
    x = torch.randn(2, 3, 32, 32)

    x = model.stem(x)
    assert x.shape == (2, 16, 32, 32)

    x = model.stage1(x)
    assert x.shape == (2, 16, 32, 32)

    x = model.stage2(x)
    assert x.shape == (2, 32, 16, 16)

    x = model.stage3(x)
    assert x.shape == (2, 64, 8, 8)
```

- [ ] **Step 2: 运行测试**

Run:

```powershell
python -m pytest tests/test_models.py::test_plainnet_stage_shapes -q
```

Expected: PASS。若失败，先修正阶段首模块的通道或 stride，不进入下一任务。

---

### Task 8: 添加池化、分类器与完整 forward

**Files:**
- Modify: `models/plainnet.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `[N, 3, 32, 32]`
- Produces: logits `[N, num_classes]`

- [ ] **Step 1: 添加完整输出测试**

```python
def test_plainnet20_output_shape() -> None:
    model = PlainNet(n=3, num_classes=10)
    x = torch.randn(2, 3, 32, 32)

    y = model(x)

    assert y.shape == (2, 10)
```

- [ ] **Step 2: 运行并确认失败**

Run:

```powershell
python -m pytest tests/test_models.py::test_plainnet20_output_shape -q
```

Expected: FAIL，因为 PlainNet 还没有完整 `forward()`。

- [ ] **Step 3: 添加池化和分类器**

在 `__init__()` 中添加：

```python
self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
self.fc = nn.Linear(64, num_classes)
```

- [ ] **Step 4: 实现 forward**

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.stem(x)
    x = self.stage1(x)
    x = self.stage2(x)
    x = self.stage3(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    return x
```

- [ ] **Step 5: 运行并确认通过**

Run:

```powershell
python -m pytest tests/test_models.py::test_plainnet20_output_shape -q
```

Expected: PASS，输出为 `[2, 10]`。

---

### Task 9: 添加权重初始化

**Files:**
- Modify: `models/plainnet.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: 模型内的 Conv2d 和 BatchNorm2d
- Produces: Kaiming 卷积权重、BN weight=1、BN bias=0

- [ ] **Step 1: 添加 BatchNorm 初始化测试**

```python
def test_plainnet_batchnorm_initialization() -> None:
    model = PlainNet(n=3)

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            assert torch.all(module.weight == 1)
            assert torch.all(module.bias == 0)
```

- [ ] **Step 2: 添加初始化方法**

```python
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
```

- [ ] **Step 3: 在 `__init__()` 最后调用**

```python
self._initialize_weights()
```

- [ ] **Step 4: 运行初始化测试**

Run:

```powershell
python -m pytest tests/test_models.py::test_plainnet_batchnorm_initialization -q
```

Expected: PASS。

---

### Task 10: 接入统一模型注册入口

**Files:**
- Modify: `models/__init__.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `create_model(name: str, num_classes: int = 10)`
- Produces: 与模型名称对应的 PlainNet

- [ ] **Step 1: 添加四种模型创建测试**

```python
from models import create_model


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
    model = create_model(name)

    assert isinstance(model, PlainNet)
    assert model.depth == expected_depth
```

- [ ] **Step 2: 运行并确认失败**

Run:

```powershell
python -m pytest tests/test_models.py::test_create_plainnet_models -q
```

Expected: FAIL，因为入口仍抛出 `NotImplementedError`。

- [ ] **Step 3: 导入 PlainNet 并创建 plain family**

在 `models/__init__.py` 中添加：

```python
from .plainnet import PlainNet
```

将函数结尾替换为：

```python
config = MODEL_CONFIGS[normalized_name]

if config["family"] == "plain":
    return PlainNet(
        n=config["n"],
        num_classes=num_classes,
    )

raise NotImplementedError(
    f"{normalized_name}尚未实现，配置为：{config}"
)
```

- [ ] **Step 4: 运行创建测试**

Run:

```powershell
python -m pytest tests/test_models.py::test_create_plainnet_models -q
```

Expected: 四种模型全部 PASS。

---

### Task 11: 验证完整输出、自定义类别数和深度

**Files:**
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: 四种公开 PlainNet 名称
- Produces: 正确输出和论文深度

- [ ] **Step 1: 添加四种模型输出测试**

```python
@pytest.mark.parametrize(
    "name",
    ["plain20", "plain32", "plain44", "plain56"],
)
def test_plainnet_output_shapes(name: str) -> None:
    model = create_model(name)
    x = torch.randn(2, 3, 32, 32)

    y = model(x)

    assert y.shape == (2, 10)
```

- [ ] **Step 2: 添加自定义类别数测试**

```python
def test_plainnet_custom_num_classes() -> None:
    model = create_model("plain20", num_classes=100)
    x = torch.randn(2, 3, 32, 32)

    y = model(x)

    assert y.shape == (2, 100)
```

- [ ] **Step 3: 添加深度统计测试**

```python
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
    model = create_model(name)
    actual_depth = sum(
        isinstance(module, (nn.Conv2d, nn.Linear))
        for module in model.modules()
    )

    assert actual_depth == expected_depth
```

- [ ] **Step 4: 运行全部模型测试**

Run:

```powershell
python -m pytest tests/test_models.py -q
```

Expected: 全部 PASS。

---

### Task 12: 完整回归、参数量和 CUDA 前向传播

**Files:**
- Verify: `models/plainnet.py`
- Verify: `models/__init__.py`
- Verify: `tests/test_models.py`
- Verify: `tests/test_train.py`

**Interfaces:**
- Consumes: 完整第三阶段实现
- Produces: 自动测试、参数量规律和 GPU 前向传播证据

- [ ] **Step 1: 运行全部自动测试**

Run:

```powershell
python -m pytest -q
```

Expected: 所有模型测试和第二阶段训练配置测试全部 PASS，无 warning。

- [ ] **Step 2: 打印四种模型参数量**

Run:

```powershell
python -c "from models import create_model; names=('plain20','plain32','plain44','plain56'); [print(name, sum(p.numel() for p in create_model(name).parameters() if p.requires_grad)) for name in names]"
```

Expected:

```text
plain20 < plain32 < plain44 < plain56
```

具体数值由实现计算，不在测试中硬编码。

- [ ] **Step 3: 运行 PlainNet-20 CUDA 前向传播**

Run:

```powershell
python -c "import torch; from models import create_model; model=create_model('plain20').cuda().eval(); x=torch.randn(1,3,32,32,device='cuda'); y=model(x); print('shape:', list(y.shape)); print('device:', y.device)"
```

Expected:

```text
shape: [1, 10]
device: cuda:0
```

- [ ] **Step 4: 确认第三阶段没有越界**

检查：

- `data` 目录没有因本阶段下载 CIFAR-10；
- `outputs` 没有生成训练 checkpoint；
- `runs` 没有生成 TensorBoard 训练日志；
- `train.py` 尚未开始训练循环。

- [ ] **Step 5: 在学习日报记录结果**

记录：

- 自动测试通过数量；
- 四种模型参数量；
- CUDA 输出形状与设备；
- 自己对 PlainBlock、阶段降采样和深度公式的理解；
- 下一阶段准备学习的内容。
