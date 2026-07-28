# CIFAR-10 ResNet 第四阶段 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现并验证论文 CIFAR-10 版 ResNet-20、32、44、56，采用无参数 shortcut Option A，且不执行训练。

**Architecture:** 在独立的 `models/resnet.py` 中实现 `OptionAShortcut`、`ResidualBlock` 和 `ResNet`。总体宽度、深度、池化、分类器和初始化与现有 `PlainNet` 保持一致，唯一核心差异是残差块加入 `ReLU(F(x) + shortcut(x))`。

**Tech Stack:** Python 3.11、PyTorch 2.5.1、pytest 8.3.3、CUDA 12.1 runtime。

## Global Constraints

- 只实现模型结构，不加载 CIFAR-10，不计算 loss，不反向传播，不训练。
- 支持 ResNet-20、32、44、56，对应 `n=3、5、7、9` 和 `depth=6n+2`。
- shortcut 使用论文 Option A：恒等映射，或 stride=2 隔点采样后对称补零。
- shortcut 不得包含可训练参数。
- ResidualBlock 顺序必须为 `Conv-BN-ReLU-Conv-BN-add-ReLU`。
- 同深度 PlainNet 与 ResNet 必须具有完全相同的可训练参数量。
- 保留并继续通过现有 37 项测试。

---

## 文件结构

- Create: `models/resnet.py`：Option A、残差块、完整 ResNet。
- Modify: `models/__init__.py`：让统一模型工厂能够创建四种 ResNet。
- Modify: `tests/test_models.py`：增加 shortcut、block、完整模型和公平性测试。
- Create: `docs/阶段四-ResNet结构实现与学习总结.md`：记录实施步骤、代码原理及与 PlainNet 的差异。

### Task 1: Option A shortcut

**Files:**
- Create: `models/resnet.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Produces: `OptionAShortcut(in_channels: int, out_channels: int, stride: int) -> nn.Module`
- Produces: `forward(x: torch.Tensor) -> torch.Tensor`

- [x] **Step 1: 写失败测试**

测试恒等映射数值不变、阶段转换输出形状与零填充位置、参数量为零，以及非法参数抛出正确异常。

- [x] **Step 2: 运行测试并确认 RED**

Run: `python -m pytest tests/test_models.py -k option_a -q -p no:cacheprovider`

Expected: 因 `models.resnet` 或 `OptionAShortcut` 不存在而失败。

- [x] **Step 3: 最小实现**

```python
class OptionAShortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x
        x = x[:, :, ::2, ::2]
        padding = (self.out_channels - self.in_channels) // 2
        return F.pad(x, (0, 0, 0, 0, padding, padding))
```

- [x] **Step 4: 运行测试并确认 GREEN**

Run: `python -m pytest tests/test_models.py -k option_a -q -p no:cacheprovider`

Expected: 所有 Option A 测试通过。

### Task 2: ResidualBlock

**Files:**
- Modify: `models/resnet.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Consumes: `OptionAShortcut`
- Produces: `ResidualBlock(in_channels: int, out_channels: int, stride: int = 1)`

- [x] **Step 1: 写失败测试**

验证普通块保持形状、转换块降采样一次、两层卷积配置正确；将残差分支归零后验证输出等于 `relu(shortcut(x))`。

- [x] **Step 2: 运行测试并确认 RED**

Run: `python -m pytest tests/test_models.py -k residual_block -q -p no:cacheprovider`

Expected: 因 `ResidualBlock` 不存在而失败。

- [x] **Step 3: 最小实现**

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    identity = self.shortcut(x)
    out = self.relu1(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out = out + identity
    return self.relu2(out)
```

- [x] **Step 4: 运行测试并确认 GREEN**

Run: `python -m pytest tests/test_models.py -k residual_block -q -p no:cacheprovider`

Expected: 所有 ResidualBlock 测试通过。

### Task 3: 完整 ResNet

**Files:**
- Modify: `models/resnet.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Consumes: `ResidualBlock`
- Produces: `ResNet(n: int, num_classes: int = 10)`
- Produces: `depth`、`stage1`、`stage2`、`stage3`、`fc`

- [x] **Step 1: 写失败测试**

验证参数校验、三个阶段各含 n 个块、中间形状、最终输出、真实加权层深度、初始化调用和 BatchNorm 初始化。

- [x] **Step 2: 运行测试并确认 RED**

Run: `python -m pytest tests/test_models.py -k resnet -q -p no:cacheprovider`

Expected: 因 `ResNet` 不存在而失败。

- [x] **Step 3: 最小实现**

实现与 PlainNet 相同的 stem、三个阶段、AdaptiveAvgPool、Linear、权重初始化和完整 forward；阶段内部改用 `ResidualBlock`。

- [x] **Step 4: 运行测试并确认 GREEN**

Run: `python -m pytest tests/test_models.py -k resnet -q -p no:cacheprovider`

Expected: ResNet 类测试通过。

### Task 4: 接入模型工厂并验证公平性

**Files:**
- Modify: `models/__init__.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Consumes: `ResNet`
- Produces: `create_model("resnet20" | "resnet32" | "resnet44" | "resnet56", num_classes)`

- [x] **Step 1: 写失败测试**

验证四个公开名称、四种输出、100 类输出、真实深度，以及每一对同深度 PlainNet/ResNet 参数量相同。

- [x] **Step 2: 运行测试并确认 RED**

Run: `python -m pytest tests/test_models.py -k "create_resnet or resnet_output or fair" -q -p no:cacheprovider`

Expected: 模型工厂仍抛出 `NotImplementedError`。

- [x] **Step 3: 最小实现**

```python
from .resnet import ResNet

if config["family"] == "resnet":
    return ResNet(n=config["n"], num_classes=num_classes)
```

- [x] **Step 4: 运行测试并确认 GREEN**

Run: `python -m pytest tests/test_models.py -q -p no:cacheprovider`

Expected: 所有模型测试通过。

### Task 5: 完整验证与学习文档

**Files:**
- Create: `docs/阶段四-ResNet结构实现与学习总结.md`

- [x] **Step 1: 完整 CPU 回归**

Run: `python -m pytest -q -p no:cacheprovider`

Expected: 全部测试通过，无 warning。

- [x] **Step 2: 参数量检查**

Run: `python -c "from models import create_model; ..."`

Expected: ResNet 参数量依次为 269722、464154、658586、853018，并分别等于 PlainNet。

- [x] **Step 3: CUDA 前向传播**

Run: `python -c "import torch; from models import create_model; ..."`

Expected: ResNet-20 输出 `[1, 10]`，设备 `cuda:0`。

- [x] **Step 4: 整理学习总结**

记录：

- 每个 TDD 循环做了什么；
- Option A、ResidualBlock、ResNet 的关键代码与数据流；
- 本阶段与第三阶段 PlainNet 实现的相同点和不同点；
- 验证结果及下一阶段入口。
