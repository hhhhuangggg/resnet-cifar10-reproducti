# 训练与评估核心引擎第六阶段 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现可复用、可测试的训练与评估核心，并用ResNet-20在真实CIFAR-10上完成有限batch GPU冒烟验证。

**Architecture:** 新增独立 `engine.py`，只负责batch移动、指标统计、训练和评估。模型、数据、损失函数、优化器和设备均由调用者传入；`train.py`留待下一阶段组织完整实验。

**Tech Stack:** Python 3.11、PyTorch 2.5.1、Torchvision 0.20.1、pytest 8.3.3、CUDA 12.1。

## Global Constraints

- 训练顺序必须为移动batch、清梯度、forward、loss、backward、step、统计。
- 评估必须使用 `model.eval()` 和 `torch.no_grad()`，不得更新参数。
- loss按样本数加权，accuracy返回0～1。
- `max_batches=None`遍历全部，正整数限制batch数。
- 真实冒烟只训练5个batch、评估3个batch。
- 不实现日志、checkpoint、调度器或完整epoch实验。
- 现有129项测试必须继续通过。

---

## 文件结构

- Create: `engine.py`：训练和评估核心。
- Create: `tests/test_engine.py`：引擎自动测试。
- Create: `docs/阶段六-训练与评估闭环学习总结.md`：详细学习记录。

### Task 1: EpochMetrics与batch移动

**Files:**
- Create: `engine.py`
- Create: `tests/test_engine.py`

**Interfaces:**
- Produces: `EpochMetrics`
- Produces: `move_batch_to_device(batch, device) -> tuple[Tensor, Tensor]`

- [x] **Step 1: 写失败测试**

验证不可修改指标对象、正确CPU移动、形状和值保持，以及错误batch结构、
Tensor类型、图片维度/通道/dtype、标签维度/dtype、样本数和空batch。

- [x] **Step 2: 运行并确认RED**

Run: `python -m pytest tests/test_engine.py -k "metrics or move_batch" -q`

Expected: 因 `engine.py` 或接口不存在而失败。

- [x] **Step 3: 最小实现**

使用 `@dataclass(frozen=True)`，并按设计校验后执行：

```python
images.to(device, non_blocking=True)
labels.to(device, non_blocking=True)
```

- [x] **Step 4: 运行并确认GREEN**

Run: `python -m pytest tests/test_engine.py -k "metrics or move_batch" -q`

Expected: 对应测试全部通过。

### Task 2: batch预测统计与限制参数

**Files:**
- Modify: `engine.py`
- Modify: `tests/test_engine.py`

**Interfaces:**
- Produces: `compute_batch_statistics(logits, labels) -> tuple[int, int]`
- Produces internal: `_validate_max_batches(max_batches) -> None`

- [x] **Step 1: 写失败测试**

使用已知logits验证correct/samples；覆盖错误维度、dtype、N、类别范围和
非有限值；验证None和正整数合法，0、负数、float和bool非法。

- [x] **Step 2: 运行并确认RED**

Run: `python -m pytest tests/test_engine.py -k "statistics or max_batches" -q`

- [x] **Step 3: 最小实现**

使用 `argmax(dim=1)` 与逐项比较；内部函数只负责参数校验。

- [x] **Step 4: 运行并确认GREEN**

Run: `python -m pytest tests/test_engine.py -k "statistics or max_batches" -q`

### Task 3: train_one_epoch

**Files:**
- Modify: `engine.py`
- Modify: `tests/test_engine.py`

**Interfaces:**
- Consumes: batch移动和预测统计
- Produces: `train_one_epoch(...) -> EpochMetrics`

- [x] **Step 1: 写失败测试**

验证训练模式、max_batches、samples/batches、有限loss、accuracy范围、
参数变化、空DataLoader和不同大小batch的加权loss。

- [x] **Step 2: 运行并确认RED**

Run: `python -m pytest tests/test_engine.py -k "train_one_epoch" -q`

- [x] **Step 3: 最小实现**

严格实现：

```text
model.train
→ move
→ zero_grad(set_to_none=True)
→ forward
→ loss
→ backward
→ step
→ metrics
```

- [x] **Step 4: 运行并确认GREEN**

Run: `python -m pytest tests/test_engine.py -k "train_one_epoch" -q`

### Task 4: evaluate

**Files:**
- Modify: `engine.py`
- Modify: `tests/test_engine.py`

**Interfaces:**
- Produces: `evaluate(...) -> EpochMetrics`

- [x] **Step 1: 写失败测试**

验证评估模式、max_batches、指标、空DataLoader、参数逐项不变和不产生
新梯度。

- [x] **Step 2: 运行并确认RED**

Run: `python -m pytest tests/test_engine.py -k "evaluate" -q`

- [x] **Step 3: 最小实现**

在 `model.eval()` 后使用 `with torch.no_grad():` 遍历并统计，不接收
optimizer。

- [x] **Step 4: 运行并确认GREEN**

Run: `python -m pytest tests/test_engine.py -q`

### Task 5: 完整验证、GPU冒烟与学习总结

**Files:**
- Create: `docs/阶段六-训练与评估闭环学习总结.md`

- [x] **Step 1: 完整自动测试**

Run: `python -m pytest -q`

- [x] **Step 2: GPU真实冒烟**

创建ResNet-20、真实DataLoader、CrossEntropyLoss和SGD。训练5个batch，
验证640个样本和参数变化；评估3个batch，验证384个样本和参数不变。

- [x] **Step 3: 写详细学习总结**

解释batch移动、logits、交叉熵、梯度、zero_grad、backward、step、SGD、
训练/评估模式、no_grad、loss加权、accuracy、自动测试和GPU结果。

- [x] **Step 4: 最终复验**

Run: `python -m pytest -q`

Expected: 所有测试通过且无warning。
