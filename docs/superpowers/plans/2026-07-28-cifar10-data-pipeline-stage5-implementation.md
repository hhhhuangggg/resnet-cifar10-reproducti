# CIFAR-10 数据管道第五阶段 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立并验证论文式 CIFAR-10 数据管道，包括逐像素均值、训练增强、Dataset、DataLoader 和真实数据冒烟测试。

**Architecture:** 所有数据逻辑集中在独立的 `data.py`，训练代码以后只调用公开创建函数。自动测试使用内存假数据和临时目录，不依赖网络；实现完成后再用真实 CIFAR-10 读取训练和测试 batch。

**Tech Stack:** Python 3.11、PyTorch 2.5.1、Torchvision 0.20.1、pytest 8.3.3、CIFAR-10。

## Global Constraints

- 只建立数据管道，不创建模型、不计算 loss、不反向传播、不训练。
- 平均图像只能由未增强的训练集计算，形状严格为 `(3, 32, 32)`。
- 训练 transform 为 `RandomCrop(32, padding=4)`、`RandomHorizontalFlip()`、`ToTensor()`、逐像素均值减法。
- 测试 transform 只使用 `ToTensor()` 和相同的逐像素均值减法。
- 默认 `batch_size=128`、`num_workers=0`、`drop_last=False`。
- 训练集打乱，测试集不打乱。
- 自动测试不得联网或下载 CIFAR-10。
- 现有90项测试必须继续通过。

---

## 文件结构

- Create: `data.py`：全部 CIFAR-10 数据处理接口。
- Create: `tests/test_data.py`：数据计算、变换、缓存与加载测试。
- Create: `docs/阶段五-CIFAR10数据管道学习总结.md`：详细学习说明和验证结果。

### Task 1: 平均图像校验与计算

**Files:**
- Create: `data.py`
- Create: `tests/test_data.py`

**Interfaces:**
- Produces: `validate_per_pixel_mean(mean_image: torch.Tensor) -> None`
- Produces: `compute_per_pixel_mean(dataset: Dataset) -> torch.Tensor`

- [x] **Step 1: 写失败测试**

创建两张已知 `(3,32,32)` float32 Tensor，验证平均值、输出形状和
float32 dtype。验证非Tensor、错误形状、整数、NaN、无穷、范围外均值、
空数据集和错误样本图片被拒绝。

- [x] **Step 2: 运行并确认 RED**

Run: `python -m pytest tests/test_data.py -k "per_pixel_mean" -q`

Expected: 因 `data.py` 或目标函数不存在而失败。

- [x] **Step 3: 实现最小代码**

```python
def validate_per_pixel_mean(mean_image: torch.Tensor) -> None:
    if not isinstance(mean_image, torch.Tensor):
        raise TypeError("mean_image必须是Tensor")
    if mean_image.shape != (3, 32, 32):
        raise ValueError("mean_image形状必须是(3, 32, 32)")
    if not mean_image.is_floating_point():
        raise ValueError("mean_image必须是浮点Tensor")
    if not torch.isfinite(mean_image).all():
        raise ValueError("mean_image必须全部为有限数")
    if torch.any(mean_image < 0) or torch.any(mean_image > 1):
        raise ValueError("mean_image数值必须位于[0, 1]")
```

`compute_per_pixel_mean()` 使用 float64 累加、检查单图并最终返回
float32。

- [x] **Step 4: 运行并确认 GREEN**

Run: `python -m pytest tests/test_data.py -k "per_pixel_mean" -q`

Expected: 平均图像相关测试全部通过。

### Task 2: 平均图像缓存

**Files:**
- Modify: `data.py`
- Modify: `tests/test_data.py`

**Interfaces:**
- Consumes: `validate_per_pixel_mean()`、`compute_per_pixel_mean()`
- Produces: `load_or_compute_per_pixel_mean(raw_train_dataset, cache_path) -> torch.Tensor`

- [x] **Step 1: 写失败测试**

验证缓存不存在时计算并保存；缓存存在时直接读取且不访问Dataset；损坏
缓存产生异常。

- [x] **Step 2: 运行并确认 RED**

Run: `python -m pytest tests/test_data.py -k "mean_cache" -q`

Expected: 因缓存函数不存在而失败。

- [x] **Step 3: 实现最小代码**

```python
def load_or_compute_per_pixel_mean(dataset, cache_path):
    cache_path = Path(cache_path)
    if cache_path.exists():
        mean_image = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=True,
        )
        validate_per_pixel_mean(mean_image)
        return mean_image

    mean_image = compute_per_pixel_mean(dataset)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mean_image, cache_path)
    return mean_image
```

- [x] **Step 4: 运行并确认 GREEN**

Run: `python -m pytest tests/test_data.py -k "mean_cache" -q`

Expected: 缓存测试全部通过。

### Task 3: 逐像素减均值与 transforms

**Files:**
- Modify: `data.py`
- Modify: `tests/test_data.py`

**Interfaces:**
- Produces: `SubtractPerPixelMean(mean_image: torch.Tensor)`
- Produces: `build_cifar10_transforms(mean_image) -> tuple[Compose, Compose]`

- [x] **Step 1: 写失败测试**

验证减法数值、不修改输入、外部均值被修改时内部副本不变、错误输入被
拒绝；验证训练和测试 Compose 的类型、顺序、长度与输出形状；验证测试
transform 连续两次输出相同。

- [x] **Step 2: 运行并确认 RED**

Run: `python -m pytest tests/test_data.py -k "subtract or transforms" -q`

Expected: 因类和构建函数不存在而失败。

- [x] **Step 3: 实现最小代码**

```python
class SubtractPerPixelMean:
    def __init__(self, mean_image: torch.Tensor) -> None:
        validate_per_pixel_mean(mean_image)
        self.mean_image = mean_image.detach().clone()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError("image必须是Tensor")
        if image.shape != (3, 32, 32):
            raise ValueError("image形状必须是(3, 32, 32)")
        if not image.is_floating_point():
            raise ValueError("image必须是浮点Tensor")
        return image - self.mean_image.to(
            device=image.device,
            dtype=image.dtype,
        )
```

按设计创建两个 `transforms.Compose`。

- [x] **Step 4: 运行并确认 GREEN**

Run: `python -m pytest tests/test_data.py -k "subtract or transforms" -q`

Expected: 变换测试全部通过。

### Task 4: CIFAR-10 Dataset

**Files:**
- Modify: `data.py`
- Modify: `tests/test_data.py`

**Interfaces:**
- Consumes: 缓存函数和 transform 构建函数
- Produces: `create_cifar10_datasets(data_dir, download=True) -> tuple[Dataset, Dataset]`

- [x] **Step 1: 写失败测试**

使用 monkeypatch 替换 `torchvision.datasets.CIFAR10`，记录三次构造
调用，验证原始训练集只使用 `ToTensor()`、正式训练/测试标志正确、
download参数传递正确、两个正式Dataset使用同一平均图像。

- [x] **Step 2: 运行并确认 RED**

Run: `python -m pytest tests/test_data.py -k "cifar10_datasets" -q`

Expected: 因 Dataset 创建函数不存在而失败。

- [x] **Step 3: 实现最小代码**

创建 raw train Dataset，读取或计算
`data_dir/cifar10_per_pixel_mean.pt`，再创建正式 train/test Dataset。

- [x] **Step 4: 运行并确认 GREEN**

Run: `python -m pytest tests/test_data.py -k "cifar10_datasets" -q`

Expected: Dataset 测试全部通过。

### Task 5: DataLoader

**Files:**
- Modify: `data.py`
- Modify: `tests/test_data.py`

**Interfaces:**
- Consumes: `create_cifar10_datasets()`
- Produces: `create_cifar10_loaders(data_dir, batch_size=128, num_workers=0, download=True, pin_memory=None) -> tuple[DataLoader, DataLoader]`

- [x] **Step 1: 写失败测试**

使用小型 TensorDataset 替换正式 Dataset 创建函数，验证 batch 形状和
dtype、RandomSampler/SequentialSampler、drop_last=False、参数传递、
自动及显式 pin_memory 和非法参数。

- [x] **Step 2: 运行并确认 RED**

Run: `python -m pytest tests/test_data.py -k "cifar10_loaders" -q`

Expected: 因 DataLoader 创建函数不存在而失败。

- [x] **Step 3: 实现最小代码**

校验参数，`pin_memory is None` 时使用 `torch.cuda.is_available()`，创建
训练和测试 DataLoader。

- [x] **Step 4: 运行并确认 GREEN**

Run: `python -m pytest tests/test_data.py -q`

Expected: 数据模块全部自动测试通过。

### Task 6: 完整验证、真实数据和学习总结

**Files:**
- Create: `docs/阶段五-CIFAR10数据管道学习总结.md`

- [x] **Step 1: 完整回归测试**

Run: `python -m pytest -q`

Expected: 新旧全部测试通过且无 warning。

- [x] **Step 2: 真实 CIFAR-10 冒烟测试**

调用 `create_cifar10_loaders("data")`，打印训练/测试数量、一个训练和
测试 batch 的形状、dtype、标签范围、有限性，并检查均值缓存。

Expected:

```text
train=50000
test=10000
images=[128,3,32,32]
labels=[128]
image dtype=torch.float32
label dtype=torch.int64
label range位于0～9
finite=True
```

- [x] **Step 3: 写详细学习总结**

逐节解释 CIFAR-10、PIL、Tensor、Dataset、transform、数据增强、逐像素
均值、缓存、DataLoader、batch、shuffle、num_workers、pin_memory、
drop_last、测试方法、真实结果，以及与模型初始化和后续训练的关系。

- [x] **Step 4: 最终复验**

Run: `python -m pytest -q`

Expected: 所有测试继续通过。
