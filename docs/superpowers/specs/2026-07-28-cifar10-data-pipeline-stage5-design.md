# 第五阶段设计：CIFAR-10 数据管道

## 1. 阶段目标

第五阶段只建立并验证 CIFAR-10 数据管道，不创建模型、不计算损失、
不反向传播，也不执行训练。

本阶段完成：

- 下载和读取 CIFAR-10；
- 从原始训练集计算逐像素平均图像；
- 缓存并复用平均图像；
- 建立论文式训练集数据增强；
- 建立无随机增强的测试集预处理；
- 创建训练和测试 DataLoader；
- 检查真实数据的数量、形状、类型、标签和数值；
- 编写详细学习总结。

## 2. 采用的论文预处理方案

采用论文优先方案：

```text
训练集：
原始图像
→ 四周补4像素
→ 随机裁剪回32×32
→ 50%概率水平翻转
→ 转为Tensor
→ 减去训练集逐像素平均图像

测试集：
原始图像
→ 转为Tensor
→ 减去同一张训练集逐像素平均图像
```

训练集和测试集共享基础中心化处理，但只有训练集使用随机增强。测试集
必须保持确定性，以便同一个模型在相同参数下得到可重复的评估结果。

## 3. 文件职责

### `data.py`

负责：

- 计算逐像素平均图像；
- 校验平均图像；
- 保存和读取平均图像缓存；
- 定义逐像素均值减法变换；
- 创建训练和测试 transform；
- 创建 CIFAR-10 Dataset；
- 创建训练和测试 DataLoader。

不负责：

- 创建 PlainNet 或 ResNet；
- 将 batch 移动到 GPU；
- 计算预测、loss 或准确率；
- 反向传播；
- 优化器；
- 训练循环；
- TensorBoard 和 checkpoint。

### `tests/test_data.py`

使用小型内存假数据验证：

- 平均图像计算；
- 平均图像校验；
- 缓存保存与读取；
- transform 组成和行为；
- Dataset 与 DataLoader 接口；
- batch 形状和类型；
- shuffle、drop_last 和 pin_memory 配置。

自动测试不下载 CIFAR-10，不依赖网络。

### `data/`

保存：

```text
CIFAR-10原始数据
cifar10_per_pixel_mean.pt
```

项目已有 `data/` 目录，继续复用，不改变位置。

### 学习总结

创建：

```text
docs/阶段五-CIFAR10数据管道学习总结.md
```

详细记录每一步的目的、代码、张量形状、常见误解、测试方法和真实数据
检查结果。

## 4. 公开接口

`data.py` 提供：

```python
compute_per_pixel_mean(dataset) -> torch.Tensor

validate_per_pixel_mean(mean_image) -> None

load_or_compute_per_pixel_mean(
    raw_train_dataset,
    cache_path,
) -> torch.Tensor

build_cifar10_transforms(
    mean_image,
) -> tuple[Callable, Callable]

create_cifar10_datasets(
    data_dir,
    download=True,
) -> tuple[Dataset, Dataset]

create_cifar10_loaders(
    data_dir,
    batch_size=128,
    num_workers=0,
    download=True,
    pin_memory=None,
) -> tuple[DataLoader, DataLoader]
```

`pin_memory=None` 时根据 `torch.cuda.is_available()` 自动决定；显式传入
布尔值时使用调用者指定值，便于测试和以后在 CPU 环境运行。

## 5. CIFAR-10 数据格式

CIFAR-10 包含：

| 内容 | 数量或格式 |
|---|---|
| 训练集 | 50,000张 |
| 测试集 | 10,000张 |
| 类别数 | 10 |
| 原始尺寸 | 32×32 |
| 颜色 | RGB |
| 标签范围 | 0～9 |

PIL 图像在概念上以高度、宽度、通道组织。`ToTensor()` 将其转换为：

```text
torch.float32
[通道, 高度, 宽度]
[3,32,32]
```

同时将8位像素从 `0～255` 缩放到 `0.0～1.0`。

## 6. 逐像素平均图像

平均图像只由未增强的原始训练集计算：

```text
50,000张训练Tensor：[50000,3,32,32]
沿样本维求平均
→ mean_image：[3,32,32]
```

实现时逐张累加，不将50,000张图同时堆入内存：

```python
total = torch.zeros(3, 32, 32, dtype=torch.float64)

for image, _ in dataset:
    total += image.to(torch.float64)

mean_image = (total / len(dataset)).to(torch.float32)
```

使用 float64 累加以减小大量求和的舍入误差，最终转换为模型输入使用的
float32。

计算均值时使用的临时原始训练集只应用 `ToTensor()`，不得使用随机裁剪、
随机翻转或均值减法。

禁止从测试集计算平均图像，也禁止将测试集参与训练集平均值计算。

## 7. 平均图像校验与缓存

合法平均图像必须满足：

- 类型为 `torch.Tensor`；
- 形状严格等于 `(3, 32, 32)`；
- dtype 为浮点类型；
- 所有数值有限，不包含 NaN 或正负无穷；
- 所有数值位于 `[0, 1]`。

类型错误抛出 `TypeError`，形状、dtype 或数值错误抛出 `ValueError`。

缓存位置：

```text
data/cifar10_per_pixel_mean.pt
```

行为：

```text
缓存存在
→ torch.load到CPU
→ 校验
→ 返回

缓存不存在
→ 从原始训练集计算
→ 校验
→ 创建父目录
→ torch.save
→ 返回
```

缓存存在但损坏时直接报告错误，不静默使用损坏数据，也不删除用户文件。

## 8. 自定义逐像素均值变换

定义：

```python
class SubtractPerPixelMean:
    def __init__(self, mean_image: torch.Tensor) -> None:
        ...

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image - self.mean_image
```

构造时校验平均图像并保存一份独立副本，防止调用者之后原地修改外部
Tensor。

调用时要求图片：

- 类型为 `torch.Tensor`；
- 形状为 `(3, 32, 32)`；
- dtype 为浮点类型。

该变换没有可训练参数，不执行标准差缩放，只执行论文要求的逐像素
中心化。

## 9. 训练与测试 transform

训练 transform：

```python
transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        SubtractPerPixelMean(mean_image),
    ]
)
```

测试 transform：

```python
transforms.Compose(
    [
        transforms.ToTensor(),
        SubtractPerPixelMean(mean_image),
    ]
)
```

裁剪和翻转位于 `ToTensor()` 之前，直接作用于 PIL 图像。均值减法必须
位于 `ToTensor()` 之后，因为它需要 Tensor 输入。

随机裁剪始终输出32×32；水平翻转默认概率为0.5。测试 transform 不包含
任何随机操作。

## 10. Dataset 创建流程

第一次创建无增强训练集：

```python
raw_train_dataset = CIFAR10(
    root=data_dir,
    train=True,
    transform=transforms.ToTensor(),
    download=download,
)
```

用它读取或计算 `mean_image`。随后使用同一个 `mean_image` 创建正式
Dataset：

```text
正式train_dataset：train=True + train_transform
正式test_dataset：train=False + test_transform
```

两者都以同一个 `data_dir` 为根目录。Torchvision 检测到数据已经存在后
不会重复下载。

函数返回：

```python
(train_dataset, test_dataset)
```

真实数据数量预期分别为50,000和10,000。

## 11. DataLoader 创建流程

默认配置：

| 参数 | 训练集 | 测试集 |
|---|---:|---:|
| `batch_size` | 128 | 128 |
| `shuffle` | True | False |
| `num_workers` | 0 | 0 |
| `pin_memory` | CUDA可用时True | CUDA可用时True |
| `drop_last` | False | False |

`batch_size` 必须是大于0的整数；`num_workers` 必须是非负整数；
`pin_memory` 必须是 `bool` 或 `None`。非法类型抛出 `TypeError`，非法
数值抛出 `ValueError`。

训练集打乱顺序以减少固定样本次序的影响。测试集不打乱，以保持评估
顺序稳定。

`drop_last=False` 保留不足一个完整 batch 的最后一批。训练集
50,000张、batch size 128时，最后一批包含80张。

DataLoader 只产生 CPU Tensor。`pin_memory=True` 只优化以后从CPU复制到
CUDA的过程，不会自动把数据移动到GPU。

## 12. Dataset、DataLoader 与 batch

Dataset 返回一条样本：

```text
image：[3,32,32]，torch.float32
label：Python整数，范围0～9
```

DataLoader 将多条样本堆叠：

```text
images：[batch_size,3,32,32]，torch.float32
labels：[batch_size]，torch.int64
```

默认 batch size 128 时：

```text
images：[128,3,32,32]
labels：[128]
```

这里 Tensor 是 PyTorch 的多维数值容器。`ToTensor()` 不只是将像素变成
数字，还会改变维度顺序、缩放数值、转换数据类型，使图片能参与卷积、
GPU计算和自动求导。

## 13. 错误处理

明确拒绝：

- 空数据集，无法计算平均值；
- 数据样本不是 `(image, label)` 结构；
- 图片无法转换为 `(3,32,32)` Tensor；
- 非法平均图像；
- 非法 batch size；
- 非法 num_workers；
- 非法 pin_memory；
- 缓存读取失败或缓存内容损坏；
- CIFAR-10 下载或校验失败。

下载失败保留 Torchvision 的原始异常信息，不切换到假数据，不假装真实
数据已经准备完成。

## 14. 自动测试设计

自动测试不联网，使用小型假 Dataset 和临时目录。

### 平均图像

- 两张已知Tensor的平均值计算正确；
- 输出形状为 `(3,32,32)`；
- 输出 dtype 为 float32；
- 空数据集被拒绝；
- 错误图片形状被拒绝。

### 平均图像校验

- 正常Tensor通过；
- 非Tensor被拒绝；
- `[3]`、`[1,3,32,32]` 等错误形状被拒绝；
- 整数Tensor被拒绝；
- NaN、无穷和范围外数值被拒绝。

### 缓存

- 缓存不存在时计算并保存；
- 缓存存在时直接读取，不再次访问数据集；
- 损坏缓存产生明确异常。

### 均值减法变换

- 输出等于 `image - mean_image`；
- 不原地修改输入；
- 错误类型、形状和dtype被拒绝。

### transform

- 训练 Compose 按设计包含四个步骤；
- 测试 Compose 只包含两个固定步骤；
- 训练增强后仍为 `(3,32,32)`；
- 测试图片连续读取两次结果完全相同。

### Dataset 与 DataLoader

通过替换测试中的 CIFAR10 构造器，验证：

- train/test 标志正确；
- 两者共享同一 mean image；
- DataLoader batch 形状与类型正确；
- 训练 sampler 为随机采样；
- 测试 sampler 为顺序采样；
- `drop_last=False`；
- 参数校验正确。

## 15. 真实数据冒烟测试

自动测试全部通过后，允许 Torchvision 下载或复用真实 CIFAR-10，然后
只读取一个训练 batch 和一个测试 batch，不创建模型。

验证：

```text
train_dataset数量：50000
test_dataset数量：10000
train_images：[128,3,32,32]
train_labels：[128]
test_images：[128,3,32,32]
test_labels：[128]
images dtype：torch.float32
labels dtype：torch.int64
标签最小值>=0
标签最大值<=9
图片不含NaN或无穷
```

同时输出平均图像的：

```text
shape
dtype
min
max
mean
```

不要求增强后的图片数值仍位于 `[0,1]`，因为减去平均图像后可以出现
负数。

## 16. 与模型参数初始化的区别

逐像素均值减法处理输入数据：

```text
x_centered = x - mean_image
```

Kaiming 初始化处理卷积权重：

```text
W = 适合ReLU传播的随机初始参数
```

模型计算：

```text
y = W * x_centered
```

两者作用于不同对象，不能互相替代。第五阶段只实现输入数据中心化；
Kaiming 初始化已经在第三、第四阶段的模型类中完成。

## 17. 实施顺序

1. 建立 `data.py` 模块外壳；
2. 实现平均图像校验；
3. 实现平均图像计算；
4. 实现缓存读取和保存；
5. 实现 `SubtractPerPixelMean`；
6. 创建训练和测试 transform；
7. 创建正式 CIFAR-10 Dataset；
8. 创建训练和测试 DataLoader；
9. 验证参数和错误信息；
10. 运行全部自动测试；
11. 下载或复用真实 CIFAR-10；
12. 检查真实训练和测试 batch；
13. 编写详细学习总结。

实现继续遵循：

```text
写失败测试
→ 运行并确认因缺少目标功能而失败
→ 添加最小实现
→ 运行并确认通过
→ 进入下一项
```

## 18. 完成标准

第五阶段只有在以下条件全部满足时完成：

- CIFAR-10 能够下载和读取；
- 训练集数量为50,000，测试集数量为10,000；
- mean image 只由未增强的原始训练集计算；
- mean image 形状为 `(3,32,32)`、dtype为float32且数值有效；
- mean image 能够缓存并再次读取；
- 训练集使用补边、随机裁剪和随机水平翻转；
- 测试集不使用随机增强；
- 两者减去同一张训练集 mean image；
- 单张图片为 `(3,32,32)`；
- 默认完整 batch 为 `(128,3,32,32)`；
- 标签 batch 为 `(128)`、dtype为int64且范围0～9；
- 训练集打乱、测试集不打乱；
- 最后一批数据不被丢弃；
- 全部旧测试继续通过；
- 新测试全部通过且无 warning；
- 未创建模型、未计算loss、未训练；
- 详细学习总结包含代码、数据流、概念和验证结果。
