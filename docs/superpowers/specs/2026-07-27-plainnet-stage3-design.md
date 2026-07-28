# 第三阶段设计：CIFAR-10 PlainNet 网络结构

## 1. 阶段目标

第三阶段只实现并验证 CIFAR-10 版本的 PlainNet 网络结构，不加载
CIFAR-10、不计算损失、不反向传播，也不进行五轮训练。

本阶段实现以下四种网络：

- PlainNet-20
- PlainNet-32
- PlainNet-44
- PlainNet-56

本阶段结束时，四种模型必须能够通过统一的 `create_model()` 接口创建，
并能完成 CPU 自动结构测试和一次 PlainNet-20 CUDA 前向传播。

## 2. 实现方案

采用独立 `PlainBlock` 方案。

PlainNet 与后续 ResNet 分别使用独立的基本模块：

```text
PlainBlock:     F(x)
ResidualBlock: F(x) + x
```

第三阶段不使用带开关的共享模块，也不通过关闭 ResNet shortcut 来生成
PlainNet。独立实现能让残差连接的作用更清楚，并避免 PlainNet 依赖尚未
完成的 ResNet。

## 3. 文件职责

### `models/plainnet.py`

负责：

- 定义 `PlainBlock`；
- 定义通用 `PlainNet`；
- 创建 CIFAR-10 PlainNet 网络结构；
- 执行模型参数初始化。

不负责：

- 加载数据；
- 计算损失；
- 反向传播；
- 优化器更新；
- 日志和 checkpoint。

### `models/__init__.py`

负责：

- 保留模型名称与 `n`、深度、模型家族的注册关系；
- 让 `create_model("plain20")` 等名称返回实际 PlainNet；
- 保持尚未实现的 ResNet 名称产生清楚的未实现提示。

### `tests/test_models.py`

负责验证：

- PlainBlock 输入输出形状；
- 四种 PlainNet 的最终输出形状；
- 三个阶段的中间形状；
- 网络深度；
- 自定义分类数；
- 参数校验；
- 模型注册入口。

## 4. PlainBlock 设计

接口：

```text
PlainBlock(in_channels, out_channels, stride=1)
```

计算顺序：

```text
输入
→ 3×3 Conv2d
→ BatchNorm2d
→ ReLU
→ 3×3 Conv2d
→ BatchNorm2d
→ ReLU
→ 输出
```

两层卷积均使用：

- `kernel_size=3`
- `padding=1`
- `bias=False`

第一层卷积使用调用者传入的 `stride`，第二层卷积固定使用
`stride=1`。

卷积后紧接 BatchNorm，因此卷积不使用 bias。BatchNorm 已包含可学习
平移参数，额外的卷积偏置没有必要。

PlainBlock 不包含 shortcut，不执行张量相加。

### 保持尺寸

```text
PlainBlock(16, 16, stride=1)
[N, 16, 32, 32] → [N, 16, 32, 32]
```

### 阶段转换

```text
PlainBlock(16, 32, stride=2)
[N, 16, 32, 32] → [N, 32, 16, 16]
```

只有第一个卷积负责空间降采样。第二个卷积不得再次降采样。

## 5. PlainNet 总体架构

接口：

```text
PlainNet(n, num_classes=10)
```

`n` 表示每个阶段包含的 PlainBlock 数量。模型深度由下式自动确定：

```text
depth = 6n + 2
```

网络数据流：

```text
输入 [N, 3, 32, 32]
→ stem: 3×3 Conv(3→16), BN, ReLU
→ stage1: n个PlainBlock，16通道，32×32
→ stage2: n个PlainBlock，32通道，16×16
→ stage3: n个PlainBlock，64通道，8×8
→ AdaptiveAvgPool2d((1, 1))
→ 展平为 [N, 64]
→ Linear(64, num_classes)
→ logits [N, num_classes]
```

模型内部不添加 Softmax。后续训练使用的 CrossEntropyLoss 会直接接收
logits。

## 6. 三个阶段

| 阶段 | 输出通道 | 空间尺寸 | 首模块 stride |
|---|---:|---:|---:|
| stage1 | 16 | 32×32 | 1 |
| stage2 | 32 | 16×16 | 2 |
| stage3 | 64 | 8×8 | 2 |

每个阶段的第一个模块单独创建：

```text
PlainBlock(当前通道, 目标通道, first_stride)
```

剩余 `n-1` 个模块均为：

```text
PlainBlock(目标通道, 目标通道, stride=1)
```

构造网络期间使用 `self.in_channels` 记录前一个阶段的输出通道，其变化
为：

```text
16 → 16 → 32 → 64
```

阶段使用 `nn.Sequential` 保存和顺序执行 PlainBlock。

## 7. 支持的深度

| 模型 | n | PlainBlock 总数 | Conv2d 数量 | Linear 数量 | 深度 |
|---|---:|---:|---:|---:|---:|
| PlainNet-20 | 3 | 9 | 19 | 1 | 20 |
| PlainNet-32 | 5 | 15 | 31 | 1 | 32 |
| PlainNet-44 | 7 | 21 | 43 | 1 | 44 |
| PlainNet-56 | 9 | 27 | 55 | 1 | 56 |

BatchNorm、ReLU 和池化不计入论文网络深度。

`PlainNet` 底层类接受任意正整数 `n`，但项目公开模型注册表仅暴露本阶段
需要的 `n=3、5、7、9`。

## 8. 参数校验

`PlainNet` 必须检查：

- `n` 必须是整数；
- `n` 必须大于 0；
- `num_classes` 必须是整数；
- `num_classes` 必须大于 0。

模型统一入口继续检查：

- 模型名称必须为字符串；
- 模型名称必须存在于注册表；
- `num_classes` 必须为正整数。

参数类型错误抛出 `TypeError`，数值范围错误抛出 `ValueError`。

## 9. 权重初始化

采用以下初始化策略：

- Conv2d：Kaiming 初始化，匹配 ReLU；
- BatchNorm2d 的缩放参数：1；
- BatchNorm2d 的平移参数：0；
- Linear：保留 PyTorch 默认初始化。

初始化逻辑属于模型构造过程，不属于训练过程。

## 10. 自动测试

### PlainBlock

1. 保持尺寸：

   ```text
   [2, 16, 32, 32] → [2, 16, 32, 32]
   ```

2. 降采样：

   ```text
   [2, 16, 32, 32] → [2, 32, 16, 16]
   ```

### 完整模型

`plain20、plain32、plain44、plain56` 均验证：

```text
[2, 3, 32, 32] → [2, 10]
```

自定义类别数验证：

```text
create_model("plain20", num_classes=100)
[2, 3, 32, 32] → [2, 100]
```

### 中间阶段

使用逐层调用的方式检查：

| 位置 | 预期形状 |
|---|---|
| stem | `[2, 16, 32, 32]` |
| stage1 | `[2, 16, 32, 32]` |
| stage2 | `[2, 32, 16, 16]` |
| stage3 | `[2, 64, 8, 8]` |
| avgpool | `[2, 64, 1, 1]` |

本阶段不引入 forward hook，保持测试直观。

### 深度

统计模型中的 Conv2d 和 Linear 实例，分别得到：

```text
20、32、44、56
```

### 错误参数

覆盖：

- `n=0`
- `n<0`
- `n` 不是整数
- `num_classes=0`
- `num_classes` 不是整数
- 未知模型名称

## 11. 手动验证

自动测试通过后，对 PlainNet-20 运行一次 CUDA 前向传播：

```text
设备：cuda
输入：[1, 3, 32, 32]
输出：[1, 10]
输出设备：cuda
```

同时打印四个模型的可训练参数量，确认参数量随深度增加而增加。

本步骤不进行反向传播，不生成训练结果文件。

## 12. 逐步实施顺序

1. 建立 PlainBlock 类外壳；
2. 添加第一组 Conv-BN-ReLU；
3. 添加第二组 Conv-BN-ReLU；
4. 验证保持尺寸的 PlainBlock；
5. 验证降采样 PlainBlock；
6. 建立 PlainNet 类与 stem；
7. 实现 `_make_stage()`；
8. 添加池化、展平和全连接层；
9. 添加权重初始化；
10. 接入 `create_model()`；
11. 完善自动结构测试；
12. 执行 PlainNet-20 CUDA 前向传播。

每一步都遵循：

```text
讲解原理 → 写一小段代码 → 运行验证 → 解释输出 → 再继续
```

## 13. 完成标准

第三阶段只有在以下条件全部满足时才完成：

- PlainBlock 包含两组 Conv-BN-ReLU；
- PlainBlock 不含 shortcut 或张量相加；
- 保持尺寸与降采样输出正确；
- PlainNet-20/32/44/56 均能通过统一入口创建；
- 四种模型输入 `[2, 3, 32, 32]` 均输出 `[2, 10]`；
- 三个阶段的通道数和空间尺寸正确；
- 网络深度分别为 20、32、44、56；
- 自定义 `num_classes` 生效；
- 参数错误产生明确异常；
- 自动测试全部通过；
- PlainNet-20 CUDA 前向传播通过；
- 没有加载 CIFAR-10 或执行训练。
