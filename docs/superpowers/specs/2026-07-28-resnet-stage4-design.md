# 第四阶段设计：CIFAR-10 ResNet 网络结构

## 1. 阶段目标

第四阶段只实现并验证 CIFAR-10 版本的 ResNet 网络结构，不加载
CIFAR-10、不计算损失、不反向传播，也不执行训练。

本阶段实现：

- ResNet-20
- ResNet-32
- ResNet-44
- ResNet-56

本阶段使用论文的 shortcut Option A，并验证同深度 ResNet 与 PlainNet
具有完全相同的深度、宽度和可训练参数量。

## 2. 论文方案与代码组织

### 论文 shortcut 选项

论文讨论三种 shortcut：

- Option A：维度增加时下采样并补零，所有 shortcut 均无参数；
- Option B：维度增加时使用投影，其他 shortcut 使用恒等映射；
- Option C：所有 shortcut 都使用投影。

本项目复现论文 CIFAR 实验，因此选择 Option A。

### 本项目代码组织

Option A 单独封装为 `OptionAShortcut`。这是为了便于学习和测试，不是
论文强制要求的类结构。

`models/resnet.py` 独立定义：

```text
OptionAShortcut
ResidualBlock
ResNet
```

ResNet 不继承 PlainNet，也不重构当前已经通过测试的 PlainNet。两个模型
保持并列关系，重复少量总体架构代码，以换取教学清晰度和修改隔离性。

## 3. 文件职责

### `models/resnet.py`

负责：

- 定义无参数 `OptionAShortcut`；
- 定义 `ResidualBlock`；
- 定义通用 `ResNet`；
- 创建三个残差阶段；
- 执行权重初始化。

不负责：

- 数据加载；
- loss；
- 反向传播；
- 优化器；
- TensorBoard；
- checkpoint。

### `models/__init__.py`

负责：

- 保留现有模型注册表；
- `family="plain"` 时创建 `PlainNet`；
- `family="resnet"` 时创建 `ResNet`；
- 将注册表中的 `n` 和 `num_classes` 传给模型。

### `tests/test_models.py`

负责验证：

- Option A 的数值、形状和零参数性质；
- ResidualBlock 的普通与降采样行为；
- 残差相加和最终 ReLU 顺序；
- 四种 ResNet 的输出和真实深度；
- ResNet 与 PlainNet 的公平性；
- 参数检查和初始化。

## 4. OptionAShortcut 设计

接口：

```text
OptionAShortcut(in_channels, out_channels, stride)
```

只允许两种行为。

### 恒等映射

```text
in_channels == out_channels
stride == 1
shortcut(x) = x
```

示例：

```text
[2,16,32,32] → [2,16,32,32]
```

输出数值必须与输入完全相同。

### 阶段转换

```text
out_channels == 2 × in_channels
stride == 2
```

先使用空间隔点采样：

```text
x[:, :, ::2, ::2]
```

再对通道维执行对称补零。

16→32 通道时：

```text
前8个零通道 + 原16个通道 + 后8个零通道
```

32→64 通道时：

```text
前16个零通道 + 原32个通道 + 后16个零通道
```

计划使用 `torch.nn.functional.pad`：

```text
pad=(0, 0, 0, 0, padding_before, padding_after)
```

OptionAShortcut 不创建 Conv2d、BatchNorm2d、Linear 或 Parameter，因此
可训练参数量必须为0。

## 5. OptionAShortcut 参数规则

所有参数必须是大于0的整数。

合法组合：

| 通道关系 | stride | 行为 |
|---|---:|---|
| `out_channels == in_channels` | 1 | 恒等映射 |
| `out_channels == 2 × in_channels` | 2 | 隔点采样并补零 |

拒绝：

- 零或负通道；
- 非整数参数；
- stride 为0、3或其他值；
- 通道相同但 stride=2；
- 通道翻倍但 stride=1；
- 输出通道不是输入通道或其两倍。

类型错误抛出 `TypeError`，数值或组合错误抛出 `ValueError`。

## 6. ResidualBlock 设计

接口：

```text
ResidualBlock(in_channels, out_channels, stride=1)
```

残差分支：

```text
Conv3×3(stride)
→ BatchNorm
→ ReLU
→ Conv3×3(stride=1)
→ BatchNorm
```

shortcut 分支：

```text
OptionAShortcut(in_channels, out_channels, stride)
```

合并：

```text
out = residual + shortcut
out = ReLU(out)
```

完整公式：

```text
y = ReLU(F(x) + shortcut(x))
```

第二个 BatchNorm 后不能提前执行 ReLU。提前激活会把负残差清零，使
`ReLU(F(x)) + shortcut(x)` 与论文公式不同。

ResidualBlock 的卷积配置：

- 两层均为3×3；
- padding=1；
- bias=False；
- 第一层使用传入 stride；
- 第二层固定 stride=1。

## 7. ResNet 总体架构

接口：

```text
ResNet(n, num_classes=10)
```

数据流：

```text
输入 [N,3,32,32]
→ stem: Conv3×3(3→16), BN, ReLU
→ stage1: n个ResidualBlock，16通道，32×32
→ stage2: n个ResidualBlock，32通道，16×16
→ stage3: n个ResidualBlock，64通道，8×8
→ AdaptiveAvgPool2d((1,1))
→ flatten [N,64]
→ Linear(64,num_classes)
→ logits [N,num_classes]
```

每个阶段第一个模块单独创建：

```text
ResidualBlock(当前通道, 目标通道, first_stride)
```

剩余 `n-1` 个模块：

```text
ResidualBlock(目标通道, 目标通道, stride=1)
```

`self.in_channels` 的构造期变化为：

```text
16 → 16 → 32 → 64
```

## 8. 深度

深度公式与 PlainNet 相同：

```text
depth = 6n + 2
```

| 模型 | n | ResidualBlock总数 | Conv2d | Linear | 深度 |
|---|---:|---:|---:|---:|---:|
| ResNet-20 | 3 | 9 | 19 | 1 | 20 |
| ResNet-32 | 5 | 15 | 31 | 1 | 32 |
| ResNet-44 | 7 | 21 | 43 | 1 | 44 |
| ResNet-56 | 9 | 27 | 55 | 1 | 56 |

Option A 不包含卷积，因此不会改变深度。

## 9. 初始化

ResNet 与 PlainNet 使用相同初始化：

- Conv2d：Kaiming normal，`mode="fan_out"`，`nonlinearity="relu"`；
- BatchNorm2d weight：1；
- BatchNorm2d bias：0；
- Linear：PyTorch 默认初始化。

OptionAShortcut 没有需要初始化的参数。

## 10. Option A 测试

### 恒等映射

```text
[2,16,32,32] → [2,16,32,32]
```

要求 `torch.equal(output, input)`。

### 降采样与补零

使用 `[1,16,4,4]` 输入，验证：

```text
输出形状：[1,32,2,2]
通道0:8：全0
通道8:24：等于 x[:,:,::2,::2]
通道24:32：全0
```

### 参数量

```text
sum(p.numel() for p in shortcut.parameters()) == 0
```

### 非法参数

覆盖错误类型、零值、错误 stride 和不支持的通道关系。

## 11. ResidualBlock 测试

### 普通模块

```text
ResidualBlock(16,16,1)
[2,16,32,32] → [2,16,32,32]
```

### 降采样模块

```text
ResidualBlock(16,32,2)
[2,16,32,32] → [2,32,16,16]
```

### shortcut 相加

将残差分支权重设为0并使用评估模式，此时：

```text
F(x)=0
y=ReLU(shortcut(x))
```

输出必须与 `torch.relu(shortcut(x))` 相同，以证明 shortcut 参与相加且
最终 ReLU 位于相加之后。

## 12. 完整 ResNet 测试

中间形状：

| 位置 | 预期形状 |
|---|---|
| stem | `[2,16,32,32]` |
| stage1 | `[2,16,32,32]` |
| stage2 | `[2,32,16,16]` |
| stage3 | `[2,64,8,8]` |
| avgpool | `[2,64,1,1]` |

四种公开模型均验证：

```text
[2,3,32,32] → [2,10]
```

自定义分类数：

```text
create_model("resnet20", num_classes=100)
→ [2,100]
```

真实深度通过统计 Conv2d 与 Linear 实例得到，不只检查 `model.depth`
属性。

## 13. PlainNet 与 ResNet 公平性

同深度模型必须具有：

- 相同 stem；
- 相同三个阶段宽度；
- 相同每阶段模块数；
- 相同每模块卷积数；
- 相同池化和分类器；
- 相同初始化规则；
- 完全相同的可训练参数量。

预期：

| 深度 | PlainNet参数量 | ResNet参数量 |
|---:|---:|---:|
| 20 | 269,722 | 269,722 |
| 32 | 464,154 | 464,154 |
| 44 | 658,586 | 658,586 |
| 56 | 853,018 | 853,018 |

唯一核心结构变量：

```text
PlainNet：F(x)
ResNet：ReLU(F(x)+shortcut(x))
```

## 14. 验证范围

自动测试使用 CPU，覆盖：

- shortcut；
- block；
- 四种 ResNet；
- 与 PlainNet 的公平性；
- 原有全部回归测试。

自动测试通过后，使用 RTX 3060 Laptop 对 ResNet-20 执行一次 CUDA
前向传播：

```text
输入：[1,3,32,32]，cuda
输出：[1,10]，cuda:0
```

不要求未训练的 PlainNet 和 ResNet 输出数值相同。

## 15. 逐步实施顺序

1. 建立 OptionAShortcut 类外壳；
2. 实现并验证恒等映射；
3. 实现空间隔点采样；
4. 实现通道对称补零；
5. 验证 shortcut 参数量为0；
6. 添加 shortcut 参数校验；
7. 建立 ResidualBlock；
8. 实现第一组 Conv-BN-ReLU；
9. 实现第二组 Conv-BN；
10. 加入 shortcut、相加和最终 ReLU；
11. 验证普通与降采样模块；
12. 验证相加和激活顺序；
13. 建立 ResNet stem；
14. 实现 `_make_stage()` 与三个阶段；
15. 添加池化、分类器和 forward；
16. 添加权重初始化；
17. 接入统一模型入口；
18. 验证四种深度、输出和自定义类别数；
19. 验证 PlainNet/ResNet 参数量公平性；
20. 完整回归和 CUDA 前向传播。

每一步继续遵循：

```text
讲解原理 → 写失败测试 → 运行确认失败
→ 添加最小实现 → 运行确认通过 → 再继续
```

## 16. 完成标准

第四阶段只有在以下条件全部满足时才完成：

- Option A 恒等映射数值不变；
- 阶段转换使用 stride=2；
- 新增通道通过0填充；
- shortcut 可训练参数量为0；
- ResidualBlock 为两层3×3卷积；
- 第二个卷积后不提前使用 ReLU；
- 相加后执行最终 ReLU；
- ResNet-20/32/44/56 均可创建；
- 中间形状与最终输出正确；
- 真实深度分别为20、32、44、56；
- 自定义 `num_classes` 生效；
- 同深度 PlainNet 与 ResNet 参数量完全相同；
- 原有37项测试继续通过；
- 新增全部测试通过且无 warning；
- ResNet-20 CUDA 前向传播成功；
- 没有加载数据集或执行训练。
