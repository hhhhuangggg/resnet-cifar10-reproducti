# 第四阶段：ResNet 结构实现与学习总结

## 一、本阶段完成了什么

第四阶段实现了论文 CIFAR-10 版本的：

- ResNet-20
- ResNet-32
- ResNet-44
- ResNet-56

本阶段只完成网络结构和结构验证，没有下载或读取数据集，没有计算
loss、反向传播或训练。

使用的 shortcut 是论文 CIFAR-10 实验中的 Option A：尺寸不变时直接
传递输入；尺寸改变时进行隔点采样和通道补零。它不使用卷积，所以不
增加参数。

## 二、实际修改的文件

| 文件 | 本阶段作用 |
|---|---|
| `models/resnet.py` | 实现 OptionAShortcut、ResidualBlock、ResNet |
| `models/__init__.py` | 让 create_model 能创建四种 ResNet |
| `tests/test_resnet.py` | 验证 shortcut、残差块、完整模型和公平性 |
| `docs/superpowers/specs/2026-07-28-resnet-stage4-design.md` | 保存设计依据 |
| `docs/superpowers/plans/2026-07-28-resnet-stage4-implementation.md` | 保存实施步骤 |

## 三、每一步做了什么

### 第1步：先确认接口

检查了第三阶段的 `PlainNet`、`create_model()` 和原有测试。目的是让
ResNet 继续使用相同的模型创建入口和相同的总体架构，避免比较时混入
无关变量。

### 第2步：实现 Option A shortcut

先写测试并观察失败，再添加实现。

构造函数只允许两类组合：

```text
通道相同 + stride=1：恒等映射
通道翻倍 + stride=2：阶段转换
```

恒等映射：

```python
if self.stride == 1:
    return x
```

这不是复制张量，而是把输入直接送到加法位置，数值完全不变。

阶段转换：

```python
x = x[:, :, ::2, ::2]
channel_padding = (self.out_channels - self.in_channels) // 2
return F.pad(
    x,
    (0, 0, 0, 0, channel_padding, channel_padding),
)
```

`::2` 表示每隔一个位置取一个像素，使高和宽减半。以 16→32 通道为例，
先保留原来的16个通道，然后前后分别补8个零通道。

为什么它没有参数：

- 没有 Conv2d；
- 没有 Linear；
- 没有 Parameter；
- 只有张量取样和补零运算。

### 第3步：实现 ResidualBlock

残差分支与 PlainBlock 一样使用两层3×3卷积：

```text
Conv-BN-ReLU-Conv-BN
```

随后加入本阶段特有的 shortcut：

```python
identity = self.shortcut(x)

out = self.conv1(x)
out = self.bn1(out)
out = self.relu1(out)
out = self.conv2(out)
out = self.bn2(out)

out = out + identity
out = self.relu2(out)
```

核心公式是：

```text
y = ReLU(F(x) + shortcut(x))
```

第二层卷积后的 ReLU 不能放在加法前面。正确顺序是先让残差
`F(x)` 与 shortcut 相加，再执行最终 ReLU。

测试中把两层卷积权重设为0。此时 `F(x)=0`，若代码正确，输出必须等于：

```text
ReLU(shortcut(x))
```

这个测试同时证明 shortcut 确实参加了相加，也证明最终 ReLU 位于加法
之后。

### 第4步：实现完整 ResNet

完整数据流为：

```text
[N,3,32,32]
→ stem，16通道
→ stage1，16通道，32×32
→ stage2，32通道，16×16
→ stage3，64通道，8×8
→ 全局平均池化
→ Linear
→ [N,num_classes]
```

`_make_stage()` 的作用是重复创建残差块。每个阶段的第一个块可能负责
改变通道和尺寸，其余块保持尺寸：

```python
blocks = [
    ResidualBlock(
        self.in_channels,
        out_channels,
        first_stride,
    )
]
self.in_channels = out_channels

for _ in range(1, num_blocks):
    blocks.append(
        ResidualBlock(self.in_channels, out_channels, 1)
    )
```

网络深度仍满足：

```text
depth = 6n + 2
```

| 模型 | n | 每阶段块数 | 标称深度 |
|---|---:|---:|---:|
| ResNet-20 | 3 | 3 | 20 |
| ResNet-32 | 5 | 5 | 32 |
| ResNet-44 | 7 | 7 | 44 |
| ResNet-56 | 9 | 9 | 56 |

### 第5步：接入统一模型工厂

在 `models/__init__.py` 中导入 `ResNet`，并根据注册表中的 family
创建对应模型：

```python
if config["family"] == "plain":
    return PlainNet(
        n=config["n"],
        num_classes=num_classes,
    )

return ResNet(
    n=config["n"],
    num_classes=num_classes,
)
```

因此训练程序以后不需要知道模型类的内部细节，只要调用：

```python
create_model("resnet20")
```

### 第6步：完整验证

本阶段最终验证结果：

```text
90 passed in 4.15s
```

参数量：

| 深度 | PlainNet | ResNet | 是否相同 |
|---:|---:|---:|---|
| 20 | 269,722 | 269,722 | 是 |
| 32 | 464,154 | 464,154 | 是 |
| 44 | 658,586 | 658,586 | 是 |
| 56 | 853,018 | 853,018 | 是 |

GPU前向传播：

```text
shape: [1, 10]
device: cuda:0
gpu: NVIDIA GeForce RTX 3060 Laptop GPU
```

## 四、与第三阶段 PlainNet 操作的相同点

| 相同内容 | 原因 |
|---|---|
| 都先写失败测试再实现 | 确认测试能发现缺失功能 |
| 都有两层3×3卷积模块 | 保证网络深度和计算主体一致 |
| 都使用16、32、64三个阶段 | 保证宽度公平 |
| stage2、stage3第一次降采样 | 保证特征图尺寸一致 |
| 都使用全局平均池化和Linear | 保证分类头一致 |
| 都用Kaiming卷积初始化 | 避免初始化成为实验变量 |
| 都通过create_model创建 | 让训练程序使用统一入口 |
| 参数量完全相同 | 保证后续比较主要考察残差连接 |

## 五、与 PlainNet 不一样的地方

### 1. 模块不同

PlainNet：

```text
PlainBlock
```

ResNet：

```text
OptionAShortcut + ResidualBlock
```

### 2. 第二个 ReLU 的位置不同

PlainBlock：

```text
Conv-BN-ReLU-Conv-BN-ReLU
```

ResidualBlock：

```text
Conv-BN-ReLU-Conv-BN-add-ReLU
```

### 3. 多了一条信息通路

PlainNet 只有：

```text
x → F(x)
```

ResNet 同时计算：

```text
x → F(x)
x → shortcut(x)
```

然后相加。这样网络可以学习残差 `F(x)`，输入信息和梯度也有更直接的
传播路径。

### 4. 阶段转换需要对齐两条分支

PlainNet 只要第一层卷积 stride=2 即可。

ResNet 中残差分支和 shortcut 分支必须输出相同形状才能相加，因此：

- 残差分支第一层卷积 stride=2；
- shortcut 分支隔点采样使空间尺寸减半；
- shortcut 分支补零使通道数翻倍。

### 5. 测试重点不同

PlainNet 主要验证卷积模块、阶段尺寸、深度和输出。

ResNet 还必须验证：

- shortcut 数值和形状；
- shortcut 参数量为0；
- 补零通道的位置；
- shortcut 确实参与相加；
- 最终 ReLU 在相加之后；
- 与 PlainNet 参数量完全相同。

## 六、为什么目前还不能比较准确率

这一阶段只证明了：

- 结构能够正确创建；
- 张量能够正确流动；
- 深度、尺寸和参数量符合设计；
- GPU能够执行前向传播。

模型参数仍是随机初始化，所以此时的输出只是未训练 logits，没有分类
意义。下一阶段需要建立数据加载和训练/验证流程，之后才能观察 loss、
准确率以及 PlainNet 和 ResNet 的收敛差异。

