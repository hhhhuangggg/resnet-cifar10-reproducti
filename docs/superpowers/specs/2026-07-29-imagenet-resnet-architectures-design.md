# ImageNet ResNet-18/34/50/101/152 结构设计

## 目标

在不影响 CIFAR-10 正式训练的前提下，亲手实现论文中的 ImageNet
ResNet-18、34、50、101、152 网络结构，并使用 Torchvision 作为结构与参数量
参照。当前阶段只构建和验证模型，不下载 ImageNet、不训练 ImageNet，也不下载
预训练权重。

## 工作区隔离

- CIFAR-10 正式训练继续使用 `E:\文档\resnet复现`，分支为
  `paper-64k-schedule`。
- ImageNet 结构开发使用 `E:\文档\resnet-imagenet结构`，分支为
  `imagenet-resnet-architectures`。
- 两个目录共享 Git 历史，但工作文件相互独立。
- ImageNet 开发不修改 CIFAR-10 正式训练参数和输出目录。

## 文件边界

新增 `models/imagenet_resnet.py`，负责：

- `BasicBlock`；
- `Bottleneck`；
- ImageNet `ResNet` 主体；
- ResNet-18/34/50/101/152 工厂函数。

现有 `models/resnet.py` 继续只负责 CIFAR-10 ResNet-20/32/44/56，不把两套
输入尺寸、stem 和阶段规则强行合并到同一个类中。

新增独立测试文件，验证 ImageNet 结构，不改变现有 CIFAR 模型测试的语义。

## 共同主干

模型输入为 `[N, 3, 224, 224]`，默认分类数为 1000：

1. `7x7 Conv`，64 通道，stride 2，padding 3；
2. BatchNorm；
3. ReLU；
4. `3x3 MaxPool`，stride 2，padding 1；
5. 四个残差阶段，输出通道基数为 64、128、256、512；
6. Adaptive Average Pooling 到 `1x1`；
7. 全连接层输出 1000 类。

除第一个阶段外，每个新阶段的第一个 block 使用 stride 2 下采样。

## Block 设计

### BasicBlock

供 ResNet-18 和 ResNet-34 使用，扩张倍数为 1：

```text
3x3 Conv -> BN -> ReLU -> 3x3 Conv -> BN
                                 + shortcut
                                 -> ReLU
```

### Bottleneck

供 ResNet-50、101、152 使用，扩张倍数为 4：

```text
1x1 Conv -> BN -> ReLU
3x3 Conv -> BN -> ReLU
1x1 Conv -> BN
           + shortcut
           -> ReLU
```

当 stride 不为 1，或输入通道与 block 输出通道不同，shortcut 使用
`1x1 Conv + BN` projection；否则使用无参数 identity。

阶段切换时遵循原始论文，将 stride 2 放在 Bottleneck 的第一个 `1x1`
卷积。Torchvision 当前实现采用 ResNet v1.5 变体，把 stride 2 放在 `3x3`
卷积；两者参数量相同，但下采样位置不同。Torchvision 在本阶段用于核对配置、
输出形状和参数量，不覆盖论文优先的 stride 位置。

## 五种配置

| 模型 | Block | 四阶段 block 数 |
|---|---|---|
| ResNet-18 | BasicBlock | `[2, 2, 2, 2]` |
| ResNet-34 | BasicBlock | `[3, 4, 6, 3]` |
| ResNet-50 | Bottleneck | `[3, 4, 6, 3]` |
| ResNet-101 | Bottleneck | `[3, 4, 23, 3]` |
| ResNet-152 | Bottleneck | `[3, 8, 36, 3]` |

## 初始化

- Conv2d 使用 Kaiming normal 初始化；
- BatchNorm 的缩放参数初始化为 1，偏置初始化为 0；
- 全连接层使用 PyTorch 的标准初始化；
- 不加载 Torchvision 或其他来源的预训练权重。

## 验证策略

训练进行期间只执行轻量 CPU 验证：

- 五个工厂函数均可创建模型；
- block 类型和每阶段数量正确；
- 默认输出类别数为 1000，并支持自定义 `num_classes`；
- 参数量与同版本 Torchvision 模型一致；
- 当前 CIFAR-10模型注册和测试不受影响；
- 不把模型或输入移动到 CUDA。

CIFAR-10 正式训练完成后，再执行全部模型的完整测试，并验证
`[1, 3, 224, 224] -> [1, 1000]` 前向传播。深层模型的重型前向测试不与正式
GPU训练并行。

参数量参照：

| 模型 | 参数量 |
|---|---:|
| ResNet-18 | 11,689,512 |
| ResNet-34 | 21,797,672 |
| ResNet-50 | 25,557,032 |
| ResNet-101 | 44,549,160 |
| ResNet-152 | 60,192,808 |

## 本阶段只完成

- 手写五种 ImageNet ResNet 结构；
- 编写结构与参数量测试；
- 使用 Torchvision 进行对照验证；
- 编写结构学习说明；
- 将工作保存在独立分支和 Worktree。

## 本阶段暂不进行

- 下载 ImageNet 数据集；
- 训练 ImageNet 模型；
- 下载官方预训练权重；
- 修改 CIFAR-10 正式训练参数；
- 将 ImageNet 分支合并进当前正式训练分支；
- 使用正在执行 CIFAR-10 训练的 GPU。

## 完成标准

1. 五种模型均能被明确的工厂函数创建；
2. 层级配置、block 类型、输出类别数符合论文；
3. 参数量与 Torchvision 对照值一致；
4. 轻量CPU测试通过且不影响CIFAR模型；
5. 学习文档能解释 BasicBlock、Bottleneck、projection shortcut 和五种深度的
   计数方法；
6. 代码与文档提交到 `imagenet-resnet-architectures` 分支，不合并到训练分支。
