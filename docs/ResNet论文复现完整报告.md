# ResNet 论文复现完整报告

## 摘要

本项目围绕 ResNet 论文提出的深层网络退化问题，完成了 CIFAR-10 PlainNet/ResNet 的结构、数据、训练、评估、日志和结果分析闭环，并实现了 ImageNet ResNet-18/34/50/101/152 的论文结构。

正式实验比较 PlainNet-20、ResNet-20、PlainNet-56 和 ResNet-56。PlainNet 从 20 层加深到 56 层后，最佳测试错误率由 9.30% 上升到 12.03%，退化 2.73 个百分点；ResNet 从 20 层加深到 56 层后，错误率由 8.07% 下降到 6.92%，改善 1.15 个百分点。在相同 56 层深度和参数量下，ResNet 相对 PlainNet 改善 5.11 个百分点。

PlainNet-56 的最终训练错误率也高于 PlainNet-20，说明观察到的现象不是普通的测试过拟合，而是更深普通网络的优化退化。实验复现了 residual shortcut 缓解深层网络优化困难的核心趋势。

## 1. 复现目标与研究问题

本次复现不以“调用现成模型得到一个准确率”为目标，而是希望回答：

1. PlainNet baseline 是什么？
2. 为什么理论上更深的网络反而可能训练得更差？
3. residual shortcut 改变了什么？
4. 数据如何从图片变成 Tensor 并送入网络？
5. loss、梯度、优化器和学习率如何共同更新权重？
6. 如何记录训练过程并区分最佳结果与最终结果？
7. 本地结果是否复现论文的核心趋势？

项目先用 5 epoch 完整跑通流程，再使用论文式 64000 iteration 日程正式训练，最后从原始日志自动生成结论。

## 2. PlainNet baseline

PlainNet 是没有 residual shortcut 的普通卷积网络。它与同深度 ResNet 使用相同的：

- 输入；
- 卷积主路径；
- stage 数量；
- block 数量；
- 通道变化；
- 分类头；
- 参数初始化；
- 数据和训练方案。

主要差别是 PlainBlock 只计算：

```text
y = F(x)
```

Residual block 计算：

```text
y = F(x) + shortcut(x)
```

因此 PlainNet 是用于判断 shortcut 是否有效的 baseline。如果不建立公平 baseline，就无法知道提升来自 shortcut、更多参数，还是不同训练设置。

## 3. 残差学习与 shortcut

### 3.1 直接映射和残差映射

普通网络让若干层直接逼近期望映射 \(H(x)\)。残差网络把学习目标改写为：

```text
F(x) = H(x) - x
H(x) = F(x) + x
```

如果当前最优行为接近 identity mapping，残差分支只需把 \(F(x)\) 学到接近 0，而 shortcut 可以直接传递 \(x\)。

### 3.2 shortcut 为什么有助于优化

shortcut：

- 为信息提供直接传播路径；
- 为反向梯度提供更短路径；
- 让 block 更容易表示 identity mapping；
- 不要求深层主路径独立保留所有输入信息；
- 缓解网络加深后的优化困难。

### 3.3 每个 block 是否都有 shortcut

本项目中的每个 residual block 都有 shortcut。输入输出形状相同时直接传递 identity；stage 切换导致尺寸或通道变化时，CIFAR 网络使用论文 Option A：

- 空间尺寸下采样；
- 通道使用零填充匹配；
- shortcut 不引入额外可训练卷积参数。

这样同深度 PlainNet 和 ResNet 具有相同参数量，便于公平比较。

## 4. CIFAR-10 网络结构

CIFAR-10 输入尺寸为 32×32，因此使用适合小图像的网络：

```text
输入 3×32×32
  ↓
3×3 Conv，16通道
  ↓
stage1：16通道，32×32
  ↓
stage2：32通道，16×16
  ↓
stage3：64通道，8×8
  ↓
全局平均池化
  ↓
全连接层，10类
```

20/32/44/56 层分别令每个 stage 包含 3/5/7/9 个 block。项目实现了所有结构，但最终只正式训练 20 和 56 层。

同一深度参数量：

| 深度 | PlainNet | ResNet |
|---|---:|---:|
| 20 | 269,722 | 269,722 |
| 56 | 853,018 | 853,018 |

## 5. ImageNet ResNet 结构

项目手写实现了 ResNet-18/34/50/101/152：

| 模型 | block | 每个 stage 的 block 数 |
|---|---|---|
| ResNet-18 | BasicBlock | 2, 2, 2, 2 |
| ResNet-34 | BasicBlock | 3, 4, 6, 3 |
| ResNet-50 | Bottleneck | 3, 4, 6, 3 |
| ResNet-101 | Bottleneck | 3, 4, 23, 3 |
| ResNet-152 | Bottleneck | 3, 8, 36, 3 |

ImageNet 输入通常为 224×224，网络入口为 7×7 stride-2 卷积和最大池化，随后进入 64、128、256、512 基础通道的四个 stage。

BasicBlock 使用两层 3×3 卷积。Bottleneck 使用：

```text
1×1降维 → 3×3处理 → 1×1升维
```

本项目实现论文版 bottleneck stride 位置，同时在文档中说明它与 torchvision 后续常用 v1.5 变体的差别。

这些模型完成结构测试和 `[1, 3, 224, 224] → [1, 1000]` 前向传播，但 ImageNet 未训练，因此没有 ImageNet accuracy 可与论文比较。

## 6. 数据集和预处理

### 6.1 从图片到 Tensor

CIFAR-10 原图像由像素组成。`ToTensor` 将图像转换成 PyTorch Tensor：

```text
[高度, 宽度, 通道] → [通道, 高度, 宽度]
0～255整数 → 0～1浮点数
```

Tensor 使图像可以参与卷积、GPU 运算和自动求导。

### 6.2 训练集处理

训练集使用随机增强：

- 随机裁剪；
- 随机水平翻转；
- 转 Tensor；
- 减去训练集 per-pixel mean。

随机增强让同一图片在不同 epoch 呈现不同局部变化，提高泛化能力。

### 6.3 测试集处理

测试集不使用随机裁剪或翻转，只做确定性 Tensor 转换和相同 mean subtraction。否则每次测试输入不同，指标不稳定，也不利于公平比较。

### 6.4 mean subtraction 不是权重初始化

减去 `mean_image [3, 32, 32]` 是输入预处理：

```text
处理对象：图片像素
目的：让输入以训练集均值为中心
```

Kaiming 初始化处理的是网络参数：

```text
处理对象：卷积权重
目的：控制深层前向激活和反向梯度尺度
```

二者发生在不同对象和不同阶段，不能互相替代。

## 7. 参数初始化

卷积层使用 Kaiming 初始化，适合后接 ReLU 的网络。它根据 fan-out 控制初始权重方差，降低深层网络刚开始时激活或梯度快速放大、缩小的风险。

BatchNorm 初始化为：

```text
weight = 1
bias = 0
```

初始化不会让模型“已经学会分类”。它只提供一个适合优化的起点，真实能力仍来自数据、loss、反向传播和参数更新。

## 8. 损失函数与优化策略

### 8.1 交叉熵

模型输出 10 个类别分数。交叉熵比较这些分数与真实标签，得到一个标量 loss。预测越偏离正确类别，loss 通常越大。

### 8.2 反向传播

每个训练 batch 的核心步骤：

```python
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()
```

- `zero_grad` 清除上个 batch 的梯度；
- forward 计算预测；
- criterion 计算 loss；
- backward 根据链式法则计算梯度；
- optimizer 根据梯度更新权重。

如果只有 forward，没有 `backward()` 和 `step()`，只能测试模型是否能运行，权重仍是初始化状态，不属于训练。

### 8.3 SGD 配置

正式实验使用：

| 配置 | 值 |
|---|---:|
| batch size | 128 |
| learning rate | 0.1 |
| momentum | 0.9 |
| weight decay | 0.0001 |
| seed | 42 |
| max iterations | 64000 |

在 32000 和 48000 iterations 将学习率依次降到 0.01 和 0.001。

## 9. 训练、评估和日志闭环

### 9.1 训练阶段

训练使用 `model.train()`，启用梯度和参数更新。程序累计 batch loss、正确数和样本数，得到 epoch 训练指标。

### 9.2 测试阶段

测试使用：

```python
model.eval()
with torch.no_grad():
    ...
```

测试集不执行反向传播，不更新权重。每个 epoch 测试一次的时间远短于训练，但可以及时发现学习异常并保存最佳模型。

### 9.3 保存内容

每组实验保存：

- `best.pt`：测试准确率最佳时的 checkpoint；
- `latest.pt`：最近一次训练状态；
- `history.csv`：每次评估的指标；
- `config.yaml`：实验配置；
- TensorBoard event：曲线数据。

### 9.4 最佳值和最终值

最佳值是全过程最高测试准确率；最终值是 64000 iteration 结束时的结果。最佳 checkpoint 适合论文对比，最终值适合检查训练结束状态。二者可能不同，因此报告分别记录。

## 10. 正式实验设置

四组实验全部使用：

- CIFAR-10；
- seed 42；
- batch size 128；
- 相同数据增强；
- 相同 SGD 超参数；
- 相同 64000 iteration 日程；
- 相同测试频率；
- 相同结果记录逻辑。

分析程序会验证上述配置；任何实验未到 64000 或配置不一致都会拒绝比较。

## 11. 四模型实验结果

| 模型 | 参数量 | 最佳测试错误率 | 最终训练错误率 | 最终测试错误率 | 时间 |
|---|---:|---:|---:|---:|---:|
| PlainNet-20 | 269,722 | 9.30% | 1.12% | 9.35% | 62.43 分钟 |
| ResNet-20 | 269,722 | 8.07% | 0.61% | 8.23% | 64.00 分钟 |
| PlainNet-56 | 853,018 | 12.03% | 3.75% | 12.32% | 92.44 分钟 |
| ResNet-56 | 853,018 | 6.92% | 0.06% | 7.13% | 101.21 分钟 |

结果文件：

- [汇总 CSV](../analysis/results/cifar_summary.csv)
- [训练与测试错误率曲线](../analysis/results/cifar_error_curves.png)
- [论文对比图](../analysis/results/paper_comparison.png)

## 12. 退化问题分析

### PlainNet 加深

```text
12.03% - 9.30% = 2.73 个百分点
```

PlainNet-56 测试更差，而且最终训练错误率 3.75% 也高于 PlainNet-20 的 1.12%。这不是普通过拟合，而是更深普通网络没有被优化到至少与浅层网络相当的训练解。

### ResNet 加深

```text
8.07% - 6.92% = 1.15 个百分点
```

ResNet-56 训练错误率接近 0，测试表现也优于 ResNet-20，说明 shortcut 让更深网络能够被有效优化。

### 56 层 shortcut 对比

```text
12.03% - 6.92% = 5.11 个百分点
```

在相同深度和参数量下，残差结构显著优于普通结构。这是本次复现最直接的核心证据。

## 13. 与论文结果对比

| 模型 | 论文错误率 | 本次错误率 | 差异 |
|---|---:|---:|---:|
| ResNet-20 | 8.75% | 8.07% | 本次低 0.68 |
| ResNet-56 | 6.97% | 6.92% | 本次低 0.05 |

本次结果与论文接近，并复现了 ResNet 加深后改善的趋势。但本项目只运行单个随机种子，略低于论文不能解释为稳定超过论文。

## 14. 复现范围与限制

本项目未训练：

- PlainNet/ResNet-32 和 44；
- ResNet-110/1202；
- ImageNet；
- 多个随机种子。

因此：

- 可以验证 20/56 层端点的退化与 residual 改善；
- 不能给出完整 20/32/44/56 深度曲线；
- 不能计算均值和标准差；
- 不能报告 ImageNet top-1/top-5 accuracy；
- 不能声称完成论文全部实验。

所有 CIFAR 正式结果来自单个随机种子 seed 42。报告保留这一边界，避免过度解释。

## 15. 学习收获

1. baseline 的意义是提供公平参照，而不是只实现目标模型。
2. Tensor 是神经网络可计算的图像表示，不等于初始化。
3. 数据中心化和 Kaiming 初始化分别处理输入与权重。
4. forward 只能得到预测；训练必须包含梯度和优化器更新。
5. loss 下降说明优化目标在改善，accuracy 说明分类结果变化。
6. 测试集只评估，不参与权重更新。
7. 深层网络问题不能只看测试集，还要结合训练误差判断。
8. shortcut 的关键价值是改善优化路径，不只是增加表达形式。
9. TensorBoard 是日志可视化工具，不是训练程序。
10. Git 分支和 worktree 可以隔离训练、结构开发和分析工作。
11. 论文复现必须区分已完成事实、趋势证据和未覆盖范围。

## 16. 复现命令和产物

### 进入环境

```bat
conda activate resnet-paper
cd /d "E:\文档\resnet-complete复现"
```

### 快速实验

```bat
python train.py --model plain20 --epochs 5
python train.py --model resnet20 --epochs 5
```

### 正式实验

```bat
python train.py --model plain20 --schedule paper
python train.py --model resnet20 --schedule paper
python train.py --model plain56 --schedule paper
python train.py --model resnet56 --schedule paper
```

### 结果分析

```bat
python analysis/compare_cifar_results.py ^
  --results-root "E:\文档\resnet复现\outputs" ^
  --output-dir "analysis\results"
```

### 自动验证

```bat
python -m pytest -q
```

## 结论

本项目从环境配置开始，逐步完成网络结构、数据管道、参数初始化、训练评估闭环、正式实验、日志观察、结果分析和 Git 分支整合。

四模型实验给出一致证据：PlainNet-56 相比 PlainNet-20 在训练集和测试集上都更差，复现了深层普通网络的退化；ResNet-56 相比 ResNet-20 更好，并在相同 56 层条件下显著优于 PlainNet-56，说明 residual shortcut 有效缓解了深层网络的优化困难。

复现没有覆盖论文所有实验，但已经完整回答最初的学习目标：讲清 baseline 与 residual 原理，亲手完成训练全过程，观察 loss 和 accuracy 收敛，并用可复查的数据验证 shortcut 的价值。
