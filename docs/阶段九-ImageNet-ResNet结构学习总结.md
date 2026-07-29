# 阶段九：ImageNet ResNet-18/34/50/101/152结构学习总结

## 1. 本阶段完成了什么

本阶段在独立分支中手写了论文版ImageNet网络：

- ResNet-18；
- ResNet-34；
- ResNet-50；
- ResNet-101；
- ResNet-152。

实现文件：

```text
models/imagenet_resnet.py
```

测试文件：

```text
tests/test_imagenet_resnet.py
```

本阶段没有下载ImageNet、没有训练ImageNet，也没有下载预训练权重。重点是理解
网络结构，并证明五个模型能产生正确的层级、参数量和输出形状。

## 2. ImageNet输入和CIFAR-10输入为什么不同

CIFAR-10图片尺寸是：

```text
3×32×32
```

ImageNet通常使用：

```text
3×224×224
```

ImageNet单张图片的像素数量约为CIFAR-10的49倍：

```text
(224×224) / (32×32) = 49
```

因此，ImageNet ResNet首先需要较快地缩小空间尺寸，避免后续卷积计算量过大。

## 3. ImageNet stem

五个模型使用相同的输入端：

```text
输入：3×224×224
→ 7×7 Conv，64通道，stride=2，padding=3
→ BatchNorm
→ ReLU
→ 3×3 MaxPool，stride=2，padding=1
→ 64×56×56
```

第一次卷积把尺寸从224降到112，最大池化再从112降到56。

与CIFAR-10版相比：

| 项目 | CIFAR-10 ResNet | ImageNet ResNet |
|---|---|---|
| 输入 | 32×32 | 224×224 |
| 首层卷积 | 3×3，stride 1 | 7×7，stride 2 |
| 首层通道 | 16 | 64 |
| MaxPool | 无 | 3×3，stride 2 |
| 残差阶段 | 3个 | 4个 |
| 最终分类 | 10类 | 1000类 |

CIFAR图片很小，如果一开始就连续下采样，空间信息会损失得太快，所以CIFAR版
不使用7×7 stride 2和MaxPool。

## 4. 四个残差阶段

stem之后依次通过：

```text
layer1：通道基数64，空间56×56
layer2：通道基数128，空间28×28
layer3：通道基数256，空间14×14
layer4：通道基数512，空间7×7
```

除layer1外，每个新阶段的第一个block使用stride 2进行下采样。

最后：

```text
7×7特征图
→ AdaptiveAvgPool到1×1
→ flatten
→ 全连接层
→ 1000个logits
```

logits是分类分数，不是概率。训练时交叉熵损失会直接接收logits。

## 5. BasicBlock

ResNet-18和ResNet-34使用BasicBlock：

```text
主分支：
x
→ 3×3 Conv
→ BN
→ ReLU
→ 3×3 Conv
→ BN
→ F(x)

shortcut：
x → identity或projection

合并：
ReLU(F(x) + shortcut(x))
```

每个BasicBlock包含两层带权重卷积，因此：

```text
BasicBlock.expansion = 1
```

例如block内部通道基数为64时，最终输出仍然是64通道。

## 6. Bottleneck

ResNet-50、101、152使用Bottleneck：

```text
x
→ 1×1 Conv：调整或降低通道
→ BN
→ ReLU
→ 3×3 Conv：处理空间信息
→ BN
→ ReLU
→ 1×1 Conv：扩张到4倍通道
→ BN
→ 加shortcut
→ ReLU
```

因此：

```text
Bottleneck.expansion = 4
```

当通道基数为64时：

```text
block内部处理通道：64
block最终输出通道：64×4=256
```

四个阶段的实际输出通道为：

```text
256、512、1024、2048
```

1×1卷积让大部分3×3卷积在较窄的通道上计算，从而在增加网络深度时控制计算量。

## 7. expansion的含义

`channels`表示block内部的通道基数，最终输出通道为：

```text
output_channels = channels × expansion
```

| Block | expansion | 基数64时的输出 |
|---|---:|---:|
| BasicBlock | 1 | 64 |
| Bottleneck | 4 | 256 |

expansion不是把图片尺寸扩大，也不是增加block数量，而是决定block最后输出多少
通道。

## 8. Shortcut什么时候需要projection

如果主分支输出形状和输入完全相同，可以直接：

```text
shortcut(x) = x
```

如果出现以下任何情况，就不能直接相加：

- stride为2，空间尺寸减半；
- block输出通道数与输入通道数不同。

这时使用：

```text
1×1 Conv + BatchNorm
```

例如：

```text
输入：64×56×56
输出：128×28×28
```

projection同时用stride 2改变空间尺寸，并用1×1卷积改变通道数，使shortcut与
主分支形状一致。

## 9. ImageNet projection与CIFAR Option A的区别

当前CIFAR-10 ResNet采用论文Option A：

```text
隔点采样 + 通道补零
```

它没有可训练参数。

ImageNet模型采用projection：

```text
1×1 Conv + BN
```

projection包含可训练参数。两套shortcut都解决形状不一致问题，但实现和参数量
不同。

## 10. 五种网络如何配置

| 模型 | Block | layer1 | layer2 | layer3 | layer4 |
|---|---|---:|---:|---:|---:|
| ResNet-18 | BasicBlock | 2 | 2 | 2 | 2 |
| ResNet-34 | BasicBlock | 3 | 4 | 6 | 3 |
| ResNet-50 | Bottleneck | 3 | 4 | 6 | 3 |
| ResNet-101 | Bottleneck | 3 | 4 | 23 | 3 |
| ResNet-152 | Bottleneck | 3 | 8 | 36 | 3 |

列表：

```text
[2,2,2,2]
```

不是卷积层数，而是四个阶段分别包含多少个block。

## 11. 模型层数怎样计算

统计带权重的卷积层和最终全连接层，不统计BN、ReLU、池化和shortcut中的
identity。

### ResNet-18

```text
stem卷积：1层
BasicBlock：8个×2层=16层
FC：1层
总计：1+16+1=18层
```

### ResNet-34

```text
block数量：3+4+6+3=16
总计：1+16×2+1=34层
```

### ResNet-50

```text
block数量：3+4+6+3=16
总计：1+16×3+1=50层
```

### ResNet-101

```text
block数量：3+4+23+3=33
总计：1+33×3+1=101层
```

### ResNet-152

```text
block数量：3+8+36+3=50
总计：1+50×3+1=152层
```

projection shortcut中的1×1卷积通常不计入模型名称所表示的主路径深度。

## 12. 为什么18/34使用BasicBlock

18层和34层的网络深度适中，每个block直接使用两层3×3卷积，结构清晰，计算量
仍可接受。

如果浅层网络也使用Bottleneck，1×1卷积带来的结构复杂度没有明显必要。

## 13. 为什么50/101/152使用Bottleneck

如果152层全部使用宽的3×3卷积，参数量、显存和计算量都会非常大。

Bottleneck让3×3卷积只在较窄通道中处理，再由最后1×1卷积扩张输出通道，因此
可以更经济地构建极深网络。

所以Bottleneck不是为了减少模型名称中的层数，而是为了让更深网络在计算上
可行。

## 14. 原论文与Torchvision的stride差异

本次手写实现优先遵循原始论文：

```text
Bottleneck阶段切换：
第一个1×1 Conv使用stride 2
3×3 Conv使用stride 1
```

当前Torchvision采用常称ResNet v1.5的变体：

```text
第一个1×1 Conv使用stride 1
3×3 Conv使用stride 2
```

两者：

- 输出形状相同；
- block数量相同；
- 参数量相同；
- 下采样发生的位置不同；
- 数值输出和训练结果不会完全相同。

因此Torchvision在本阶段是结构规模和参数量参照，不是逐行复制目标。

## 15. 参数量验证结果

| 模型 | 手写实现参数量 | Torchvision参数量 | 结果 |
|---|---:|---:|---|
| ResNet-18 | 11,689,512 | 11,689,512 | 一致 |
| ResNet-34 | 21,797,672 | 21,797,672 | 一致 |
| ResNet-50 | 25,557,032 | 25,557,032 | 一致 |
| ResNet-101 | 44,549,160 | 44,549,160 | 一致 |
| ResNet-152 | 60,192,808 | 60,192,808 | 一致 |

参数量一致说明没有遗漏主要卷积、projection、BatchNorm或全连接层，但参数量
一致本身不能证明数据流一定正确，所以还需要形状传播测试。

## 16. 不训练ImageNet怎样验证结构

本阶段使用PyTorch meta device：

```python
model = model.to("meta")
x = torch.empty(1, 3, 224, 224, device="meta")
y = model(x)
```

meta tensor只记录：

- shape；
- dtype；
- device语义。

它不保存真实像素，也不执行真实卷积数值计算。因此可以低成本检查五个模型的
完整尺寸传播：

```text
[1,3,224,224] → [1,1000]
```

测试还确认meta验证前后CUDA没有初始化。

## 17. weights=None、随机权重和预训练权重

Torchvision示例：

```python
torchvision.models.resnet50(weights=None)
```

表示：

- 创建网络结构；
- 使用随机初始化；
- 不下载权重；
- 没有ImageNet分类能力；
- 可以从头训练。

如果指定预训练权重枚举，Torchvision会尝试下载官方权重。那是“使用已经训练好
的模型”，不是本阶段的“只构建结构”。

我们的手写工厂函数没有`weights`参数：

```python
from models.imagenet_resnet import resnet50

model = resnet50(num_classes=1000)
print(sum(parameter.numel() for parameter in model.parameters()))
```

它只创建随机初始化结构，不访问网络。

## 18. 为什么不加入现有create_model

现有：

```python
from models import create_model
```

专门服务于CIFAR-10训练CLI，默认类别数是10，输入是32×32。

ImageNet模型默认类别数是1000，输入语义是224×224。如果把两套模型混进同一个
入口，容易误用预处理、分类数和训练日程。

因此本阶段明确使用：

```python
from models.imagenet_resnet import resnet18
```

这让CIFAR和ImageNet结构边界清楚。

## 19. 本阶段测试结果

实施结束时：

```text
ImageNet结构测试：22项通过
CIFAR模型回归测试：88项通过
完整项目测试：261项通过
ResNet-152 meta输出：[1,1000]
CUDA初始化状态：False
```

这些结果证明：

- 五种模型配置正确；
- 参数量与参照一致；
- 224×224到1000类的形状传播正确；
- 原有CIFAR模型没有被破坏；
- 验证过程没有占用GPU。

CIFAR-10正式训练结束后，又使用RTX 3060 Laptop对五个随机初始化模型依次执行
了一次真实前向传播：

| 模型 | 输入 | 输出 | 前向时已分配显存 |
|---|---|---|---:|
| ResNet-18 | `[1,3,224,224]` | `[1,1000]` | 53.4MB |
| ResNet-34 | `[1,3,224,224]` | `[1,1000]` | 92.0MB |
| ResNet-50 | `[1,3,224,224]` | `[1,1000]` | 107.8MB |
| ResNet-101 | `[1,3,224,224]` | `[1,1000]` | 180.2MB |
| ResNet-152 | `[1,3,224,224]` | `[1,1000]` | 239.9MB |

这里的显存是单张图片、`eval()`和`torch.no_grad()`条件下的已分配显存，不代表
ImageNet训练显存。训练还要保存中间激活、梯度和优化器状态，显存会大得多。

## 20. 本阶段学到什么

完成本阶段后，应当能够解释：

1. CIFAR和ImageNet为什么使用不同stem；
2. 一个block只有一次残差相加；
3. BasicBlock和Bottleneck的数据流；
4. expansion为什么影响输出通道；
5. projection shortcut为什么必要；
6. 四阶段block列表如何决定网络深度；
7. 极深网络为什么使用Bottleneck；
8. 参数量相同为什么不代表实现逐行相同；
9. 原论文和Torchvision v1.5的stride位置差异；
10. 不训练模型时如何验证网络结构。

这一阶段的价值不是得到ImageNet准确率，而是亲手建立从论文结构表到可运行
PyTorch模型的对应关系。
