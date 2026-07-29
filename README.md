# ResNet 论文复现：CIFAR-10 训练、退化实验与 ImageNet 结构

## 项目目标

本项目围绕《Deep Residual Learning for Image Recognition》完成一次学习型复现：

1. 从 PlainNet baseline 理解深层卷积网络；
2. 手写 CIFAR-10 PlainNet 和 ResNet；
3. 走完数据、训练、评估、checkpoint 和日志闭环；
4. 观察 loss 与 accuracy 从初始状态到收敛；
5. 比较 PlainNet-20/56 与 ResNet-20/56；
6. 验证 PlainNet 加深后的退化问题；
7. 实现但不训练 ImageNet ResNet-18/34/50/101/152。

## 已完成范围

### CIFAR-10

- PlainNet-20/32/44/56 和 ResNet-20/32/44/56 结构；
- 5 epoch 快速实验和论文 64000 iteration 正式训练流程；
- PlainNet-20、ResNet-20、PlainNet-56、ResNet-56 正式结果；
- TensorBoard、CSV、best/latest checkpoint；
- 自动结果校验、错误率曲线和论文对比图。

32/44 层只实现结构，未进行正式训练。

### ImageNet

- ResNet-18/34/50/101/152 结构；
- 层数、参数量、结构和前向传播测试。

项目没有下载或训练 ImageNet，也没有下载预训练权重。

## 核心实验结论

| 模型 | 最佳测试错误率 | 最终训练错误率 | 最终测试错误率 |
|---|---:|---:|---:|
| PlainNet-20 | 9.30% | 1.12% | 9.35% |
| ResNet-20 | 8.07% | 0.61% | 8.23% |
| PlainNet-56 | 12.03% | 3.75% | 12.32% |
| ResNet-56 | 6.92% | 0.06% | 7.13% |

- PlainNet 从 20 层到 56 层退化 2.73 个百分点；
- ResNet 从 20 层到 56 层改善 1.15 个百分点；
- 56 层 ResNet 相对 PlainNet 改善 5.11 个百分点；
- PlainNet-56 训练误差也更高，因此不是普通过拟合；
- ResNet-20/56 的 8.07% 和 6.92% 接近论文的 8.75% 和 6.97%。

详细解释见[完整复现报告](docs/ResNet论文复现完整报告.md)。

## 项目目录

```text
resnet-complete复现/
├─ models/
│  ├─ plainnet.py              CIFAR PlainNet
│  ├─ resnet.py                CIFAR ResNet
│  └─ imagenet_resnet.py       ImageNet ResNet
├─ data.py                     CIFAR-10数据与预处理
├─ engine.py                   单个epoch训练和评估
├─ schedules.py                学习率日程
├─ train.py                    训练CLI
├─ analysis/
│  ├─ compare_cifar_results.py 结果分析CLI
│  └─ results/                 汇总CSV和图表
├─ tests/                      自动测试
└─ docs/                       学习总结和最终报告
```

原始正式训练结果保存在：

```text
E:\文档\resnet复现\outputs
```

大型 checkpoint 和 TensorBoard event 不进入 Git。

## 环境准备

推荐环境：

| 项目 | 配置 |
|---|---|
| Python | 3.11 |
| PyTorch | 2.5.1+cu121 |
| Torchvision | 0.20.1+cu121 |
| Conda 环境 | `resnet-paper` |

在 Anaconda Prompt 中：

```bat
conda activate resnet-paper
cd /d "E:\文档\resnet-complete复现"
python --version
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 5 epoch 快速验证

快速实验用于检查训练闭环，不用于和论文正式结果比较：

```bat
python train.py --model plain20 --epochs 5
python train.py --model resnet20 --epochs 5
```

## 64k 正式训练

```bat
python train.py --model plain20 --schedule paper
python train.py --model resnet20 --schedule paper
python train.py --model plain56 --schedule paper
python train.py --model resnet56 --schedule paper
```

默认使用 batch size 128、momentum 0.9、weight decay 0.0001、seed 42 和 64000 iterations。学习率从 0.1 开始，在 32000 和 48000 iterations 降至 0.01 和 0.001。

## TensorBoard 查看训练日志

```bat
tensorboard --logdir "E:\文档\resnet复现\runs"
```

浏览器打开：

```text
http://localhost:6006
```

TensorBoard 只读取日志并画曲线，不会继续训练或修改权重。

## 重新生成结果分析

```bat
python analysis/compare_cifar_results.py ^
  --results-root "E:\文档\resnet复现\outputs" ^
  --output-dir "analysis\results"
```

生成：

```text
analysis\results\cifar_summary.csv
analysis\results\cifar_error_curves.png
analysis\results\paper_comparison.png
```

分析程序只读原始 `config.yaml` 和 `history.csv`，不会修改 checkpoint。

## CIFAR 与 ImageNet 模型入口

### CIFAR

```python
from models import create_model

model = create_model("resnet20", num_classes=10)
```

输入 `[batch, 3, 32, 32]`，输出 `[batch, 10]`。

### ImageNet

文件：`models/imagenet_resnet.py`

```python
from models.imagenet_resnet import resnet18, resnet50

model18 = resnet18(num_classes=1000)
model50 = resnet50(num_classes=1000)
```

输入 `[batch, 3, 224, 224]`，输出 `[batch, 1000]`。这里只完成结构和前向验证，不代表完成 ImageNet 训练。

## 自动测试

```bat
python -m pytest -q
```

Windows 临时目录出现权限问题时，使用新的临时路径：

```bat
python -m pytest -q --basetemp "%TEMP%\resnet-pytest-final"
```

## 文档阅读顺序

1. [阶段四：ResNet 结构](docs/阶段四-ResNet结构实现与学习总结.md)
2. [阶段五：CIFAR-10 数据](docs/阶段五-CIFAR10数据管道学习总结.md)
3. [阶段六：训练与评估](docs/阶段六-训练与评估闭环学习总结.md)
4. [阶段七：训练系统](docs/阶段七-训练系统学习总结.md)
5. [阶段八：64k 正式训练](docs/阶段八-64k正式训练操作与学习总结.md)
6. [阶段九：ImageNet ResNet 结构](docs/阶段九-ImageNet-ResNet结构学习总结.md)
7. [阶段十：正式结果分析](docs/阶段十-CIFAR正式实验结果分析总结.md)
8. [阶段十一：分支整合](docs/阶段十一-完整复现分支整合总结.md)
9. [阶段十三：四模型最终分析](docs/阶段十三-四模型最终结果分析总结.md)
10. [ResNet论文复现完整报告](docs/ResNet论文复现完整报告.md)

阶段十二被取消，因此编号从十一直接到十三。

## Git 分支说明

| 分支 | 用途 |
|---|---|
| `main` | 5 epoch 早期 baseline |
| `paper-64k-schedule` | 正式训练流程 |
| `imagenet-resnet-architectures` | ImageNet 结构 |
| `cifar-results-analysis` | 结果分析 |
| `resnet-complete-reproduction` | 完整整合版本 |

## 未包含内容与结论边界

未完成：

- PlainNet/ResNet-32、44 正式训练；
- 多随机种子实验；
- ResNet-110/1202；
- ImageNet 数据集训练和准确率复现；
- 预训练权重。

本项目支持的结论是：

> 在本次单个随机种子的 CIFAR-10 四模型实验中，PlainNet 从 20 层加深至 56 层出现训练和测试退化，而 residual shortcut 使 56 层 ResNet 更容易优化并获得更低测试错误率。
