# ResNet 论文复现：CIFAR-10 退化实验与 ImageNet 结构

本项目围绕 He 等人的《Deep Residual Learning for Image Recognition》，独立实现
CIFAR-10 PlainNet/ResNet 和 ImageNet ResNet，并完整走通数据、训练、评估、
checkpoint、TensorBoard 与结果分析流程。

仓库当前为 **Private**。只有仓库所有者和被邀请的协作者登录 GitHub 后才能
克隆。克隆到本地后可以直接验证代码，也可以重新训练；GitHub 网页本身不执行
CUDA 正式训练。

## 已完成内容与结果

- CIFAR PlainNet-20/32/44/56；
- CIFAR ResNet-20/32/44/56，使用论文 Option A shortcut；
- PlainNet-20、ResNet-20、PlainNet-56、ResNet-56 的 64,000 iteration 正式实验；
- ImageNet ResNet-18/34/50/101/152 结构与前向传播验证；
- 自动测试、checkpoint、CSV、TensorBoard 和正式结果分析。

| 模型 | 最佳测试错误率 | 最终训练错误率 | 最终测试错误率 |
|---|---:|---:|---:|
| PlainNet-20 | 9.30% | 1.12% | 9.35% |
| ResNet-20 | 8.07% | 0.61% | 8.23% |
| PlainNet-56 | 12.03% | 3.75% | 12.32% |
| ResNet-56 | 6.92% | 0.06% | 7.13% |

主要观察：

- PlainNet 从20层加深到56层，最佳测试错误率恶化2.73个百分点；
- ResNet 从20层加深到56层，最佳测试错误率改善1.15个百分点；
- 56层ResNet相对同深度PlainNet改善5.11个百分点；
- PlainNet-56的训练错误率也更高，因此这是优化退化证据，不是普通过拟合。

本次只有一个训练seed，不能把略低于论文的数值解释成统计上优于论文。

## 一、从Private仓库克隆

先确认当前GitHub账户拥有仓库权限。任选一种方式。

SSH：

```bat
git clone git@github.com:hhhhuangggg/resnet-cifar10-reproducti.git
cd resnet-cifar10-reproducti
```

HTTPS：

```bat
git clone https://github.com/hhhhuangggg/resnet-cifar10-reproducti.git
cd resnet-cifar10-reproducti
```

HTTPS访问Private仓库时，GitHub可能要求浏览器登录或Personal Access Token；
SSH方式需要先在GitHub账户中添加SSH公钥。

## 二、创建环境

推荐Windows 64位、Python 3.11、NVIDIA GPU和支持CUDA 12.1运行时的驱动。
当前验证环境为RTX 3060 Laptop 6 GB、PyTorch 2.5.1+cu121和
Torchvision 0.20.1+cu121。

### 方式A：使用environment.yml

在Anaconda Prompt中进入仓库根目录：

```bat
conda env create -f environment.yml
conda activate resnet-paper
python --version
```

如果环境已经存在：

```bat
conda activate resnet-paper
conda env update -f environment.yml --prune
```

### 方式B：分步安装GPU版PyTorch

如果Conda在解析带`+cu121`的wheel时失败，使用：

```bat
conda create -n resnet-paper python=3.11 pip -y
conda activate resnet-paper
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 ^
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==1.26.4 pillow==10.4.0 matplotlib==3.9.2 ^
  pandas==2.2.3 tqdm==4.66.5 scikit-learn==1.5.2 ^
  tensorboard==2.18.0 pyyaml==6.0.2 pytest==8.3.3
```

不需要另行安装CUDA Toolkit或cuDNN；PyTorch wheel自带所需CUDA运行库。

## 三、验证环境和模型

检查CUDA：

```bat
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

验证四个CIFAR正式模型：

```bat
python verify_reproduction.py
```

强制使用CPU：

```bat
python verify_reproduction.py --device cpu
```

同时验证ImageNet ResNet-18/34/50/101/152：

```bat
python verify_reproduction.py --device cuda --include-imagenet
```

ImageNet 结构验证不代表完成 ImageNet 训练，也不会下载ImageNet或预训练权重。

运行全部自动测试：

```bat
python -m pytest -q
```

若Windows临时目录出现权限问题：

```bat
mkdir .pytest-tmp
python -m pytest -q --basetemp .pytest-tmp\clean-clone
```

## 四、三个复现层级

### 1 epoch：检查完整链路

```bat
python train.py --model resnet20 --epochs 1 --run-name clean_clone_smoke
```

这一步验证CIFAR-10下载、预处理、GPU训练、测试、checkpoint、CSV和TensorBoard，
不用于论文性能比较。首次运行会自动下载CIFAR-10到`data/`。

### 5 epoch：观察网络开始学习

```bat
python train.py --model plain20 --epochs 5
python train.py --model resnet20 --epochs 5
```

5 epoch可观察loss下降和accuracy上升，但训练太短，不能用于判断深层网络退化。

### 64k paper：重新执行正式实验

```bat
python train.py --model plain20 --schedule paper
python train.py --model resnet20 --schedule paper
python train.py --model plain56 --schedule paper
python train.py --model resnet56 --schedule paper
```

正式日程使用batch size 128、SGD、初始学习率0.1、momentum 0.9、
weight decay 0.0001和seed 42。在32k、48k iteration时学习率分别下降到
0.01、0.001，并严格停止在64k。

不同GPU、驱动、PyTorch版本和随机执行环境可能造成数值差异。论文复现关注的是
公平设置下PlainNet退化和ResNet缓解退化的趋势，不要求checkpoint逐位相同。

## 五、训练输出和TensorBoard

每次训练在以下位置生成文件：

```text
outputs/<model>/<run_name>/
  config.yaml
  history.csv
  best.pt
  latest.pt

runs/<model>/<run_name>/
  events.out.tfevents...
```

启动TensorBoard：

```bat
tensorboard --logdir runs
```

浏览器打开：

```text
http://localhost:6006
```

TensorBoard只读取`runs/`中的日志并绘图，不会继续训练或修改权重。

`data/`、`outputs/`、`runs/`和`*.pt`已被Git忽略，不会随普通提交上传。

## 六、重新分析正式结果

仓库保留已经验证过的轻量结果：

- [四模型汇总CSV](analysis/results/cifar_summary.csv)
- [训练和测试错误率曲线](analysis/results/cifar_error_curves.png)
- [与论文数值对比图](analysis/results/paper_comparison.png)

重新训练四个模型后，可以运行：

```bat
python analysis/compare_cifar_results.py ^
  --results-root outputs ^
  --output-dir analysis\results
```

分析程序会检查四组配置、64k完整性和history字段，再生成CSV和图片。

## 七、模型调用入口

CIFAR：

```python
from models import create_model

model = create_model("resnet20", num_classes=10)
```

输入形状为`[batch, 3, 32, 32]`，输出为`[batch, 10]`。

ImageNet：

```python
from models.imagenet_resnet import resnet18, resnet50

model18 = resnet18(num_classes=1000)
model50 = resnet50(num_classes=1000)
```

输入形状为`[batch, 3, 224, 224]`，输出为`[batch, 1000]`。
完整实现位于`models/imagenet_resnet.py`。

## 八、项目结构

```text
.
├─ models/
│  ├─ plainnet.py
│  ├─ resnet.py
│  └─ imagenet_resnet.py
├─ analysis/
│  ├─ compare_cifar_results.py
│  └─ results/
├─ tests/
├─ data.py
├─ engine.py
├─ schedules.py
├─ train.py
├─ verify_reproduction.py
├─ environment.yml
└─ requirements-lock.txt
```

## 九、当前边界

本仓库尚未完成：

- PlainNet/ResNet-32、44的正式训练；
- 多seed均值和标准差；
- ResNet-110/1202；
- ImageNet训练、top-1/top-5结果和预训练权重；
- 与论文原始框架、硬件和随机环境的严格等价。

当前证据支持的结论是：在本次单seed、统一64k训练日程下，PlainNet从20层
加深到56层后训练和测试表现同时恶化，而residual shortcut使56层网络更容易
优化，并恢复了随深度增加而改善的趋势。
