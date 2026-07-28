# 第五阶段：CIFAR-10 数据管道学习总结

## 一、本阶段解决了什么问题

第三、第四阶段已经实现了 PlainNet 和 ResNet，但模型还没有真正接触
图片。第五阶段负责把磁盘中的 CIFAR-10 图片变成模型以后能够接收的
batch Tensor。

完整数据流是：

```text
磁盘中的CIFAR-10
→ Dataset按编号取出单张图片和标签
→ transform增强并转换图片
→ DataLoader组合多张图片
→ images和labels两个batch Tensor
→ 后续阶段送入模型
```

本阶段没有创建模型、没有计算loss、没有反向传播、没有训练。

## 二、本阶段实际创建和修改的内容

| 文件或目录 | 作用 |
|---|---|
| `data.py` | 实现全部数据处理和DataLoader接口 |
| `tests/test_data.py` | 39项数据管道自动测试 |
| `data/cifar-10-batches-py/` | 解压后的真实CIFAR-10 |
| `data/cifar10_per_pixel_mean.pt` | 训练集逐像素平均图像缓存 |
| 本文档 | 解释本阶段全部操作与原理 |

第五阶段完成后，全项目自动测试从90项增加到129项。

## 三、第一步：认识CIFAR-10

CIFAR-10是一个小型彩色图片分类数据集：

| 内容 | 数量或格式 |
|---|---|
| 训练图片 | 50,000张 |
| 测试图片 | 10,000张 |
| 类别 | 10类 |
| 图片大小 | 32×32 |
| 颜色通道 | RGB三通道 |
| 标签编号 | 0～9 |

十个类别为：

```text
0 airplane
1 automobile
2 bird
3 cat
4 deer
5 dog
6 frog
7 horse
8 ship
9 truck
```

训练集用于更新网络参数；测试集只用于评估模型从未参与训练的数据上的
表现。测试集不能参与均值计算，也不能参与参数更新。

## 四、第二步：理解PIL图片与Tensor

Torchvision从CIFAR-10中取出图片时，首先得到PIL图片对象。PIL图片适合：

- 显示；
- 裁剪；
- 翻转；
- 保存；
- 常规图像操作。

神经网络的卷积层不能直接计算PIL对象，所以使用：

```python
transforms.ToTensor()
```

`ToTensor()`完成三件重要事情。

### 1. 改变数据类型

```text
PIL.Image → torch.Tensor
```

Tensor是PyTorch用于数学运算的多维数字容器，支持卷积、GPU和自动求导。

### 2. 改变维度顺序

普通图片通常按：

```text
[高度,宽度,通道]
```

组织。CIFAR-10概念形状是：

```text
[32,32,3]
```

PyTorch卷积使用：

```text
[通道,高度,宽度]
```

因此转换后是：

```text
[3,32,32]
```

### 3. 缩放像素值

8位图片像素通常是整数：

```text
0～255
```

`ToTensor()`除以255并转换为float32：

```text
0～1
```

例如：

| 原像素 | Tensor数值 |
|---:|---:|
| 0 | 0.000 |
| 64 | 约0.251 |
| 128 | 约0.502 |
| 255 | 1.000 |

一张CIFAR图片转换后包含：

```text
3 × 32 × 32 = 3072个浮点数
```

## 五、第三步：计算训练集逐像素平均图像

论文使用逐像素均值减法。它不是只计算RGB三个平均数，而是为每个通道、
每个坐标分别计算平均值。

```text
50,000张 [3,32,32] 训练图片
                 ↓ 沿样本维求平均
1张 [3,32,32] 平均图像
```

主要代码：

```python
total = torch.zeros((3, 32, 32), dtype=torch.float64)

for sample in dataset:
    image, _ = sample
    total += image.to(dtype=torch.float64)

mean_image = (total / len(dataset)).to(dtype=torch.float32)
```

### 为什么逐张累加

如果一次把50,000张图片全部堆叠起来，需要额外保存：

```text
[50000,3,32,32]
```

逐张累加只长期保留一个 `[3,32,32]` 的累加器，更节省内存。

### 为什么累加时使用float64

50,000次浮点求和会产生少量舍入误差。float64比float32精度更高，适合
做大量累加。得到均值后再转成模型通常使用的float32。

### 为什么不能使用增强后的图片

随机裁剪和随机翻转每次结果可能不同。如果用增强图片计算均值，平均图像
也会带有随机性。

正确流程是：

```text
原始训练图片
→ 只做ToTensor
→ 计算固定平均图像
→ 再创建正式训练增强
```

### 为什么不能使用测试集

测试集应该模拟未知数据。用测试集计算平均值会让训练流程提前获得测试
数据的信息，造成数据泄漏。

正确关系：

```text
训练图片 - 训练集平均图像
测试图片 - 训练集平均图像
```

## 六、第四步：校验平均图像

程序不会仅凭文件名相信平均图像正确，而是检查：

```text
必须是Tensor
形状必须是[3,32,32]
必须是浮点数
不能包含NaN
不能包含正负无穷
数值必须在0～1
```

为什么需要这些检查：

- 错误形状会在图片减法时发生广播错误或产生错误结果；
- 整数Tensor不适合中心化计算；
- NaN会传播到网络输出和loss；
- 无穷大会导致训练数值崩溃；
- 均值计算自0～1原图，所以合法均值也应在0～1。

## 七、第五步：缓存平均图像

第一次运行需要遍历50,000张图片。计算完成后使用：

```python
torch.save(mean_image, cache_path)
```

保存到：

```text
data/cifar10_per_pixel_mean.pt
```

以后运行：

```python
torch.load(
    cache_path,
    map_location="cpu",
    weights_only=True,
)
```

即可直接读取。

流程为：

```text
缓存存在
→ 读取
→ 校验
→ 使用

缓存不存在
→ 计算
→ 校验
→ 保存
→ 使用
```

`map_location="cpu"`确保缓存不依赖某张GPU；`weights_only=True`限制
PyTorch反序列化内容，更适合只读取Tensor。

## 八、第六步：实现逐像素均值减法

自定义变换：

```python
class SubtractPerPixelMean:
    def __init__(self, mean_image):
        validate_per_pixel_mean(mean_image)
        self.mean_image = mean_image.detach().clone()

    def __call__(self, image):
        return image - self.mean_image
```

数学表达：

```text
x_centered = x - mean_image
```

例如：

```text
图片像素0.8 - 平均像素0.5 = 0.3
图片像素0.2 - 平均像素0.5 = -0.3
```

减均值后出现负数是正常现象。它表示该像素比训练集对应位置的平均值暗。

### 为什么保存clone

如果直接保存调用者传入的Tensor，外部代码执行：

```python
mean_image.fill_(0)
```

可能偷偷改变变换内部行为。`detach().clone()`建立独立副本，保护变换。

### 它不是模型初始化

逐像素均值处理输入：

```text
x_centered = x - mean_image
```

Kaiming初始化处理网络卷积权重：

```text
W = 合理范围内的随机初始权重
```

模型计算：

```text
y = W * x_centered
```

两者作用对象不同，不能互相替代。

## 九、第七步：训练集数据增强

训练transform：

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

`Compose`表示按照列表顺序依次执行。

### RandomCrop

```text
原始32×32
→ 四周补4像素
→ 40×40
→ 随机选择一个32×32区域
```

同一张物体会产生位置略有不同的版本，使网络减少对固定位置的依赖。

### RandomHorizontalFlip

默认以50%概率左右翻转图片。它增加样本变化，例如一辆朝左的汽车可以
变成朝右。

### 为什么增强不能改变标签

裁剪和水平翻转不会把猫变成汽车，因此标签仍然不变。会改变语义的增强
不能随意加入。

## 十、第八步：测试集固定预处理

测试transform：

```python
transforms.Compose(
    [
        transforms.ToTensor(),
        SubtractPerPixelMean(mean_image),
    ]
)
```

没有随机裁剪和随机翻转。同一张测试图片连续读取两次，应得到完全相同
的Tensor。

测试集像一把固定的尺子。如果尺子每次随机变化，准确率变化可能来自
测试图片变化，而不是模型变化。

## 十一、第九步：创建Dataset

Dataset可以理解为可按编号访问的数据仓库：

```python
len(dataset)
dataset[index]
```

取出一条数据：

```python
image, label = train_dataset[0]
```

会自动执行该Dataset保存的transform。

本项目创建Dataset的顺序：

```text
1. 创建raw_train_dataset，只使用ToTensor
2. 读取或计算mean image
3. 创建train_transform和test_transform
4. 创建正式train_dataset
5. 创建正式test_dataset
```

正式训练集和测试集都使用同一个训练集平均图像。

## 十二、第十步：创建DataLoader

Dataset一次返回一张图片；DataLoader负责把多张图片组合成batch。

```python
images, labels = next(iter(train_loader))
```

默认batch size为128：

```text
128张 [3,32,32] 图片
→ [128,3,32,32]

128个标签
→ [128]
```

### batch四个图片维度

```text
[样本数量,颜色通道,高度,宽度]
```

### shuffle

训练集：

```python
shuffle=True
```

每轮重新打乱样本顺序，减少固定次序对学习的影响。

测试集：

```python
shuffle=False
```

不更新参数，保持顺序固定便于复查结果。

### num_workers

```python
num_workers=0
```

表示由主Python进程准备数据。Windows初期使用0最稳定。它不是GPU数量。
以后流程稳定后可以测试2或4是否更快。

### pin_memory

CUDA可用时默认启用：

```python
pin_memory=True
```

它让CPU到GPU的数据复制通常更高效，但不会自动把Tensor移动到GPU。

后续仍要执行：

```python
images = images.to("cuda")
labels = labels.to("cuda")
```

### drop_last

```python
drop_last=False
```

训练集50,000张，batch size 128：

```text
390个完整batch + 最后80张
```

最后80张仍然保留，不浪费训练数据。

## 十三、自动测试做了什么

`tests/test_data.py`包含39项测试，使用小型内存假图片，不联网。

| 测试类别 | 验证内容 |
|---|---|
| 均值校验 | 类型、形状、dtype、NaN、无穷、范围 |
| 均值计算 | 已知图片平均值、空数据集、错误图片 |
| 缓存 | 首次保存、再次读取、损坏缓存 |
| 减均值 | 数值、非原地修改、独立副本、错误输入 |
| transform | 步骤顺序、输出形状、测试确定性 |
| Dataset | 构造顺序、train标志、共享均值、样本格式 |
| DataLoader | batch、采样器、最后一批、pin_memory、参数校验 |

测试使用假数据，是为了验证我们写的逻辑，不让自动测试受到网络速度和
下载服务器影响。

## 十四、真实数据下载与完整性检查

浏览器下载的压缩包外层MD5与Torchvision记录不同，说明它经过了重新
压缩。没有直接相信或解压使用，而是先在临时目录检查内部文件。

逐一校验：

```text
data_batch_1
data_batch_2
data_batch_3
data_batch_4
data_batch_5
test_batch
batches.meta
```

七个内部文件MD5全部与Torchvision官方记录一致，因此确认数据内容完整
且真实，再解压到项目。临时校验目录随后被清理。

## 十五、真实数据验证结果

真实数据数量：

```text
train: 50000
test: 10000
```

训练batch：

```text
images: [128,3,32,32] torch.float32
labels: [128] torch.int64
label range: 0～9
finite: True
```

测试batch：

```text
images: [128,3,32,32] torch.float32
labels: [128] torch.int64
label range: 0～9
finite: True
```

平均图像：

```text
shape: [3,32,32]
dtype: torch.float32
min: 0.3921375
max: 0.5500738
mean: 0.4733630
```

`finite=True`表示Tensor中没有NaN或无穷大。

## 十六、这一阶段最容易混淆的概念

### Dataset与DataLoader

```text
Dataset：管理单条数据
DataLoader：组织读取顺序并组成batch
```

### ToTensor与减均值

```text
ToTensor：转换类型、维度和像素范围
减均值：让输入以训练集平均图像为中心
```

### 减均值与Kaiming初始化

```text
减均值：处理图片x
Kaiming：初始化权重W
```

### pin_memory与移动到GPU

```text
pin_memory：优化CPU内存传输
.to("cuda")：真正把Tensor放到GPU
```

### 数据增强与增加磁盘图片

数据增强通常在读取时随机执行，不会在磁盘中复制出更多图片。原始训练
集仍是50,000张，但同一图片在不同轮次可能产生不同增强版本。

## 十七、第五阶段与下一阶段的连接

现在可以得到：

```python
train_loader, test_loader = create_cifar10_loaders("data")
```

下一阶段训练循环会从中读取：

```python
for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
```

第五阶段只保证数据正确到达模型入口。下一阶段还需要实现：

- CrossEntropyLoss；
- SGD优化器；
- 训练模式和评估模式；
- 前向传播；
- loss计算；
- 梯度清零；
- 反向传播；
- 参数更新；
- loss和accuracy统计。

## 十八、本阶段结论

第五阶段已经把“磁盘图片”转换成了稳定、可测试、符合论文预处理要求的
PyTorch batch。现在PlainNet和ResNet可以接收完全相同的数据管道，为
后续公平比较训练收敛情况建立了基础。

