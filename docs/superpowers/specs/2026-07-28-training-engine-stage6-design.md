# 第六阶段设计：训练与评估核心引擎

## 1. 阶段目标

第六阶段实现第一个完整训练与评估闭环，但只执行少量真实 batch 冒烟
验证，不运行完整 epoch 或5轮预实验。

本阶段完成：

- 将 DataLoader batch 移动到指定设备；
- 执行模型前向传播；
- 使用 CrossEntropyLoss 计算损失；
- 清除旧梯度；
- 反向传播；
- 使用 SGD 更新模型参数；
- 统计按样本加权的平均 loss 和 accuracy；
- 在评估阶段禁用梯度且不修改参数；
- 支持完整 DataLoader 或有限 batch；
- 使用 ResNet-20 和真实 CIFAR-10 在 GPU 上训练5个 batch；
- 在测试集评估3个 batch；
- 编写详细学习总结。

不在本阶段实现：

- 完整 epoch 实验；
- 5轮 PlainNet/ResNet 预实验；
- 学习率调度器；
- CSV；
- TensorBoard；
- checkpoint；
- 断点恢复；
- 模型对比图。

## 2. 代码组织方案

新增独立：

```text
engine.py
```

文件职责：

```text
data.py
→ 创建Dataset和DataLoader

models/
→ 创建PlainNet或ResNet

engine.py
→ 执行训练、评估和指标统计

train.py
→ 后续阶段组织一次完整实验
```

第六阶段不大规模改写现有 `train.py`。先独立验证训练引擎，下一阶段再
将配置、模型、数据、日志和引擎接入命令行入口。

## 3. 文件职责

### `engine.py`

定义：

```text
EpochMetrics
move_batch_to_device()
compute_batch_statistics()
train_one_epoch()
evaluate()
```

不负责：

- 创建模型；
- 创建 Dataset 或 DataLoader；
- 创建 CrossEntropyLoss；
- 创建 SGD；
- 读取命令行参数；
- 保存文件；
- 日志和实验目录。

模型、DataLoader、criterion、optimizer 和 device 均由调用者传入。

### `tests/test_engine.py`

使用小型模型和内存 TensorDataset 验证训练引擎，不读取真实 CIFAR-10，
不依赖网络，不要求CUDA。

### 学习总结

创建：

```text
docs/阶段六-训练与评估闭环学习总结.md
```

详细解释每个操作的目的、顺序、梯度、参数变化、模式切换、指标统计和
真实GPU结果。

## 4. EpochMetrics

定义不可修改的数据类：

```python
@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    correct: int
    samples: int
    batches: int
```

字段含义：

| 字段 | 含义 |
|---|---|
| `loss` | 对所有样本加权的平均loss |
| `accuracy` | `correct / samples`，范围0～1 |
| `correct` | 预测正确的样本数 |
| `samples` | 实际处理的样本数 |
| `batches` | 实际处理的batch数 |

虽然函数支持有限 batch，仍使用 `EpochMetrics`，因为同一结构也适用于
以后完整 epoch，不需要创建另一套统计类型。

## 5. move_batch_to_device

接口：

```python
move_batch_to_device(
    batch: object,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]
```

校验：

- batch 必须是长度为2的tuple或list；
- 第一项和第二项必须都是Tensor；
- images 必须是4维 `(N,3,H,W)`；
- images 的通道维必须为3；
- labels 必须是1维 `(N,)`；
- images和labels的N必须相同；
- N必须大于0；
- images必须是浮点Tensor；
- labels必须为 `torch.int64`；
- device必须能被 `torch.device()`解析。

本函数不强制H和W为32，以便自动测试使用更小图片，也让训练引擎与具体
空间尺寸适度解耦；CIFAR真实数据仍为32×32。

移动：

```python
images = images.to(device, non_blocking=True)
labels = labels.to(device, non_blocking=True)
```

`non_blocking=True`与DataLoader的 `pin_memory=True` 配合时允许更高效的
CPU到GPU复制；它不改变计算正确性。

## 6. compute_batch_statistics

接口：

```python
compute_batch_statistics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[int, int]
```

校验：

- logits必须是二维 `(N,C)`；
- C必须大于1；
- labels必须是一维 `(N,)`；
- N必须大于0；
- logits与labels的N相同；
- logits必须是浮点Tensor；
- labels必须为int64；
- 两者必须位于同一设备；
- logits必须全部为有限数；
- labels必须满足 `0 <= label < C`。

计算：

```python
predictions = logits.argmax(dim=1)
correct = (predictions == labels).sum().item()
samples = labels.size(0)
```

返回Python整数 `(correct, samples)`，不保留计算图。

## 7. max_batches规则

训练和评估都接受：

```python
max_batches: int | None = None
```

含义：

| 值 | 行为 |
|---|---|
| `None` | 遍历完整DataLoader |
| 正整数N | 最多处理N个batch |

拒绝：

- 0；
- 负数；
- 非整数；
- 布尔值。

循环在读取第 `max_batches` 个batch后结束，不多读取一个batch。

如果DataLoader自身少于限制数量，处理其全部数据并正常返回实际batch数。

## 8. train_one_epoch

接口：

```python
train_one_epoch(
    model: nn.Module,
    data_loader: Iterable,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: str | torch.device,
    max_batches: int | None = None,
) -> EpochMetrics
```

开始时：

```python
model.train()
```

每个batch严格执行：

```text
1. move_batch_to_device
2. optimizer.zero_grad(set_to_none=True)
3. logits = model(images)
4. 校验logits形状为(N,C)
5. loss = criterion(logits, labels)
6. 校验loss为单个有限标量
7. loss.backward()
8. optimizer.step()
9. 统计correct和samples
10. 累加loss × 当前batch样本数
```

为什么先清梯度：

PyTorch默认将新梯度累加到旧梯度。如果不清除，不同batch的梯度会意外
叠加。

为什么 `backward()` 在 `step()` 前：

`backward()`计算梯度，`step()`读取这些梯度并更新参数。顺序颠倒时，
当前batch无法用于本次更新。

返回：

```python
EpochMetrics(
    loss=loss_sum / samples,
    accuracy=correct / samples,
    correct=correct,
    samples=samples,
    batches=batches,
)
```

若DataLoader没有产生任何batch，抛出 `ValueError`，不返回除零结果。

## 9. CrossEntropyLoss

调用者创建：

```python
criterion = nn.CrossEntropyLoss()
```

输入：

```text
logits：[N,10]，未经softmax的原始分数
labels：[N]，int64类别编号
```

不在模型forward中手动执行softmax。`CrossEntropyLoss`内部以数值稳定的
方式组合log-softmax与负对数似然。

刚初始化的均匀10分类模型理论参考loss约为：

```text
-ln(1/10) ≈ 2.3026
```

实际随机模型和随机batch不要求精确等于该数值。

## 10. SGD

真实冒烟测试由调用者创建：

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0001,
)
```

含义：

- `lr=0.1`：更新步长；
- `momentum=0.9`：保留历史更新方向，减少来回震荡；
- `weight_decay=0.0001`：对过大权重施加惩罚。

第六阶段不实现学习率里程碑；正式论文日程在后续阶段处理。

## 11. evaluate

接口：

```python
evaluate(
    model: nn.Module,
    data_loader: Iterable,
    criterion: nn.Module,
    device: str | torch.device,
    max_batches: int | None = None,
) -> EpochMetrics
```

开始时：

```python
model.eval()
```

完整循环位于：

```python
with torch.no_grad():
    ...
```

评估执行：

```text
move batch
→ forward
→ loss
→ correct和samples
→ 累加统计
```

评估函数不接收optimizer，也不调用：

```text
zero_grad
backward
step
```

因此不会用测试集更新模型参数。

`model.eval()`控制BatchNorm等模块行为；`torch.no_grad()`控制Autograd。
两者作用不同，必须同时使用。

评估结束后模型保持 `training=False`。下一次训练调用
`train_one_epoch()`时会重新调用 `model.train()`。

## 12. 平均loss

CrossEntropyLoss默认返回当前batch平均loss。考虑最后一个batch可能不足
标准batch size，采用按样本加权：

```python
loss_sum += loss.item() * batch_samples
sample_count += batch_samples
average_loss = loss_sum / sample_count
```

不能简单地：

```python
sum(batch_losses) / batch_count
```

否则小batch会与大batch拥有相同权重。

## 13. accuracy

计算：

```python
predictions = logits.argmax(dim=1)
correct += (predictions == labels).sum().item()
samples += labels.size(0)
accuracy = correct / samples
```

返回0～1：

```text
0.1125表示11.25%
```

函数不乘100，显示层以后再决定使用小数或百分比。

## 14. loss校验

criterion返回值必须：

- 是Tensor；
- 只包含一个元素；
- 是浮点数；
- 是有限数；
- `requires_grad=True`（训练阶段）；
- 评估阶段不要求requires_grad。

若loss为NaN或无穷，立即抛出 `FloatingPointError`，避免继续更新并污染
模型。

模型logits也必须全部有限。

## 15. 自动测试设计

自动测试使用CPU、小型TensorDataset和轻量模型，不读取真实CIFAR-10。

### move_batch_to_device

- 正确返回设备上的Tensor；
- 保留形状和数值；
- 拒绝错误batch结构；
- 拒绝非Tensor；
- 拒绝错误维度、通道、dtype和不匹配N。

### compute_batch_statistics

使用已知logits和labels验证正确数量；
- 验证返回Python整数；
- 拒绝错误维度、样本数、类别范围、设备和非有限logits。

### train_one_epoch

- 调用后模型处于训练模式；
- 只处理 `max_batches`；
- 返回正确samples和batches；
- loss有限；
- accuracy位于0～1；
- 至少一个参数发生变化；
- 每个处理过的batch执行一次optimizer step；
- 使用不同大小batch验证loss按样本加权；
- 空DataLoader被拒绝。

### evaluate

- 调用后模型处于评估模式；
- 只处理 `max_batches`；
- loss和accuracy统计正确；
- 参数评估前后逐项完全相同；
- 不创建新的参数梯度；
- 空DataLoader被拒绝。

### max_batches

- `None`处理全部；
- 正整数限制batch数；
- 0、负数、float和bool被拒绝。

## 16. 真实GPU冒烟测试

配置：

```text
模型：ResNet-20
数据：真实CIFAR-10
设备：NVIDIA GeForce RTX 3060 Laptop GPU
batch_size：128
训练batch：5
评估batch：3
criterion：CrossEntropyLoss
optimizer：SGD
lr：0.1
momentum：0.9
weight_decay：0.0001
```

训练前复制至少一个参数Tensor。训练后验证至少一个参数不再完全相同。

训练结果必须满足：

```text
batches=5
samples=640
loss为有限数
0 <= accuracy <= 1
```

评估前复制全部模型参数。评估后逐项验证完全相同。

评估结果必须满足：

```text
batches=3
samples=384
loss为有限数
0 <= accuracy <= 1
```

不要求5个batch的loss单调下降。不同batch、数据增强和SGD更新会产生正常
波动，本阶段只验证闭环。

## 17. 测试与真实运行范围

自动测试：

- CPU；
- 假数据；
- 速度快；
- 可重复；
- 覆盖错误分支。

真实冒烟：

- GPU；
- ResNet-20；
- 真实CIFAR-10；
- 验证组件组合；
- 不替代完整实验。

两者都需要。只有自动测试无法证明真实GPU组合成功；只有真实冒烟无法
稳定覆盖全部边界条件。

## 18. 实施顺序

1. 建立 `engine.py` 和 `EpochMetrics`；
2. 实现并测试 `move_batch_to_device()`；
3. 实现并测试 `compute_batch_statistics()`；
4. 实现max_batches校验；
5. 实现并测试 `train_one_epoch()`；
6. 验证参数在训练后改变；
7. 实现并测试 `evaluate()`；
8. 验证评估不改变参数；
9. 运行新旧全部自动测试；
10. 使用真实CIFAR-10执行GPU训练5个batch；
11. 使用真实测试集评估3个batch；
12. 编写详细学习总结；
13. 最终复验。

继续遵循：

```text
写失败测试
→ 运行并观察预期失败
→ 添加最小实现
→ 运行并确认通过
→ 进入下一功能
```

## 19. 完成标准

第六阶段只有在以下条件全部满足时完成：

- `engine.py`职责独立；
- batch设备移动正确；
- 训练调用 `model.train()`；
- 每个训练batch先清梯度；
- forward、loss、backward、step顺序正确；
- 训练后至少一个模型参数改变；
- 评估调用 `model.eval()`和`torch.no_grad()`；
- 评估不接收optimizer、不反向传播、不更新参数；
- logits、loss、形状和数值检查生效；
- loss按样本加权；
- accuracy统计正确且位于0～1；
- `max_batches=None`支持完整epoch；
- 正整数max_batches能够限制冒烟范围；
- 空DataLoader和非法参数产生明确错误；
- 新增测试全部通过；
- 现有129项测试继续通过；
- ResNet-20 GPU训练5个batch成功；
- 测试集GPU评估3个batch成功；
- 训练处理640张，评估处理384张；
- 不执行完整epoch或5轮实验；
- 不写CSV、TensorBoard或checkpoint；
- 详细学习总结解释代码、梯度、指标和真实结果。
