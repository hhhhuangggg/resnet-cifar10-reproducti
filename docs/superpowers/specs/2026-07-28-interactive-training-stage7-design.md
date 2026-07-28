# 第七阶段设计：可交互的五轮训练实验系统

## 1. 阶段目标

第七阶段把已经验证的模型、数据管道和训练引擎接入 `train.py`，形成用户
可以在Anaconda Prompt中亲自启动、观察、停止、恢复和查看结果的实验
系统。

本阶段完成：

- 固定随机种子；
- 创建实验目录并防止误覆盖；
- 保存 `config.yaml`；
- 创建真实CIFAR-10 DataLoader；
- 创建PlainNet或ResNet；
- 创建CrossEntropyLoss和SGD；
- 执行短日程epoch循环；
- tqdm实时进度；
- 每轮终端总结；
- 保存 `history.csv`；
- 写TensorBoard标量；
- 保存 `latest.pt` 与 `best.pt`；
- 支持按完整epoch恢复；
- 捕获用户Ctrl+C并说明恢复方式；
- 编写用户训练操作指南。

本阶段不由Codex替用户执行正式五轮PlainNet/ResNet实验。完成代码和自动
验证后，由用户亲自运行：

```text
PlainNet-20五轮
→ 观察并记录
→ ResNet-20五轮
→ 观察并记录
```

## 2. 实验范围

第一组实验：

```text
plain20
epochs=5
seed=42
batch_size=128
lr=0.1
momentum=0.9
weight_decay=0.0001
device=cuda
```

第二组只将模型改为：

```text
resnet20
```

两者参数量均为269,722，使用相同数据、均值图、增强、优化器和随机种子。

第七阶段只正式支持：

```text
--schedule short
```

现有 `paper` 参数定义保留给后续64k iteration正式复现，但在本阶段运行
入口中明确拒绝，避免用户误以为论文调度已经实现。

## 3. 现有组件

### `data.py`

提供：

```python
create_cifar10_loaders(...)
```

### `models`

提供：

```python
create_model(name)
```

### `engine.py`

提供：

```python
train_one_epoch(...)
evaluate(...)
EpochMetrics
```

### `train.py`

目前只负责参数、路径和配置展示。第七阶段将它扩展为实验编排入口，不在
其中重复实现模型结构、数据预处理或梯度公式。

## 4. 文件职责

### `train.py`

负责：

- 命令行参数；
- 实验配置；
- 设备检查；
- 随机种子；
- 输出目录；
- 创建各组件；
- epoch循环；
- tqdm和终端输出；
- CSV、TensorBoard和checkpoint；
- resume；
- KeyboardInterrupt。

不负责：

- 定义PlainNet/ResNet；
- 定义CIFAR transform；
- 重新实现训练和评估数学逻辑。

### `engine.py`

增加可选batch进度回调，但默认行为保持不变：

```python
on_batch_end: Callable[[BatchProgress], None] | None = None
```

它仍然不直接依赖tqdm或TensorBoard。`train.py`用回调把引擎统计送给
tqdm，实现计算与界面解耦。

### `tests/test_train.py`

扩展为实验编排、文件记录、checkpoint和恢复测试。

### `tests/test_engine.py`

增加batch进度回调测试，保证每个已处理batch恰好触发一次。

### 文档

创建：

```text
docs/阶段七-亲自训练操作与实验记录指南.md
```

## 5. BatchProgress

`engine.py`新增：

```python
@dataclass(frozen=True)
class BatchProgress:
    batch: int
    loss: float
    accuracy: float
    correct: int
    samples: int
```

这里的loss和accuracy是从本轮开始到当前batch的运行平均值，不是只看
当前单个batch。

训练和评估每处理完一个batch后，如果回调不为None：

```python
on_batch_end(progress)
```

回调只接收不可修改数据，不接收模型或优化器，不能干预梯度更新。

## 6. tqdm进度条

`train.py`为训练和测试分别创建tqdm：

```text
Epoch 1/5 Train
Epoch 1/5 Test
```

回调更新：

```text
loss
accuracy百分比
```

示例：

```text
Epoch 1/5 Train: 63%|...| 247/391
loss=1.8423 accuracy=32.15%
```

在非交互测试或调用时允许：

```python
show_progress=False
```

避免自动测试输出大量进度条。

## 7. 随机种子

定义：

```python
set_random_seed(seed: int) -> None
```

设置：

```python
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

为提高复现一致性：

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

这可能牺牲少量性能，但当前学习型复现实验优先保证结果可追踪。

DataLoader的随机顺序和数据增强使用PyTorch全局随机状态。checkpoint同时
保存Python、NumPy、CPU PyTorch和CUDA RNG状态，以便按epoch恢复时尽量
延续随机序列。

## 8. 设备解析

定义：

```python
resolve_device(device_name: str) -> torch.device
```

规则：

- `cpu`始终返回CPU；
- `cuda`要求 `torch.cuda.is_available()` 为True；
- CUDA不可用时抛出明确 `RuntimeError`；
- 不静默回退CPU。

程序启动时打印：

```text
device: cuda
gpu: NVIDIA GeForce RTX 3060 Laptop GPU
```

## 9. 实验目录

无resume时：

```text
outputs/<model>/<run_name>/
runs/<model>/<run_name>/
```

例如：

```text
outputs/plain20/seed42_5epochs/
runs/plain20/seed42_5epochs/
```

目标输出目录不存在时创建。

如果输出目录已存在且包含任意文件或子目录，未指定resume时拒绝开始，并
提示：

```text
实验目录已经存在，请更换--run-name或使用--resume
```

空目录允许复用。TensorBoard目录非空时也拒绝新实验，防止曲线混合。

resume时沿用checkpoint所在输出目录和对应日志目录，不创建新run name。

## 10. config.yaml

开始训练前：

```python
validate_config_round_trip(config)
```

然后以UTF-8写入：

```text
output_path/config.yaml
```

使用临时文件原子替换，防止半写文件：

```text
config.yaml.tmp
→ 写完
→ replace config.yaml
```

resume时读取checkpoint中的config，与当前关键参数比较：

- model；
- schedule；
- epochs目标；
- batch size；
- learning rate；
- momentum；
- weight decay；
- seed；
- device。

冲突时拒绝恢复，不悄悄改变实验含义。

## 11. history.csv

固定列：

```text
epoch
train_loss
train_accuracy
test_loss
test_accuracy
learning_rate
elapsed_seconds
best_test_accuracy
```

accuracy在文件中保存0～1小数，显示时乘100。

每轮结束重新根据内存中的完整history原子写入整个CSV。五轮数据很小，
这种方式比追加写更容易保证文件与checkpoint一致。

resume时history来自checkpoint，并验证：

```text
history行数 == 已完成epoch
epoch从1连续递增
```

## 12. TensorBoard

每轮写入：

```text
Loss/train
Loss/test
Accuracy/train
Accuracy/test
LearningRate
Time/epoch_seconds
```

global step使用epoch编号，从1开始。

写入本轮标量后调用：

```python
writer.flush()
```

确保训练期间浏览器能及时看到新曲线。

训练结束或中断时始终：

```python
writer.close()
```

## 13. checkpoint格式

保存字典：

```python
{
    "format_version": 1,
    "model_name": str,
    "epoch": int,
    "model_state": dict,
    "optimizer_state": dict,
    "best_test_accuracy": float,
    "history": list[dict],
    "config": dict,
    "rng_state": {
        "python": object,
        "numpy": tuple,
        "torch_cpu": Tensor,
        "torch_cuda": list[Tensor] | None,
    },
}
```

保存位置：

```text
latest.pt
best.pt
```

使用：

```text
目标.pt.tmp
→ torch.save成功
→ Path.replace目标.pt
```

`latest.pt`每轮覆盖；当本轮test accuracy严格高于历史最佳时，同时更新
`best.pt`。第一轮必然建立best。

## 14. checkpoint恢复

读取时使用：

```python
torch.load(path, map_location="cpu", weights_only=False)
```

之所以不是 `weights_only=True`，是因为checkpoint包含Python和NumPy RNG
状态等非Tensor安全类型。只加载用户自己项目生成的checkpoint，不加载
不可信来源。

校验：

- 顶层必须是dict；
- 必需字段齐全；
- format_version为1；
- model_name与命令行一致；
- epoch为非负整数；
- epoch小于目标epochs，或等于目标时直接说明已完成；
- history合法且长度等于epoch；
- best accuracy位于0～1；
- config关键参数一致；
- state dict和optimizer state能够加载。

恢复顺序：

```text
创建模型和optimizer
→ 加载model state
→ 加载optimizer state
→ 将optimizer Tensor移到目标device
→ 恢复RNG状态
→ 从epoch+1继续
```

## 15. 每轮执行顺序

对于epoch：

```text
记录开始时间
→ 创建训练进度条与回调
→ train_one_epoch
→ 创建测试进度条与回调
→ evaluate
→ 读取当前learning rate
→ 计算耗时
→ 更新best
→ 添加history
→ 写history.csv
→ 写TensorBoard并flush
→ 保存latest
→ 必要时保存best
→ 打印本轮总结
```

CSV、TensorBoard和checkpoint只记录完整epoch。

## 16. 终端每轮总结

打印：

```text
Epoch 1/5
  train_loss: ...
  train_accuracy: ...%
  test_loss: ...
  test_accuracy: ...%
  learning_rate: 0.100000
  elapsed: ...s
  best_test_accuracy: ...%
```

训练结束打印：

```text
训练完成
最佳测试准确率
best.pt位置
latest.pt位置
history.csv位置
TensorBoard日志位置
```

## 17. KeyboardInterrupt

围绕epoch循环捕获：

```python
except KeyboardInterrupt:
```

行为：

- 不保存当前未完成epoch；
- 保留上一完整epoch的latest和history；
- 关闭TensorBoard writer；
- 打印已完成epoch；
- 打印resume命令示例；
- 返回非零退出码或由main显示中断状态。

如果在第一轮完成前中断，没有checkpoint，明确提示需要重新开始。

恢复粒度为完整epoch，不恢复epoch内部batch位置。

## 18. short与paper日程

本阶段epoch循环只实现：

```text
schedule=short
```

五轮内学习率保持0.1，不使用scheduler。

如果用户输入：

```text
--schedule paper
```

程序停止并提示：

```text
paper 64k iteration日程将在后续正式复现阶段实现
```

不使用错误的epoch近似冒充论文日程。

## 19. 自动测试

自动测试使用临时目录、小型模型、假DataLoader和替代SummaryWriter，不
运行真实五轮。

覆盖：

- `set_random_seed()`可重复；
- CUDA不可用明确报错；
- 空实验目录允许；
- 非空输出/日志目录拒绝；
- config原子写入和UTF-8回读；
- CSV列和精度；
- BatchProgress回调次数与运行指标；
- TensorBoard标签和epoch step；
- checkpoint必需字段；
- latest每轮更新；
- best只在准确率提升时更新；
- 临时文件成功替换且不残留；
- checkpoint模型名冲突；
- config冲突；
- history长度/epoch连续性；
- model与optimizer恢复；
- RNG状态保存和恢复；
-epoch循环调用训练与评估；
- KeyboardInterrupt关闭writer并保留完整epoch；
- paper日程被明确拒绝；
- 现有167项测试继续通过。

## 20. 小规模实现验证

代码完成后由Codex执行：

- 全部自动测试；
- 真实CIFAR-10；
- ResNet-20；
- GPU；
- 一个训练batch；
- 一个测试batch；
- 使用独立临时run name；
- 验证config、CSV、TensorBoard、latest和best均生成；
- 随后不删除用户正式实验目录之外的数据。

冒烟输出使用明确名称：

```text
stage7_smoke
```

冒烟完成后可保留在 `outputs/resnet20/stage7_smoke` 供学习检查，正式曲线
使用 `seed42_5epochs`，不会混合。

## 21. 用户操作流程

### 训练前

```bat
conda activate resnet-paper
cd /d "E:\文档\resnet复现"
python -m pytest -q
```

### 启动TensorBoard

第二个Anaconda Prompt：

```bat
conda activate resnet-paper
cd /d "E:\文档\resnet复现"
tensorboard --logdir runs
```

浏览器：

```text
http://localhost:6006
```

### PlainNet-20

第一个Anaconda Prompt：

```bat
python train.py --model plain20 --epochs 5
```

### ResNet-20

PlainNet结束并记录结果后：

```bat
python train.py --model resnet20 --epochs 5
```

### 恢复

```bat
python train.py --model plain20 --epochs 5 --resume "outputs\plain20\seed42_5epochs\latest.pt"
```

## 22. 用户观察清单

训练过程中：

- tqdm batch数持续增加；
- loss为有限数且没有nan；
- 每轮结束出现训练和测试总结；
- TensorBoard逐轮出现新点；
- GPU任务管理器中有负载。

每个模型结束后记录：

- 第1轮训练/测试loss与accuracy；
- 第5轮训练/测试loss与accuracy；
- 最佳test accuracy；
- 最佳epoch；
- 总耗时；
- loss总体趋势；
- accuracy总体趋势；
- 是否有训练accuracy上升但test停滞的过拟合迹象。

## 23. 完成标准

第七阶段只有在以下条件全部满足时完成：

- `train.py`能启动短日程实验；
- 组件使用现有data/models/engine接口；
- tqdm实时显示训练与测试进度；
- 每轮终端总结完整；
- random seed和确定性设置生效；
- 非空实验目录不会被误覆盖；
- config.yaml生成并可读；
- history.csv每轮完整记录；
- TensorBoard标量按epoch写入并flush；
- latest.pt每轮更新；
- best.pt只随最佳准确率更新；
- checkpoint原子保存；
- resume恢复模型、optimizer、history、best和RNG；
- Ctrl+C保留上一完整epoch；
- CUDA不可用时不静默回退；
- paper日程不被错误实现；
- 新旧自动测试全部通过；
- 一个真实GPU训练/评估batch冒烟成功；
- 用户操作指南完成；
- 正式PlainNet-20和ResNet-20五轮由用户亲自运行。
