# 第八阶段设计：CIFAR-10 论文 64k Iteration 正式训练日程

## 1. 阶段目标

第八阶段在已有 5-epoch 训练系统上增加论文 CIFAR-10 正式训练日程：

```text
batch size = 128
initial learning rate = 0.1
momentum = 0.9
weight decay = 0.0001
learning rate / 10 after 32k updates
learning rate / 10 after 48k updates
stop after exactly 64k updates
```

目标是先正式对比 PlainNet-20 与 ResNet-20，再在后续阶段扩展到 32、44、56 层。ResNet-110 所需的约 400-iteration warm-up 不属于本阶段。

本阶段完成代码、测试、缩短日程 GPU 冒烟和用户操作指南，但不由 Codex 运行约一小时的正式 64k 实验。

## 2. 论文依据

论文 CIFAR-10 实验采用：

- weight decay `0.0001`；
- momentum `0.9`；
- mini-batch size `128`；
- 两块 GPU（本复现使用一块 RTX 3060，不改变全局 batch size）；
- 初始学习率 `0.1`；
- 在 32k 和 48k iterations 将学习率除以 10；
- 在 64k iterations 结束；
- 训练时四周填充 4 像素，随机裁剪 32×32，并随机水平翻转；
- 测试时只评估原始 32×32 单视图。

现有 `data.py` 已实现相同的训练增强和测试单视图；本阶段不修改数据预处理。

## 3. Git 与实验隔离

稳定 5-epoch 版本保存在：

```text
main
commit 08d07ae
```

第八阶段只在：

```text
paper-64k-schedule
```

分支开发。

5-epoch 结果目录保持不变：

```text
outputs/plain20/seed42_5epochs
outputs/resnet20/seed42_5epochs
runs/plain20/seed42_5epochs
runs/resnet20/seed42_5epochs
```

正式训练使用独立目录：

```text
outputs/plain20/seed42_paper64k
outputs/resnet20/seed42_paper64k
runs/plain20/seed42_paper64k
runs/resnet20/seed42_paper64k
```

新实验继续拒绝覆盖非空输出或日志目录。

## 4. 严格 Iteration 语义

定义：

```text
global_iteration = 已经成功执行 optimizer.step() 的次数
```

每个训练 batch 的顺序为：

```text
读取batch
→ 前向传播
→ 计算loss
→ 反向传播
→ optimizer.step()
→ global_iteration += 1
→ 必要时设置下一次更新的学习率
```

下一次更新使用的学习率由“已经完成的更新数”决定：

| 已完成更新数 | 下一次更新使用的学习率 |
|---:|---:|
| 0～31,999 | 0.1 |
| 32,000～47,999 | 0.01 |
| 48,000～63,999 | 0.001 |
| 64,000 | 停止，不再执行下一次更新 |

因此：

- 第 32,000 次参数更新使用 `0.1`；
- 第 32,001 次参数更新使用 `0.01`；
- 第 48,000 次参数更新使用 `0.01`；
- 第 48,001 次参数更新使用 `0.001`；
- 第 64,000 次更新完成后立即停止。

训练绝不近似成固定 164 个完整 epoch，也不允许执行第 64,001 次更新。

## 5. Epoch 与最后部分 Epoch

CIFAR-10 训练集有 50,000 张图片，batch size 为 128，且 `drop_last=False`：

```text
ceil(50000 / 128) = 391 batches per epoch
```

预计：

```text
163个完整epoch = 63,733 iterations
最后剩余 = 267 iterations
```

正式日程执行：

```text
163个完整epoch
→ 第164轮只训练267个batch
→ global_iteration达到64000
→ 最后一次完整测试
→ 结束
```

最后一轮记录：

```text
epoch = 164
iteration = 64000
train_batches = 267
is_partial_epoch = true
```

如果未来 DataLoader 长度发生变化，程序根据实际 `len(train_loader)` 和剩余 iterations 动态计算，不写死 391、163 或 267。

## 6. 每 Epoch 评估

用户已选择方案 A：每个 epoch 评估一次。

每次评估：

```text
model.eval()
→ 关闭梯度
→ 遍历完整10,000张测试图片
→ 计算test loss和test accuracy
→ 恢复后续训练流程
```

评估不执行反向传播，不修改权重。

每个完整 epoch 后评估一次；最后部分 epoch 达到 64k 后也评估一次。按当前实测约 1.9 秒一次，整个正式实验预计增加约 5 分钟评估时间。

## 7. 新增 `schedules.py`

新增独立模块：

```text
schedules.py
```

定义不可变 `PaperSchedule`，默认值：

```python
PaperSchedule(
    max_iterations=64000,
    milestones=(32000, 48000),
    learning_rates=(0.1, 0.01, 0.001),
)
```

职责：

- 校验最大 iterations、里程碑和学习率；
- 根据已完成更新数返回下一次更新的学习率；
- 返回剩余更新数；
- 根据 DataLoader 长度返回本轮最大 batch 数；
- 判断日程是否完成。

核心接口：

```python
learning_rate_after(completed_iterations: int) -> float
remaining_iterations(completed_iterations: int) -> int
max_batches_for_epoch(
    completed_iterations: int,
    loader_batches: int,
) -> int
is_complete(completed_iterations: int) -> bool
```

`PaperSchedule` 不依赖模型、DataLoader、优化器、TensorBoard 或文件系统。

测试可注入缩短日程，例如：

```python
PaperSchedule(
    max_iterations=7,
    milestones=(3, 5),
    learning_rates=(0.1, 0.01, 0.001),
)
```

这允许在数秒内验证所有边界，不运行真实 64k。

## 8. `engine.py` 的边界

不重写现有前向传播、loss、反向传播和优化器步骤。

继续使用：

```python
train_one_epoch(
    ...,
    max_batches=本轮允许的batch数,
    on_batch_end=组合回调,
)
```

组合回调在每次成功更新后：

1. 增加 `global_iteration`；
2. 更新 tqdm；
3. 如果达到 32k 或 48k，为下一 batch 设置新学习率；
4. 断言不超过 64k。

`engine.py` 仍不导入 tqdm、TensorBoard 或 `PaperSchedule`。

## 9. `train.py` 的正式日程编排

`run_training()` 根据 `args.schedule` 分派：

```text
short → 保留现有5-epoch行为
paper → 执行严格iteration行为
```

paper 模式循环：

```text
读取或初始化global_iteration
→ schedule计算本轮max_batches
→ 设置本轮第一个batch的正确学习率
→ 训练完整或部分epoch
→ 校验global_iteration增量等于实际batch数
→ 完整测试
→ 形成history row
→ 写CSV和TensorBoard
→ 保存latest及必要时best
→ global_iteration==64000时结束
```

当前阻止 `--schedule paper` 的参数错误将被移除；所有 paper 参数改为严格校验。

## 10. 命令行

正式 PlainNet-20：

```bat
python train.py --model plain20 --schedule paper
```

正式 ResNet-20：

```bat
python train.py --model resnet20 --schedule paper
```

默认：

```text
--max-iterations 64000
--learning-rate 0.1
--batch-size 128
```

`--epochs` 在 paper 模式下不控制停止条件；配置中保存为 `null`。帮助文本明确说明 short 使用 epochs，paper 使用 max iterations。

## 11. History CSV

正式日程 CSV 固定字段：

```text
epoch
iteration
train_batches
is_partial_epoch
train_loss
train_accuracy
test_loss
test_accuracy
learning_rate
elapsed_seconds
best_test_accuracy
```

字段含义：

- `iteration`：本次评估前已完成的全局更新数；
- `train_batches`：本轮实际训练 batch 数；
- `is_partial_epoch`：`train_batches < len(train_loader)`；
- `learning_rate`：下一次更新将使用的学习率；最终 64k 行记录最后阶段学习率 `0.001`。

short 日程保持现有 CSV 结构，避免无意义字段破坏已理解的五轮流程。写入函数根据日程选择固定字段表，并继续使用原子写入。

## 12. TensorBoard

paper 日程的全局 step 使用 `iteration`。

每次评估写入：

```text
Loss/train
Loss/test
Accuracy/train
Accuracy/test
Time/epoch_seconds
Progress/epoch
```

`LearningRate` 单独在以下 step 明确写入：

```text
0     → 0.1
32000 → 0.01
48000 → 0.001
64000 → 0.001
```

这样学习率图能够准确显示阶梯边界，不依赖某个 epoch 是否恰好结束在里程碑。

short 日程继续使用 epoch 作为 global step。

## 13. Checkpoint

paper checkpoint 在现有格式基础上保存：

```text
global_iteration
epoch
history
current_learning_rate
schedule配置
```

保留：

- model state；
- optimizer state；
- best test accuracy；
- config；
- Python、NumPy、CPU PyTorch 和 CUDA RNG；
- format version。

恢复前校验：

- 模型名相同；
- schedule 为 paper；
- max iterations 相同；
- milestones 为 32k/48k；
- batch size、初始学习率、momentum、weight decay、seed 和 device 相同；
- global iteration 位于 0～64k；
- history 连续；
- history 最后一行 iteration 等于 checkpoint global iteration；
- optimizer 当前学习率等于 schedule 对下一次更新计算的值；64k 完成状态允许保持 `0.001`；
- 模型和优化器 state 可加载。

恢复后从下一未完成 epoch 开始，不重复已保存的评估点。

## 14. Ctrl+C

checkpoint 粒度保持为评估点：

- 完整 epoch 结束并评估后保存；
- 最后部分 epoch 完成并评估后保存；
- epoch 中间不保存。

若在 epoch 中按 Ctrl+C：

- 当前未完成 epoch 的内存更新丢弃；
- 磁盘上的上一 checkpoint 不变；
- 最多损失约一轮、当前机器约 20 秒训练；
- 恢复后从上一评估点重新执行该 epoch；
- RNG 从上一 checkpoint 恢复，因此尽量重放相同的数据顺序和增强序列。

如果第一轮完成前中断，则没有 paper checkpoint，需重新开始。

## 15. 输出保护

正式输出：

```text
config.yaml
history.csv
latest.pt
best.pt
```

新实验发现输出或日志目录非空时拒绝覆盖。resume 只沿用 checkpoint 所在正式实验目录。

正式训练文件不会写入或覆盖：

```text
seed42_5epochs
stage7_smoke
```

Git 忽略 `.pt`、数据集和 TensorBoard event 文件；CSV 和 YAML 可按实验总结需要提交。

## 16. 自动测试

新增 `tests/test_schedules.py`：

- 默认 paper 配置正确；
- 非法 max iterations、里程碑和学习率被拒绝；
- completed iterations 的类型和范围被校验；
- 0、31,999、32,000、47,999、48,000、63,999、64,000 边界正确；
- remaining iterations 正确；
- 每轮 max batches 不超过 DataLoader 长度或剩余 iterations；
- 完成状态正确。

扩展 `tests/test_stage7_train.py` 或新增 `tests/test_paper_training.py`：

- 微型 7-iteration 日程恰好停止在 7；
- 学习率在第 3、5 次更新后切换；
- 两轮分别训练 4 和 3 个 batch；
- 最后一轮标记 partial；
- 每轮评估一次；
- CSV 最后一行 iteration 为 7；
- TensorBoard 指标 step 使用 iteration；
- LearningRate 在精确里程碑写入；
- checkpoint 包含 global iteration 和 schedule；
- latest 每次评估更新；
- best 只在测试准确率提高时更新；
- resume 继续 iteration 而不是归零；
- checkpoint 配置和 history 冲突被拒绝；
- Ctrl+C 保留上一评估点；
- paper 与 short 输出目录不冲突；
- short 现有行为和测试不回归。

全量测试必须继续通过。

## 17. GPU 冒烟

实现完成后使用缩短日程执行真实：

- CIFAR-10；
- ResNet-20；
- RTX 3060；
- 少量 iterations；
- 至少跨过两个缩短的学习率节点；
- 至少产生一个完整评估和一个 partial 评估；
- 生成 config、CSV、TensorBoard、latest 和 best；
- 验证最终 iteration 精确等于缩短目标；
- 验证学习率记录正确。

冒烟使用独立名称，不运行真实 64k，不覆盖五轮结果。

## 18. 正式实验顺序

代码和冒烟验证完成后，由用户亲自运行：

```text
PlainNet-20 paper64k
→ 检查最终结果和曲线
→ ResNet-20 paper64k
→ 公平对比
```

预计每个模型约 60～65 分钟，两者合计约两小时。

完成 20 层正式对比后，再决定是否按：

```text
Plain/ResNet-32
Plain/ResNet-44
Plain/ResNet-56
```

依次扩展。110 层 warm-up 和 1202 层实验另行设计。

## 19. 完成标准

第八阶段只有在以下条件全部满足时完成：

- `PaperSchedule` 边界经过自动测试；
- paper 模式执行恰好 64k 更新；
- 第 32,001 和 48,001 次更新使用正确新学习率；
- 每个完整 epoch 评估一次；
- 最后 partial epoch 正确评估和标记；
- CSV、TensorBoard 使用 iteration 记录；
- 学习率阶梯位置精确；
- checkpoint 保存和恢复 global iteration；
- Ctrl+C 行为明确；
- 正式目录不覆盖五轮结果；
- short 日程无回归；
- 全量测试通过；
- 缩短 paper 日程 GPU 冒烟通过；
- 用户操作与学习总结完成；
- 正式 64k 训练由用户亲自启动。
