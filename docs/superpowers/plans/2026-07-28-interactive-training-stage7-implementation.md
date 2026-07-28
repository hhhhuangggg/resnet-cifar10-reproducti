# 第七阶段：交互式五轮训练系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把现有模型、CIFAR-10 数据管道和训练引擎接入 `train.py`，让用户能够亲自完成 PlainNet-20 与 ResNet-20 的五轮 GPU 训练、实时观察、结果保存和断点续训。

**Architecture:** `engine.py` 只增加与界面无关的 batch 进度回调；`train.py` 负责设备、随机性、实验目录、训练编排、tqdm、TensorBoard、CSV 与 checkpoint。自动测试使用小模型、假数据和假 writer，不执行正式五轮实验；最后只做一个真实 CIFAR-10 GPU batch 的冒烟验证。

**Tech Stack:** Python 3.11、PyTorch 2.5.1、torchvision 0.20.1、tqdm、TensorBoard、PyYAML、pytest、CIFAR-10、Windows/Anaconda。

## Global Constraints

- 正式实验固定先运行 `plain20` 五轮，再运行 `resnet20` 五轮；正式运行由用户亲自在终端启动。
- 两组实验默认参数完全一致：`seed=42`、`batch_size=128`、`lr=0.1`、`momentum=0.9`、`weight_decay=0.0001`、`device=cuda`。
- 本阶段只执行 `schedule=short`；`schedule=paper` 必须明确拒绝。
- 不静默从 CUDA 回退到 CPU。
- 新实验不得覆盖非空输出目录或非空 TensorBoard 日志目录。
- 只在完整 epoch 结束后记录 CSV、TensorBoard 和 checkpoint。
- checkpoint 只加载本项目自行生成且来源可信的文件。
- `latest.pt` 每轮更新；`best.pt` 仅在测试准确率严格提高时更新。
- checkpoint、YAML 和 CSV 均使用同目录临时文件加 `Path.replace()` 原子写入。
- Ctrl+C 不保存未完成 epoch，只保留上一个完整 epoch。
- 当前 `.git` 目录无效，实施时不执行 Git commit；每个任务以测试通过作为检查点。

---

## 文件结构

- Modify: `engine.py` — 提供不可变的 batch 运行指标并在训练/评估时触发可选回调。
- Modify: `train.py` — 完整实验编排入口及其可独立测试的辅助函数。
- Modify: `tests/test_engine.py` — 验证回调次数、内容、运行平均值和默认兼容性。
- Modify: `tests/test_train.py` — 验证设备、随机性、目录、文件记录、checkpoint、resume 和训练编排。
- Create: `docs/阶段七-亲自训练操作与实验记录指南.md` — 用户实际运行和观察 PlainNet/ResNet 的中文指南。

### Task 1: 给训练引擎增加 batch 进度回调

**Files:**
- Modify: `engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Produces: `BatchProgress(batch: int, loss: float, accuracy: float, correct: int, samples: int)`
- Produces: `train_one_epoch(..., max_batches=None, on_batch_end=None) -> EpochMetrics`
- Produces: `evaluate(..., max_batches=None, on_batch_end=None) -> EpochMetrics`

- [ ] **Step 1: 写回调的失败测试**

在 `tests/test_engine.py` 增加：

```python
def test_batch_progress_is_immutable() -> None:
    progress = engine.BatchProgress(1, 1.0, 0.5, 2, 4)
    with pytest.raises(FrozenInstanceError):
        progress.batch = 2


@pytest.mark.parametrize("operation", ["train", "evaluate"])
def test_epoch_operation_reports_running_batch_progress(operation: str) -> None:
    reports: list[engine.BatchProgress] = []
    model = TinyClassifier()
    loader = _make_loader(sample_count=10, batch_size=4)
    criterion = BatchSizeLoss()

    if operation == "train":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        metrics = engine.train_one_epoch(
            model, loader, criterion, optimizer, "cpu",
            on_batch_end=reports.append,
        )
    else:
        metrics = engine.evaluate(
            model, loader, criterion, "cpu",
            on_batch_end=reports.append,
        )

    assert [item.batch for item in reports] == [1, 2, 3]
    assert [item.samples for item in reports] == [4, 8, 10]
    assert reports[-1].loss == pytest.approx(metrics.loss)
    assert reports[-1].accuracy == pytest.approx(metrics.accuracy)
    assert reports[-1].correct == metrics.correct
```

- [ ] **Step 2: 运行测试并确认先失败**

Run:

```powershell
python -m pytest tests/test_engine.py -q
```

Expected: FAIL，原因是 `BatchProgress` 或 `on_batch_end` 尚不存在。

- [ ] **Step 3: 实现最小回调接口**

在 `engine.py`：

```python
from collections.abc import Callable, Iterable


@dataclass(frozen=True)
class BatchProgress:
    batch: int
    loss: float
    accuracy: float
    correct: int
    samples: int
```

给 `train_one_epoch` 和 `evaluate` 增加：

```python
on_batch_end: Callable[[BatchProgress], None] | None = None,
```

每次完成累加后调用：

```python
if on_batch_end is not None:
    on_batch_end(
        BatchProgress(
            batch=batch_count,
            loss=loss_sum / sample_sum,
            accuracy=correct_sum / sample_sum,
            correct=correct_sum,
            samples=sample_sum,
        )
    )
```

- [ ] **Step 4: 验证新旧引擎测试**

Run:

```powershell
python -m pytest tests/test_engine.py -q
```

Expected: PASS；原有不传回调的调用继续通过。

### Task 2: 实现设备、随机种子与安全实验目录

**Files:**
- Modify: `train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Produces: `set_random_seed(seed: int) -> None`
- Produces: `resolve_device(device_name: str) -> torch.device`
- Produces: `prepare_experiment_directories(output_path: Path, log_path: Path, resume: bool) -> None`
- Produces: `atomic_write_text(path: Path, text: str) -> None`

- [ ] **Step 1: 写设备、复现性和目录保护测试**

覆盖以下具体行为：

```python
def test_set_random_seed_repeats_python_numpy_and_torch() -> None:
    set_random_seed(42)
    first = (random.random(), np.random.rand(), torch.rand(2))
    set_random_seed(42)
    second = (random.random(), np.random.rand(), torch.rand(2))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_resolve_cuda_refuses_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_new_run_rejects_nonempty_output_or_log_directory(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    log = tmp_path / "runs"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError, match="run-name|resume"):
        prepare_experiment_directories(output, log, resume=False)
```

另测：CPU 正常返回；空目录允许；非空日志目录拒绝；resume 模式允许已有目录。

- [ ] **Step 2: 运行并确认失败**

Run:

```powershell
python -m pytest tests/test_train.py -q
```

Expected: FAIL，原因是新辅助函数尚不存在。

- [ ] **Step 3: 实现辅助函数并拒绝 paper 日程**

实现：

```python
def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("请求使用CUDA，但当前PyTorch无法使用CUDA")
```

目录函数使用 `directory.iterdir()` 判断非空；新实验检查两个目标后才创建，避免只创建一半。`validate_args()` 对 `schedule == "paper"` 直接 `parser.error("paper 64k iteration日程将在后续正式复现阶段实现")`。

- [ ] **Step 4: 验证任务测试**

Run:

```powershell
python -m pytest tests/test_train.py -q
```

Expected: PASS。

### Task 3: 实现配置、CSV 与 TensorBoard 指标记录

**Files:**
- Modify: `train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Produces: `write_config(path: Path, config: dict[str, object]) -> None`
- Produces: `write_history(path: Path, history: list[dict[str, object]]) -> None`
- Produces: `write_tensorboard_metrics(writer, row: dict[str, object]) -> None`
- Consumes: 现有 `config_to_yaml()` 和 `validate_config_round_trip()`

- [ ] **Step 1: 写文件内容和 writer 调用测试**

固定 CSV 字段：

```python
HISTORY_FIELDS = [
    "epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy",
    "learning_rate", "elapsed_seconds", "best_test_accuracy",
]
```

测试 UTF-8 YAML 可回读、CSV 字段顺序与数值、`.tmp` 不残留，并用记录 `add_scalar()`/`flush()` 调用的 `FakeWriter` 验证六个 tag：

```text
Loss/train
Loss/test
Accuracy/train
Accuracy/test
LearningRate
Time/epoch_seconds
```

- [ ] **Step 2: 运行并确认失败**

Run:

```powershell
python -m pytest tests/test_train.py -q
```

Expected: FAIL，原因是写入函数尚不存在。

- [ ] **Step 3: 实现原子配置和历史记录**

`atomic_write_text()` 先写 `path.with_suffix(path.suffix + ".tmp")`，再 `temp_path.replace(path)`。CSV 使用 `io.StringIO` 与 `csv.DictWriter` 生成完整文本后一次原子写入。TensorBoard 使用 `epoch` 作为 global step，并在六次 `add_scalar()` 后调用 `writer.flush()`。

- [ ] **Step 4: 验证任务测试**

Run:

```powershell
python -m pytest tests/test_train.py -q
```

Expected: PASS。

### Task 4: 实现 checkpoint 保存、校验与恢复

**Files:**
- Modify: `train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Produces: `capture_rng_state() -> dict[str, object]`
- Produces: `restore_rng_state(state: dict[str, object]) -> None`
- Produces: `build_checkpoint(...) -> dict[str, object]`
- Produces: `atomic_save_checkpoint(path: Path, checkpoint: dict[str, object]) -> None`
- Produces: `load_checkpoint(path: Path) -> dict[str, object]`
- Produces: `validate_resume_checkpoint(checkpoint, args, config) -> None`
- Produces: `restore_training_state(checkpoint, model, optimizer, device) -> tuple[int, float, list[dict[str, object]]]`

- [ ] **Step 1: 写 checkpoint 合同测试**

测试字典必须包括：

```python
{
    "format_version", "model_name", "epoch", "model_state",
    "optimizer_state", "best_test_accuracy", "history", "config", "rng_state",
}
```

分别测试：

- 保存后 `.tmp` 不残留；
- `torch.load(..., map_location="cpu", weights_only=False)` 可读；
- 缺字段、错误版本、模型名冲突、配置冲突、非法 epoch、非法 best、history 长度或连续性错误均拒绝；
- 模型与优化器状态恢复；
- optimizer state 中 Tensor 移到目标 device；
- Python/NumPy/PyTorch RNG 保存后能够恢复并继续相同序列。

- [ ] **Step 2: 运行并确认失败**

Run:

```powershell
python -m pytest tests/test_train.py -q
```

Expected: FAIL，原因是 checkpoint 函数尚不存在。

- [ ] **Step 3: 实现 checkpoint**

格式版本固定为 `1`。`atomic_save_checkpoint()` 使用 `<name>.pt.tmp` 和 `Path.replace()`。恢复顺序固定为：

```text
model.load_state_dict
optimizer.load_state_dict
optimizer state Tensor 移到目标 device
restore_rng_state
返回 epoch + 1、best、history
```

配置对比键固定为：model、schedule、epochs、batch size、learning rate、momentum、weight decay、seed、device。恢复前完整验证 checkpoint，不在验证过程中修改模型。

- [ ] **Step 4: 验证 checkpoint 测试**

Run:

```powershell
python -m pytest tests/test_train.py -q
```

Expected: PASS。

### Task 5: 实现单次实验的 epoch 编排

**Files:**
- Modify: `train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Produces: `run_training(args: argparse.Namespace, *, show_progress: bool = True) -> list[dict[str, object]]`
- Consumes: `create_cifar10_loaders()`、`create_model()`、`train_one_epoch()`、`evaluate()` 与 Tasks 1–4 的接口。

- [ ] **Step 1: 写无真实数据的编排测试**

使用 monkeypatch 注入：

- 小型 `nn.Linear` 模型；
- 两个内存 DataLoader；
- 记录调用的假 `train_one_epoch`/`evaluate`；
- `FakeWriter`；
- 临时输出与日志目录。

断言两轮运行时：

```text
train、evaluate 各调用2次
history共有2行且epoch为1、2
config.yaml、history.csv、latest.pt、best.pt存在
latest.pt的epoch为2
best.pt只在测试accuracy严格提高时被替换
TensorBoard六个tag每轮各写一次
```

另写：

- `show_progress=False` 不创建可见进度输出；
- resume 从下一完整 epoch 开始；
- 已完成目标 epoch 时不重复训练；
- 第二轮中断时，只保留第一轮记录且 writer 始终 close；
- 第一轮完成前中断时明确说明没有可恢复 checkpoint。

- [ ] **Step 2: 运行并确认失败**

Run:

```powershell
python -m pytest tests/test_train.py -q
```

Expected: FAIL，原因是 `run_training()` 尚不存在。

- [ ] **Step 3: 实现组件创建与 epoch 循环**

创建顺序：

```text
校验 short 日程
解析设备并设置随机种子
构造路径并保护目录
写 config 或加载并验证 checkpoint
create_cifar10_loaders(root="data", batch_size=..., num_workers=...)
create_model(args.model, num_classes=10).to(device)
CrossEntropyLoss
SGD
SummaryWriter(log_dir=str(log_path))
```

每轮顺序严格为：

```text
计时
train_one_epoch
evaluate
读取当前learning rate
形成history row并更新best
原子写history.csv
写TensorBoard并flush
原子写latest.pt
若严格改善则原子写best.pt
打印epoch总结
```

tqdm 回调通过 `BatchProgress` 更新 `loss` 和百分比 accuracy；`engine.py` 不导入 tqdm。使用 `try/except KeyboardInterrupt/finally` 保证中断提示与 `writer.close()`。

- [ ] **Step 4: 运行编排测试**

Run:

```powershell
python -m pytest tests/test_train.py -q
```

Expected: PASS。

### Task 6: 接通 CLI，并完成全量自动测试

**Files:**
- Modify: `train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Produces: `main() -> None` 调用 `run_training(parse_args())`

- [ ] **Step 1: 写 CLI 入口测试**

通过 monkeypatch `sys.argv` 和 `run_training`，验证：

```text
python train.py --model plain20 --epochs 5
```

会解析为 `model == "plain20"`、`epochs == 5` 并恰好调用一次 `run_training()`；`--schedule paper` 在任何目录创建前失败。

- [ ] **Step 2: 运行并确认失败**

Run:

```powershell
python -m pytest tests/test_train.py -q
```

Expected: FAIL，因为 `main()` 仍只打印配置。

- [ ] **Step 3: 替换展示型 main**

实现：

```python
def main() -> None:
    args = parse_args()
    run_training(args)
```

保留启动时的关键参数、设备、GPU 名称和最终文件路径提示；删除原来只展示配置而不训练的逻辑。

- [ ] **Step 4: 运行本文件与全量测试**

Run:

```powershell
python -m pytest tests/test_train.py tests/test_engine.py -q
python -m pytest -q
```

Expected: 两条命令均 PASS，原有 167 项和新增测试全部通过。

### Task 7: 写用户指南并做真实单 batch 冒烟验证

**Files:**
- Create: `docs/阶段七-亲自训练操作与实验记录指南.md`
- Verify: `outputs/resnet20/stage7_smoke/`
- Verify: `runs/resnet20/stage7_smoke/`

**Interfaces:**
- Consumes: 已完成的 CLI。
- Produces: 用户可逐条执行的训练、TensorBoard、观察、恢复与比较说明。

- [ ] **Step 1: 编写中文操作指南**

文档必须包含：

```bat
conda activate resnet-paper
cd /d "E:\文档\resnet复现"
python -m pytest -q
tensorboard --logdir runs
python train.py --model plain20 --epochs 5
python train.py --model resnet20 --epochs 5
python train.py --model plain20 --epochs 5 --resume "outputs\plain20\seed42_5epochs\latest.pt"
```

解释两个 Prompt 的职责、`http://localhost:6006` 只读取训练日志、终端/tqdm/TensorBoard/CSV/checkpoint 分别观察什么，并提供 PlainNet-20 与 ResNet-20 的逐轮记录表。

- [ ] **Step 2: 为冒烟验证增加受控 batch 限制**

不要把调试参数暴露为正式默认训练行为。给 `run_training()` 增加仅供程序化验证的可选关键字：

```python
max_train_batches: int | None = None
max_test_batches: int | None = None
```

将它们分别传入 `train_one_epoch()` 与 `evaluate()`；CLI 不提供这两个参数。补测试确认限制被正确转发。

- [ ] **Step 3: 运行最终全量测试**

Run:

```powershell
python -m pytest -q
```

Expected: PASS。

- [ ] **Step 4: 执行真实 CIFAR-10 GPU 单 batch 冒烟**

在已激活的 `resnet-paper` 环境内，从 Python 调用 `run_training()`，参数使用：

```text
model=resnet20
epochs=1
run_name=stage7_smoke
device=cuda
max_train_batches=1
max_test_batches=1
```

Expected:

- CUDA 设备名为 `NVIDIA GeForce RTX 3060 Laptop GPU`；
- 训练和评估各完成一个 batch；
- loss 为有限数；
- `outputs/resnet20/stage7_smoke/` 内存在 `config.yaml`、`history.csv`、`latest.pt`、`best.pt`；
- `runs/resnet20/stage7_smoke/` 内存在 TensorBoard event 文件。

- [ ] **Step 5: 检查冒烟产物内容**

Run:

```powershell
Get-ChildItem -Recurse "outputs\resnet20\stage7_smoke"
Get-ChildItem -Recurse "runs\resnet20\stage7_smoke"
python -c "import pandas as pd; print(pd.read_csv(r'outputs\resnet20\stage7_smoke\history.csv'))"
```

Expected: 文件齐全，CSV 只有 epoch 1 一行，所有 loss/accuracy 字段可读取。

---

## 完成后的用户正式操作

实现完成后，Codex 不运行正式五轮实验。用户按以下顺序亲自执行：

1. 在 Prompt A 启动 `python train.py --model plain20 --epochs 5`。
2. 在 Prompt B 启动 `tensorboard --logdir runs`，浏览器打开 `http://localhost:6006`。
3. 观察终端每个 batch 与每个 epoch 的变化，并在 TensorBoard 查看曲线。
4. PlainNet-20 完成后记录最佳准确率、最佳 epoch、总耗时及曲线特点。
5. 启动 `python train.py --model resnet20 --epochs 5`。
6. 使用相同指标对比两者，判断残差连接在短训练中是否已经呈现可学习性优势。
