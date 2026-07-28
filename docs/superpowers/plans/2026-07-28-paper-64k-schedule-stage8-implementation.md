# 第八阶段：论文 64k Iteration 正式训练 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不破坏现有 5-epoch short 日程的前提下，实现 CIFAR-10 论文严格 64k iteration、32k/48k 学习率衰减、按 epoch 评估和可恢复训练。

**Architecture:** 新建纯逻辑 `schedules.py` 管理 iteration 与学习率边界；`engine.py` 继续负责 batch 数学计算并通过现有回调暴露成功更新；`train.py` 分派 short/paper 两套编排，共用模型、数据、文件、TensorBoard 和 checkpoint 基础设施。测试使用可注入的 7-iteration 微型日程，最终只做缩短 paper 日程的真实 GPU 冒烟。

**Tech Stack:** Python 3.11、PyTorch 2.5.1、torchvision 0.20.1、pytest、TensorBoard、tqdm、PyYAML、Windows、Git 分支 `paper-64k-schedule`。

## Global Constraints

- 开发分支固定为 `paper-64k-schedule`；`main` 的 `08d07ae` 五轮基线不修改。
- paper 默认 `max_iterations=64000`、`milestones=(32000, 48000)`、`learning_rates=(0.1, 0.01, 0.001)`。
- 第 32,000 次更新使用 0.1，第 32,001 次使用 0.01；第 48,000 次使用 0.01，第 48,001 次使用 0.001。
- 完成第 64,000 次 `optimizer.step()` 后停止，禁止第 64,001 次更新。
- 每个完整 epoch 评估一次，最后不足一轮的部分 epoch 也评估一次。
- paper 输出目录为 `seed<seed>_paper64k`，不得覆盖 `seed42_5epochs`。
- short 日程现有 CLI、CSV、TensorBoard、checkpoint 和断点恢复行为保持兼容。
- 不由 Codex 执行正式 64k；只执行微型自动测试和缩短日程 GPU 冒烟。
- 每个任务遵循 RED→GREEN→REFACTOR，并在测试通过后提交到当前分支。

---

## 文件结构

- Create: `schedules.py` — 不依赖训练组件的 paper iteration 日程和值校验。
- Create: `tests/test_schedules.py` — 日程边界、剩余量和每轮 batch 上限测试。
- Create: `tests/test_paper_training.py` — 微型 paper 编排、记录、checkpoint、resume 和中断测试。
- Modify: `train.py` — paper CLI、记录格式、checkpoint 扩展和正式日程编排。
- Modify: `tests/test_train.py` — 现有参数/config/checkpoint 合同兼容性测试。
- Create: `docs/阶段八-64k正式训练操作与学习总结.md` — 用户操作、观察和恢复指南。

### Task 1: 实现纯逻辑 `PaperSchedule`

**Files:**
- Create: `schedules.py`
- Create: `tests/test_schedules.py`

**Interfaces:**
- Produces: `PaperSchedule(max_iterations: int = 64000, milestones: tuple[int, int] = (32000, 48000), learning_rates: tuple[float, float, float] = (0.1, 0.01, 0.001))`
- Produces: `learning_rate_after(completed_iterations: int) -> float`
- Produces: `remaining_iterations(completed_iterations: int) -> int`
- Produces: `max_batches_for_epoch(completed_iterations: int, loader_batches: int) -> int`
- Produces: `is_complete(completed_iterations: int) -> bool`

- [ ] **Step 1: Write failing construction and boundary tests**

```python
def test_default_paper_schedule_matches_cifar10_paper() -> None:
    schedule = PaperSchedule()
    assert schedule.max_iterations == 64000
    assert schedule.milestones == (32000, 48000)
    assert schedule.learning_rates == (0.1, 0.01, 0.001)


@pytest.mark.parametrize(
    ("completed", "expected"),
    [
        (0, 0.1),
        (31999, 0.1),
        (32000, 0.01),
        (47999, 0.01),
        (48000, 0.001),
        (63999, 0.001),
        (64000, 0.001),
    ],
)
def test_learning_rate_after_exact_boundaries(
    completed: int,
    expected: float,
) -> None:
    assert PaperSchedule().learning_rate_after(completed) == expected
```

另写参数化测试拒绝：

```text
max_iterations <= 0
milestones不严格递增
里程碑不在(0,max_iterations)内
learning_rates长度不是3
学习率非正数或非有限数
completed_iterations为bool、非整数、负数或大于max
loader_batches为bool、非整数或不大于0
```

测试微型日程：

```python
schedule = PaperSchedule(7, (3, 5), (0.1, 0.01, 0.001))
assert schedule.remaining_iterations(4) == 3
assert schedule.max_batches_for_epoch(0, 4) == 4
assert schedule.max_batches_for_epoch(4, 4) == 3
assert schedule.is_complete(7) is True
```

- [ ] **Step 2: Run RED**

```powershell
python -m pytest tests/test_schedules.py -q
```

Expected: collection FAIL because `schedules.py` / `PaperSchedule` does not exist.

- [ ] **Step 3: Implement immutable schedule**

```python
@dataclass(frozen=True)
class PaperSchedule:
    max_iterations: int = 64000
    milestones: tuple[int, int] = (32000, 48000)
    learning_rates: tuple[float, float, float] = (0.1, 0.01, 0.001)

    def learning_rate_after(self, completed_iterations: int) -> float:
        self._validate_completed(completed_iterations)
        first, second = self.milestones
        if completed_iterations < first:
            return self.learning_rates[0]
        if completed_iterations < second:
            return self.learning_rates[1]
        return self.learning_rates[2]

    def remaining_iterations(self, completed_iterations: int) -> int:
        self._validate_completed(completed_iterations)
        return self.max_iterations - completed_iterations

    def max_batches_for_epoch(
        self,
        completed_iterations: int,
        loader_batches: int,
    ) -> int:
        self._validate_loader_batches(loader_batches)
        return min(
            loader_batches,
            self.remaining_iterations(completed_iterations),
        )

    def is_complete(self, completed_iterations: int) -> bool:
        self._validate_completed(completed_iterations)
        return completed_iterations == self.max_iterations
```

在 `__post_init__()` 和私有校验函数中实现测试列出的精确异常条件。

- [ ] **Step 4: Run GREEN**

```powershell
python -m pytest tests/test_schedules.py -q
```

Expected: PASS。

- [ ] **Step 5: Commit**

```powershell
git add schedules.py tests/test_schedules.py
git commit -m "feat: add paper iteration schedule"
```

### Task 2: 允许 paper 参数并生成隔离配置

**Files:**
- Modify: `train.py`
- Modify: `tests/test_train.py`
- Test: `tests/test_paper_training.py`

**Interfaces:**
- Consumes: `PaperSchedule`
- Produces: `build_paper_schedule(args: argparse.Namespace) -> PaperSchedule`
- Produces: paper `build_run_name(args) == "seed42_paper64k"`
- Produces: `build_experiment_config()` 中明确的 paper schedule 配置。

- [ ] **Step 1: Write failing CLI/config tests**

```python
def test_validate_args_accepts_default_paper_schedule() -> None:
    parser = argparse.ArgumentParser()
    validate_args(parser, _make_args(schedule="paper"))


def test_paper_run_name_is_isolated_from_short() -> None:
    args = _make_args(schedule="paper", max_iterations=64000)
    assert build_run_name(args) == "seed42_paper64k"


def test_paper_config_records_exact_schedule() -> None:
    args = _make_args(schedule="paper", max_iterations=64000)
    output, log = build_experiment_paths(args)
    config = build_experiment_config(args, output, log)
    assert config["training"]["epochs"] is None
    assert config["training"]["max_iterations"] == 64000
    assert config["training"]["learning_rate_milestones"] == [32000, 48000]
    assert config["training"]["learning_rates"] == [0.1, 0.01, 0.001]
```

另测 paper 模式拒绝 `max_iterations <= 48000`，short 原有 run name 仍为 `seed42_5epochs`。

- [ ] **Step 2: Run RED**

```powershell
python -m pytest tests/test_train.py tests/test_paper_training.py -q
```

Expected: paper 被现有 `parser.error` 拒绝，或新配置字段缺失。

- [ ] **Step 3: Implement paper argument/config resolution**

删除无条件拒绝 paper 的逻辑。实现：

```python
def build_paper_schedule(args: argparse.Namespace) -> PaperSchedule:
    return PaperSchedule(
        max_iterations=args.max_iterations,
        milestones=(32000, 48000),
        learning_rates=(
            args.learning_rate,
            args.learning_rate / 10,
            args.learning_rate / 100,
        ),
    )
```

paper run name：

```python
return f"seed{args.seed}_paper{args.max_iterations // 1000}k"
```

默认 64k 得到 `seed42_paper64k`。配置保存完整 `learning_rates`；帮助文本明确 short/paper 停止条件。

- [ ] **Step 4: Run GREEN and short regression**

```powershell
python -m pytest tests/test_train.py tests/test_stage7_train.py tests/test_paper_training.py -q
```

Expected: PASS。

- [ ] **Step 5: Commit**

```powershell
git add train.py tests/test_train.py tests/test_paper_training.py
git commit -m "feat: configure isolated paper training runs"
```

### Task 3: 扩展 history 与 TensorBoard 的 iteration 记录

**Files:**
- Modify: `train.py`
- Modify: `tests/test_paper_training.py`
- Modify: `tests/test_stage7_train.py`

**Interfaces:**
- Produces: `SHORT_HISTORY_FIELDS`（现有8列）
- Produces: `PAPER_HISTORY_FIELDS`（设计文档中的11列）
- Produces: `write_history(path, history, fields=SHORT_HISTORY_FIELDS)`
- Produces: `write_paper_tensorboard_metrics(writer, row) -> None`
- Produces: `write_paper_learning_rate_trace(writer, schedule) -> None`

- [ ] **Step 1: Write failing paper record tests**

```python
def test_paper_history_has_iteration_and_partial_columns(tmp_path: Path) -> None:
    row = {
        "epoch": 2,
        "iteration": 7,
        "train_batches": 3,
        "is_partial_epoch": True,
        "train_loss": 1.0,
        "train_accuracy": 0.5,
        "test_loss": 0.9,
        "test_accuracy": 0.6,
        "learning_rate": 0.001,
        "elapsed_seconds": 2.0,
        "best_test_accuracy": 0.6,
    }
    path = tmp_path / "history.csv"
    write_history(path, [row], fields=PAPER_HISTORY_FIELDS)
    assert next(csv.DictReader(path.open(encoding="utf-8"))).keys() == {
        *PAPER_HISTORY_FIELDS
    }
```

FakeWriter 断言 paper loss/accuracy/time/epoch 的 step 都为 `iteration=7`，并断言 LearningRate 精确写：

```text
(0, 0.1)
(3, 0.01)
(5, 0.001)
(7, 0.001)
```

保留 short 测试，确认默认 `write_history()` 仍使用原8列且 TensorBoard step 为 epoch。

- [ ] **Step 2: Run RED**

```powershell
python -m pytest tests/test_paper_training.py tests/test_stage7_train.py -q
```

Expected: `PAPER_HISTORY_FIELDS` 或 paper writer 不存在。

- [ ] **Step 3: Implement dual record formats**

将现有 `HISTORY_FIELDS` 重命名/别名为 `SHORT_HISTORY_FIELDS`，保留：

```python
HISTORY_FIELDS = SHORT_HISTORY_FIELDS
```

以兼容已有测试/导入。`write_history()` 接收显式字段列表。paper writer：

```python
def write_paper_tensorboard_metrics(writer: object, row: dict[str, object]) -> None:
    step = int(row["iteration"])
    # 写Loss/Accuracy/Time及Progress/epoch，然后flush
```

学习率 trace 使用 schedule 的默认边界，不逐 batch 写 64k 个点。

- [ ] **Step 4: Run GREEN**

```powershell
python -m pytest tests/test_paper_training.py tests/test_stage7_train.py -q
```

Expected: PASS，且 Windows CSV 不含 `\r\r\n`。

- [ ] **Step 5: Commit**

```powershell
git add train.py tests/test_paper_training.py tests/test_stage7_train.py
git commit -m "feat: record paper metrics by iteration"
```

### Task 4: 扩展 paper checkpoint 与恢复校验

**Files:**
- Modify: `train.py`
- Modify: `tests/test_paper_training.py`
- Modify: `tests/test_stage7_train.py`

**Interfaces:**
- Produces: paper checkpoint fields `global_iteration`, `current_learning_rate`, `schedule_state`
- Produces: `validate_paper_resume_checkpoint(checkpoint, args, config, schedule) -> None`
- Produces: `restore_paper_training_state(...) -> tuple[int, int, float, list[dict[str, object]]]`
- 保持 short `validate_resume_checkpoint()` 与 `restore_training_state()` 可用。

- [ ] **Step 1: Write failing checkpoint tests**

创建 7-iteration schedule checkpoint，断言：

```python
assert checkpoint["global_iteration"] == 4
assert checkpoint["current_learning_rate"] == 0.01
assert checkpoint["schedule_state"] == {
    "max_iterations": 7,
    "milestones": [3, 5],
    "learning_rates": [0.1, 0.01, 0.001],
}
```

参数化篡改并断言拒绝：

```text
global_iteration < 0 或 > max
schedule_state里程碑/学习率/最大值冲突
history最后iteration与checkpoint不一致
optimizer学习率与schedule不一致
model、batch size、seed或优化器参数冲突
```

恢复测试断言返回下一 epoch、global iteration、best 和 history，并恢复模型、optimizer 与 RNG。

- [ ] **Step 2: Run RED**

```powershell
python -m pytest tests/test_paper_training.py -q
```

Expected: paper checkpoint API/字段不存在。

- [ ] **Step 3: Implement versioned paper state**

保持 checkpoint `format_version` 的兼容读取；paper checkpoint 明确加入三个字段。schedule state 从 dataclass 转为普通 JSON/YAML 兼容值。

校验顺序：

```text
结构
→ schedule/config语义
→ iteration/history连续性
→ 当前学习率
→ state dict可加载
```

恢复 optimizer 后将 Tensor 移到目标 device，再恢复 RNG。

- [ ] **Step 4: Run GREEN and short checkpoint regression**

```powershell
python -m pytest tests/test_paper_training.py tests/test_stage7_train.py -q
```

Expected: PASS。

- [ ] **Step 5: Commit**

```powershell
git add train.py tests/test_paper_training.py tests/test_stage7_train.py
git commit -m "feat: resume paper training by global iteration"
```

### Task 5: 实现严格 paper epoch/partial-epoch 编排

**Files:**
- Modify: `train.py`
- Modify: `tests/test_paper_training.py`

**Interfaces:**
- Produces: `run_paper_training(args, *, schedule=None, show_progress=True, writer_factory=SummaryWriter, loader_batch_limit: int | None = None) -> list[dict[str, object]]`
- `run_training()` 按 `args.schedule` 调用 short 或 paper。
- Consumes: `PaperSchedule`、现有 `train_one_epoch()`/`evaluate()`、Tasks 2–4 文件与 checkpoint 接口。

- [ ] **Step 1: Write failing 7-iteration integration test**

注入：

```text
train_loader长度4
PaperSchedule(7, (3,5), ...)
可更新的小模型和SGD
受控train_one_epoch（触发每个batch回调）
受控evaluate
FakeWriter
临时输出目录
```

断言：

```python
assert [row["iteration"] for row in history] == [4, 7]
assert [row["train_batches"] for row in history] == [4, 3]
assert [row["is_partial_epoch"] for row in history] == [False, True]
assert learning_rates_used == [
    0.1, 0.1, 0.1,
    0.01, 0.01,
    0.001, 0.001,
]
assert load_checkpoint(latest)["global_iteration"] == 7
```

测试 evaluate 恰好2次；CSV最后 iteration=7；best/latest存在；TensorBoard指标 step 为4、7；`run_training()` 对 short 仍走原路径。

- [ ] **Step 2: Run RED**

```powershell
python -m pytest tests/test_paper_training.py -q
```

Expected: `run_paper_training` 不存在或 paper 仍被拒绝。

- [ ] **Step 3: Implement paper loop**

每轮开始：

```python
epoch_batches = len(train_loader)
if loader_batch_limit is not None:
    if isinstance(loader_batch_limit, bool) or loader_batch_limit <= 0:
        raise ValueError("loader_batch_limit必须是正整数或None")
    epoch_batches = min(epoch_batches, loader_batch_limit)

max_batches = schedule.max_batches_for_epoch(
    global_iteration,
    epoch_batches,
)
start_iteration = global_iteration
set_optimizer_learning_rate(
    optimizer,
    schedule.learning_rate_after(global_iteration),
)
```

组合 `on_batch_end`：

```python
global_iteration += 1
next_lr = schedule.learning_rate_after(global_iteration)
set_optimizer_learning_rate(optimizer, next_lr)
update_tqdm(...)
```

训练后断言：

```python
global_iteration - start_iteration == train_metrics.batches
global_iteration <= schedule.max_iterations
```

然后评估、记录、原子写入和 checkpoint。完成 64k 后退出。

- [ ] **Step 4: Add resume and Ctrl+C integration tests**

第一段运行至 iteration 4 保存；第二次通过 resume 继续到7，断言不归零、不重复 history 1。模拟第二轮 `KeyboardInterrupt`，断言 latest 仍停在4且 writer close。

- [ ] **Step 5: Run GREEN and all training tests**

```powershell
python -m pytest tests/test_paper_training.py tests/test_stage7_train.py tests/test_train.py tests/test_engine.py -q
```

Expected: PASS。

- [ ] **Step 6: Commit**

```powershell
git add train.py tests/test_paper_training.py
git commit -m "feat: run exact paper 64k training loop"
```

### Task 6: CLI、全量回归与用户文档

**Files:**
- Modify: `train.py`
- Modify: `tests/test_paper_training.py`
- Create: `docs/阶段八-64k正式训练操作与学习总结.md`

**Interfaces:**
- Produces CLI:
  - `python train.py --model plain20 --schedule paper`
  - `python train.py --model resnet20 --schedule paper`

- [ ] **Step 1: Write CLI dispatch test**

monkeypatch `sys.argv` 或 `parse_args()`，断言 paper 命令调用 paper 分支且不读取 `epochs` 作为停止条件；short 命令继续调用 short 分支。

- [ ] **Step 2: Run RED**

```powershell
python -m pytest tests/test_paper_training.py -q
```

Expected: dispatch/帮助文本行为尚未满足。

- [ ] **Step 3: Finish CLI and write learning guide**

文档必须解释：

- iteration 与 epoch 区别；
- 32k/48k 学习率边界；
- 每 epoch 测试约1.9秒；
- 预计一个模型约60～65分钟；
- 两个 Prompt 的 TensorBoard/训练命令；
- 正式输出目录；
- Ctrl+C 与 resume 命令；
- CSV/TensorBoard如何观察学习率下降前后；
- PlainNet-20 完成后才启动 ResNet-20；
- 正式64k不是由Codex代跑。

- [ ] **Step 4: Run full verification**

```powershell
python -m pytest -q
python -m py_compile train.py engine.py data.py schedules.py models/plainnet.py models/resnet.py
git diff --check
```

Expected: 全量测试 PASS、编译退出0、Git diff无空白错误。

- [ ] **Step 5: Commit**

```powershell
git add train.py tests/test_paper_training.py docs/阶段八-64k正式训练操作与学习总结.md
git commit -m "docs: add paper training workflow"
```

### Task 7: 缩短 paper 日程 GPU 冒烟与分支推送

**Files:**
- Verify: `outputs/resnet20/stage8_paper_smoke/`
- Verify: `runs/resnet20/stage8_paper_smoke/`
- No formal 64k output is created.

**Interfaces:**
- Consumes: `run_paper_training()` 可注入 schedule。
- Produces: 真实 GPU/CIFAR-10 的跨里程碑证据。

- [ ] **Step 1: Confirm smoke paths are absent or empty**

只读检查：

```powershell
Get-ChildItem outputs\resnet20\stage8_paper_smoke -Force
Get-ChildItem runs\resnet20\stage8_paper_smoke -Force
```

若非空，不覆盖；改用新的明确 smoke 名称。

- [ ] **Step 2: Run shortened real GPU paper schedule**

使用独立 args/run name 与：

```python
PaperSchedule(
    max_iterations=7,
    milestones=(3, 5),
    learning_rates=(0.1, 0.01, 0.001),
)
```

真实 DataLoader 每轮391 batch。为在冒烟中同时验证完整和 partial 两轮，程序化调用明确传入 `loader_batch_limit=4`，使测试日程形成4 batch完整轮和3 batch部分轮。CLI 不提供这个参数；正式训练时固定为 `None` 并使用真实 DataLoader 长度。

Expected:

- GPU 为 RTX 3060；
- 实际学习率跨过3、5两个节点；
- 总更新精确为7；
- history 行 iteration 为4、7；
- 第二行 partial=true；
- latest/best/config/history/event文件存在。

- [ ] **Step 3: Inspect artifacts**

```powershell
python -c "import pandas as pd; print(pd.read_csv(r'outputs\resnet20\stage8_paper_smoke\history.csv'))"
tensorboard --inspect --logdir runs\resnet20\stage8_paper_smoke
```

checkpoint 断言 `global_iteration == 7`，optimizer 当前学习率 `0.001`。

- [ ] **Step 4: Run fresh final suite**

```powershell
python -m pytest -q
git status --short --branch
git log --oneline --decorate -8
```

Expected: tests PASS；只剩可说明的忽略型 smoke 文件；代码和文档均已提交。

- [ ] **Step 5: Push branch**

```powershell
git push
```

Expected: `paper-64k-schedule` 与 `origin/paper-64k-schedule` 同步。

---

## 完成后的正式训练交接

Codex 完成上述任务后，用户亲自在 Prompt A 运行：

```bat
conda activate resnet-paper
cd /d "E:\文档\resnet复现"
git switch paper-64k-schedule
python train.py --model plain20 --schedule paper
```

Prompt B：

```bat
conda activate resnet-paper
cd /d "E:\文档\resnet复现"
tensorboard --logdir runs
```

PlainNet-20 完成、核对和记录后，再运行：

```bat
python train.py --model resnet20 --schedule paper
```

正式训练结果不提交 `.pt` 或 TensorBoard event；用户确认实验完成后，将 CSV、YAML 和实验总结提交到当前实验分支。
