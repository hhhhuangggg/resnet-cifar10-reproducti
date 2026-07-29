# CIFAR-10 Results Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 自动验证、汇总并可视化四个CIFAR-10正式64k实验，以论文错误率口径分析PlainNet退化和ResNet改善。

**Architecture:** `analysis/compare_cifar_results.py`提供纯函数完成CSV/YAML读取、严格校验、指标汇总和绘图，命令行入口只负责组织路径。单元测试使用临时合成实验，正式集成运行只读访问 `E:\文档\resnet复现\outputs`，生成文件写入分析Worktree。

**Tech Stack:** Python 3.11、标准库csv/dataclasses/argparse/pathlib、PyYAML 6.0.2、Matplotlib 3.9.2、pytest 8.3.3。

## Global Constraints

- 所有代码和生成产物只写入 `E:\文档\resnet-results-analysis` 的 `cifar-results-analysis` 分支。
- 原始结果目录只读；不得修改、移动或删除 `history.csv`、`config.yaml`、`best.pt`、`latest.pt`。
- 输入模型固定为 `plain20`、`resnet20`、`plain56`、`resnet56`。
- 错误率统一为 `100 × (1 - accuracy)`，最佳值与最终值分列保存。
- 公平性校验固定要求：schedule=paper、seed=42、batch_size=128、max_iterations=64000、milestones=[32000,48000]、learning_rates=[0.1,0.01,0.001]。
- 图表直接使用history.csv原始点，不做平滑、插值或删点。
- 论文表格只对照ResNet-20的8.75%和ResNet-56的6.97%，不虚构PlainNet论文精确数值。
- 每项生产代码遵循红-绿TDD并单独提交。

---

## File Map

- Create: `analysis/__init__.py`
  - 使分析代码能够被测试导入。
- Create: `analysis/compare_cifar_results.py`
  - 读取、校验、汇总、绘图和CLI。
- Create: `tests/test_compare_cifar_results.py`
  - 合成实验测试与错误路径测试。
- Create: `analysis/results/cifar_summary.csv`
  - 四模型正式汇总。
- Create: `analysis/results/cifar_error_curves.png`
  - 训练/测试错误率双图。
- Create: `analysis/results/paper_comparison.png`
  - 论文与复现ResNet结果柱状图。
- Create: `docs/阶段十-CIFAR正式实验结果分析总结.md`
  - 数据解释、论文对照、结论边界和日报材料。

---

### Task 1: History CSV读取与严格校验

**Files:**
- Create: `analysis/__init__.py`
- Create: `analysis/compare_cifar_results.py`
- Create: `tests/test_compare_cifar_results.py`

**Interfaces:**
- Produces: `HistoryRow` dataclass
- Produces: `load_history(path: str | Path) -> list[HistoryRow]`
- Guarantees: 缺文件、缺字段、空CSV、iteration非递增、未停在64000或最后不是部分epoch时抛出明确异常。

- [ ] **Step 1: 写history红灯测试**

```python
import csv
from pathlib import Path

import pytest

from analysis.compare_cifar_results import load_history


FIELDS = [
    "epoch", "iteration", "train_batches", "is_partial_epoch",
    "train_loss", "train_accuracy", "test_loss", "test_accuracy",
    "learning_rate", "elapsed_seconds", "best_test_accuracy",
]


def write_history(
    path: Path,
    rows: list[dict[str, object]],
    fields: list[str] = FIELDS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def completed_rows() -> list[dict[str, object]]:
    return [
        {
            "epoch": 1, "iteration": 391, "train_batches": 391,
            "is_partial_epoch": False, "train_loss": 1.5,
            "train_accuracy": 0.4, "test_loss": 1.4,
            "test_accuracy": 0.5, "learning_rate": 0.1,
            "elapsed_seconds": 20.0, "best_test_accuracy": 0.5,
        },
        {
            "epoch": 164, "iteration": 64000, "train_batches": 267,
            "is_partial_epoch": True, "train_loss": 0.1,
            "train_accuracy": 0.98, "test_loss": 0.3,
            "test_accuracy": 0.91, "learning_rate": 0.001,
            "elapsed_seconds": 15.0, "best_test_accuracy": 0.91,
        },
    ]


def test_load_history_parses_completed_run(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    write_history(path, completed_rows())

    rows = load_history(path)

    assert [row.iteration for row in rows] == [391, 64000]
    assert rows[-1].is_partial_epoch is True
    assert rows[-1].test_accuracy == pytest.approx(0.91)


def test_load_history_rejects_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    fields = [field for field in FIELDS if field != "test_accuracy"]
    write_history(path, completed_rows(), fields)

    with pytest.raises(ValueError, match="缺少字段"):
        load_history(path)


def test_load_history_rejects_unfinished_run(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    rows = completed_rows()
    rows[-1]["iteration"] = 63999
    write_history(path, rows)

    with pytest.raises(ValueError, match="64000"):
        load_history(path)
```

- [ ] **Step 2: 运行测试确认因分析模块不存在而失败**

Run:

```bat
E:\Anaconda3\envs\resnet-paper\python.exe -m pytest tests\test_compare_cifar_results.py -q
```

Expected: `No module named 'analysis.compare_cifar_results'`.

- [ ] **Step 3: 实现HistoryRow与load_history**

实现固定字段集合、布尔解析、数值范围和递增校验。关键接口：

```python
@dataclass(frozen=True)
class HistoryRow:
    epoch: int
    iteration: int
    train_batches: int
    is_partial_epoch: bool
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
    learning_rate: float
    elapsed_seconds: float
    best_test_accuracy: float


def load_history(path: str | Path) -> list[HistoryRow]:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"history.csv不存在：{resolved}")
    # 使用csv.DictReader读取；检查REQUIRED_HISTORY_FIELDS。
    # 将每行转换为HistoryRow；accuracy必须在[0,1]。
    # iteration必须严格递增；最后iteration必须为64000；
    # 最后一行is_partial_epoch必须为True。
```

- [ ] **Step 4: 运行history测试确认通过**

Run:

```bat
E:\Anaconda3\envs\resnet-paper\python.exe -m pytest tests\test_compare_cifar_results.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: 提交CSV读取**

```bat
git add analysis\__init__.py analysis\compare_cifar_results.py tests\test_compare_cifar_results.py
git commit -m "feat: validate CIFAR experiment history"
```

---

### Task 2: Config读取与公平比较校验

**Files:**
- Modify: `analysis/compare_cifar_results.py`
- Modify: `tests/test_compare_cifar_results.py`

**Interfaces:**
- Produces: `load_config(path: str | Path) -> dict[str, object]`
- Produces: `validate_experiment_config(config: dict[str, object], expected_model: str) -> None`
- Guarantees: 模型名、schedule、seed、batch size、64k和学习率日程任一不一致均停止。

- [ ] **Step 1: 写配置红灯测试**

```python
import yaml

from analysis.compare_cifar_results import (
    load_config,
    validate_experiment_config,
)


def valid_config(model: str = "plain20") -> dict[str, object]:
    return {
        "model": {"name": model},
        "training": {
            "schedule": "paper",
            "max_iterations": 64000,
            "learning_rate_milestones": [32000, 48000],
            "learning_rates": [0.1, 0.01, 0.001],
            "batch_size": 128,
            "seed": 42,
        },
    }


def test_config_accepts_paper_schedule(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(valid_config(), sort_keys=False),
        encoding="utf-8",
    )

    config = load_config(path)
    validate_experiment_config(config, "plain20")


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("schedule", "short"),
        ("max_iterations", 63999),
        ("batch_size", 64),
        ("seed", 7),
        ("learning_rate_milestones", [30000, 45000]),
        ("learning_rates", [0.1, 0.02, 0.001]),
    ],
)
def test_config_rejects_unfair_setting(
    field: str,
    bad_value: object,
) -> None:
    config = valid_config()
    config["training"][field] = bad_value

    with pytest.raises(ValueError, match=field):
        validate_experiment_config(config, "plain20")
```

- [ ] **Step 2: 运行配置测试确认缺少接口而失败**

Run:

```bat
E:\Anaconda3\envs\resnet-paper\python.exe -m pytest tests\test_compare_cifar_results.py -q
```

Expected: import error for `load_config` or `validate_experiment_config`.

- [ ] **Step 3: 实现配置读取和固定期望值校验**

使用 `yaml.safe_load`；要求根对象、`model`和`training`均为dict。逐字段比较：

```python
EXPECTED_TRAINING = {
    "schedule": "paper",
    "max_iterations": 64000,
    "learning_rate_milestones": [32000, 48000],
    "learning_rates": [0.1, 0.01, 0.001],
    "batch_size": 128,
    "seed": 42,
}
```

异常消息包含具体字段名、期望值和实际值。

- [ ] **Step 4: 运行全部分析测试确认通过**

Run:

```bat
E:\Anaconda3\envs\resnet-paper\python.exe -m pytest tests\test_compare_cifar_results.py -q
```

Expected: all current tests pass.

- [ ] **Step 5: 提交配置校验**

```bat
git add analysis\compare_cifar_results.py tests\test_compare_cifar_results.py
git commit -m "feat: validate comparable CIFAR configs"
```

---

### Task 3: 四模型汇总与summary CSV

**Files:**
- Modify: `analysis/compare_cifar_results.py`
- Modify: `tests/test_compare_cifar_results.py`

**Interfaces:**
- Produces: `ExperimentSummary` dataclass
- Produces: `summarize_experiment(model: str, rows: list[HistoryRow], parameters: int) -> ExperimentSummary`
- Produces: `write_summary_csv(summaries: list[ExperimentSummary], path: str | Path) -> None`
- Uses literal parameter counts: plain20/resnet20=269722，plain56/resnet56=853018。

- [ ] **Step 1: 写汇总红灯测试**

```python
from analysis.compare_cifar_results import (
    summarize_experiment,
    write_summary_csv,
)


def test_summary_keeps_best_and_final_separate(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    rows = completed_rows()
    rows[0]["test_accuracy"] = 0.92
    rows[0]["best_test_accuracy"] = 0.92
    rows[-1]["test_accuracy"] = 0.91
    rows[-1]["best_test_accuracy"] = 0.92
    write_history(path, rows)

    summary = summarize_experiment("plain20", load_history(path), 269722)

    assert summary.best_test_error_percent == pytest.approx(8.0)
    assert summary.final_test_error_percent == pytest.approx(9.0)
    assert summary.final_train_error_percent == pytest.approx(2.0)
    assert summary.best_iteration == 391
    assert summary.total_minutes == pytest.approx(35.0 / 60.0)


def test_write_summary_csv_uses_fixed_header(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    write_history(path, completed_rows())
    summary = summarize_experiment("plain20", load_history(path), 269722)
    output = tmp_path / "summary.csv"

    write_summary_csv([summary], output)

    header = output.read_text(encoding="utf-8").splitlines()[0]
    assert header == (
        "model,parameters,best_epoch,best_iteration,"
        "best_test_accuracy_percent,best_test_error_percent,"
        "final_train_accuracy_percent,final_train_error_percent,"
        "final_test_accuracy_percent,final_test_error_percent,"
        "total_minutes,completed_iterations"
    )
```

- [ ] **Step 2: 运行汇总测试确认缺少接口而失败**

Run:

```bat
E:\Anaconda3\envs\resnet-paper\python.exe -m pytest tests\test_compare_cifar_results.py -q
```

Expected: import error for summary interfaces.

- [ ] **Step 3: 实现汇总和原子CSV写入**

最佳行使用 `max(rows, key=lambda row: row.test_accuracy)`；最终行固定为最后一行。
总时间为所有 `elapsed_seconds` 之和除以60。输出百分数保留6位小数，写入临时
文件后使用 `Path.replace` 原子替换目标文件。

- [ ] **Step 4: 运行分析测试确认通过**

Run:

```bat
E:\Anaconda3\envs\resnet-paper\python.exe -m pytest tests\test_compare_cifar_results.py -q
```

Expected: all current tests pass.

- [ ] **Step 5: 提交汇总逻辑**

```bat
git add analysis\compare_cifar_results.py tests\test_compare_cifar_results.py
git commit -m "feat: summarize CIFAR paper runs"
```

---

### Task 4: 绘图、CLI和正式数据集成运行

**Files:**
- Modify: `analysis/compare_cifar_results.py`
- Modify: `tests/test_compare_cifar_results.py`
- Create: `analysis/results/cifar_summary.csv`
- Create: `analysis/results/cifar_error_curves.png`
- Create: `analysis/results/paper_comparison.png`

**Interfaces:**
- Produces: `plot_error_curves(histories: dict[str, list[HistoryRow]], path: str | Path) -> None`
- Produces: `plot_paper_comparison(summaries: list[ExperimentSummary], path: str | Path) -> None`
- Produces: `run_analysis(results_root: str | Path, output_dir: str | Path) -> list[ExperimentSummary]`
- Produces CLI: `--results-root` and `--output-dir`

- [ ] **Step 1: 写端到端合成实验红灯测试**

```python
from analysis.compare_cifar_results import run_analysis


def write_experiment(root: Path, model: str) -> None:
    experiment = root / model / "seed42_paper64k"
    write_history(experiment / "history.csv", completed_rows())
    (experiment / "config.yaml").write_text(
        yaml.safe_dump(valid_config(model), sort_keys=False),
        encoding="utf-8",
    )


def test_run_analysis_creates_all_artifacts(tmp_path: Path) -> None:
    results_root = tmp_path / "outputs"
    for model in ("plain20", "resnet20", "plain56", "resnet56"):
        write_experiment(results_root, model)
    output_dir = tmp_path / "analysis-results"

    summaries = run_analysis(results_root, output_dir)

    assert [summary.model for summary in summaries] == [
        "plain20", "resnet20", "plain56", "resnet56"
    ]
    for filename in (
        "cifar_summary.csv",
        "cifar_error_curves.png",
        "paper_comparison.png",
    ):
        artifact = output_dir / filename
        assert artifact.is_file()
        assert artifact.stat().st_size > 0
```

- [ ] **Step 2: 运行端到端测试确认缺少run_analysis而失败**

Run:

```bat
E:\Anaconda3\envs\resnet-paper\python.exe -m pytest tests\test_compare_cifar_results.py::test_run_analysis_creates_all_artifacts -q
```

Expected: import error for `run_analysis`.

- [ ] **Step 3: 实现绘图和CLI**

固定设置：

```python
MODELS = ("plain20", "resnet20", "plain56", "resnet56")
PARAMETERS = {
    "plain20": 269_722,
    "resnet20": 269_722,
    "plain56": 853_018,
    "resnet56": 853_018,
}
PAPER_TEST_ERRORS = {"resnet20": 8.75, "resnet56": 6.97}
COLORS = {
    "plain20": "#0072B2",
    "resnet20": "#009E73",
    "plain56": "#D55E00",
    "resnet56": "#CC79A7",
}
```

曲线图为12×8英寸、160 DPI，上图训练错误率、下图测试错误率；在32000和48000
画灰色虚线，在64000画黑色点线。柱状图只包含ResNet-20/56的论文值和复现最佳
值，并为每根柱标注两位小数。

模块导入顺序必须先设置无界面后端，再导入pyplot：

```python
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
```

CLI：

```python
def main() -> None:
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/results"),
    )
    args = parser.parse_args()
    summaries = run_analysis(args.results_root, args.output_dir)
    for summary in summaries:
        print(...)
```

- [ ] **Step 4: 运行全部分析测试**

Run:

```bat
E:\Anaconda3\envs\resnet-paper\python.exe -m pytest tests\test_compare_cifar_results.py -q
```

Expected: all analysis tests pass.

- [ ] **Step 5: 用正式数据生成结果**

Run:

```bat
E:\Anaconda3\envs\resnet-paper\python.exe analysis\compare_cifar_results.py --results-root "E:\文档\resnet复现\outputs" --output-dir "analysis\results"
```

Expected console values:

```text
plain20 best_test_error=9.30%
resnet20 best_test_error=8.07%
plain56 best_test_error=12.03%
resnet56 best_test_error=6.92%
```

- [ ] **Step 6: 视觉检查两张PNG**

使用本地图像查看工具打开：

```text
analysis/results/cifar_error_curves.png
analysis/results/paper_comparison.png
```

确认文字不裁切、图例清晰、四条线可区分、32k/48k标记可见、柱顶数值正确。

- [ ] **Step 7: 提交脚本、测试和正式分析产物**

```bat
git add analysis tests\test_compare_cifar_results.py
git commit -m "feat: analyze CIFAR paper results"
```

---

### Task 5: 阶段十学习总结与最终验证

**Files:**
- Create: `docs/阶段十-CIFAR正式实验结果分析总结.md`

**Interfaces:**
- Consumes: 正式summary和两张图。
- Produces: 可直接用于日报和复现报告的中文分析。

- [ ] **Step 1: 写入结果表和关键差值**

文档必须记录：

```text
PlainNet-20最佳错误率：9.30%
ResNet-20最佳错误率：8.07%
PlainNet-56最佳错误率：12.03%
ResNet-56最佳错误率：6.92%

PlainNet加深恶化：12.03 - 9.30 = 2.73个百分点
ResNet加深改善：8.07 - 6.92 = 1.15个百分点
56层shortcut改善：12.03 - 6.92 = 5.11个百分点
ResNet-20比论文低：8.75 - 8.07 = 0.68个百分点
ResNet-56比论文低：6.97 - 6.92 = 0.05个百分点
```

- [ ] **Step 2: 解释结论与边界**

必须解释：

- PlainNet-56最终训练错误率3.75%高于PlainNet-20的1.12%，因此不是“训练集拟合
  更好但测试变差”的普通过拟合；
- ResNet-56最终训练错误率0.06%，说明shortcut显著缓解优化困难；
- 结果来自单个seed，能复现论文趋势但不能估计均值和标准差；
- PlainNet论文精确数值不在表格中，因此只做趋势对照；
- 32/44层未训练，所以当前曲线验证端点而不是完整四深度序列。

- [ ] **Step 3: 加入产物路径和复现命令**

文档链接summary和两张图，记录完整命令行；明确脚本只读原始结果。

- [ ] **Step 4: 最终验证**

Run:

```bat
E:\Anaconda3\envs\resnet-paper\python.exe -m py_compile analysis\compare_cifar_results.py tests\test_compare_cifar_results.py
git diff --check
E:\Anaconda3\envs\resnet-paper\python.exe -m pytest -q
```

Expected: compilation succeeds, diff check clean, full suite passes.

- [ ] **Step 5: 提交学习总结**

```bat
git add docs\阶段十-CIFAR正式实验结果分析总结.md
git commit -m "docs: summarize CIFAR reproduction results"
```

- [ ] **Step 6: 核对隔离状态**

Run:

```bat
git status --short
git branch --show-current
git log --oneline -7
```

Expected: clean worktree on `cifar-results-analysis`。不合并、不推送，等待用户选择。
