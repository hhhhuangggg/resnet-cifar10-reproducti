"""验证、汇总并绘制CIFAR-10论文复现实验结果。"""
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt


REQUIRED_HISTORY_FIELDS = {
    "epoch",
    "iteration",
    "train_batches",
    "is_partial_epoch",
    "train_loss",
    "train_accuracy",
    "test_loss",
    "test_accuracy",
    "learning_rate",
    "elapsed_seconds",
    "best_test_accuracy",
}

EXPECTED_TRAINING = {
    "schedule": "paper",
    "max_iterations": 64000,
    "learning_rate_milestones": [32000, 48000],
    "learning_rates": [0.1, 0.01, 0.001],
    "batch_size": 128,
    "seed": 42,
}


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


@dataclass(frozen=True)
class HistoryRow:
    """一次epoch评估对应的正式实验记录。"""

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


@dataclass(frozen=True)
class ExperimentSummary:
    """一组完整64k实验的关键指标。"""

    model: str
    parameters: int
    best_epoch: int
    best_iteration: int
    best_test_accuracy_percent: float
    best_test_error_percent: float
    final_train_accuracy_percent: float
    final_train_error_percent: float
    final_test_accuracy_percent: float
    final_test_error_percent: float
    total_minutes: float
    completed_iterations: int


SUMMARY_FIELDS = [
    "model",
    "parameters",
    "best_epoch",
    "best_iteration",
    "best_test_accuracy_percent",
    "best_test_error_percent",
    "final_train_accuracy_percent",
    "final_train_error_percent",
    "final_test_accuracy_percent",
    "final_test_error_percent",
    "total_minutes",
    "completed_iterations",
]


def _parse_bool(value: str, *, field: str, line: int) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(
        f"第{line}行{field}必须是True或False，实际为{value!r}"
    )


def _parse_history_row(
    row: dict[str, str],
    *,
    line: int,
) -> HistoryRow:
    try:
        parsed = HistoryRow(
            epoch=int(row["epoch"]),
            iteration=int(row["iteration"]),
            train_batches=int(row["train_batches"]),
            is_partial_epoch=_parse_bool(
                row["is_partial_epoch"],
                field="is_partial_epoch",
                line=line,
            ),
            train_loss=float(row["train_loss"]),
            train_accuracy=float(row["train_accuracy"]),
            test_loss=float(row["test_loss"]),
            test_accuracy=float(row["test_accuracy"]),
            learning_rate=float(row["learning_rate"]),
            elapsed_seconds=float(row["elapsed_seconds"]),
            best_test_accuracy=float(row["best_test_accuracy"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"history.csv第{line}行包含无效数值：{error}"
        ) from error

    for field, value in (
        ("train_accuracy", parsed.train_accuracy),
        ("test_accuracy", parsed.test_accuracy),
        ("best_test_accuracy", parsed.best_test_accuracy),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"第{line}行{field}必须位于[0,1]，实际为{value}"
            )
    if parsed.epoch <= 0 or parsed.iteration <= 0:
        raise ValueError(f"第{line}行epoch和iteration必须大于0")
    if parsed.train_batches <= 0:
        raise ValueError(f"第{line}行train_batches必须大于0")
    if parsed.elapsed_seconds < 0:
        raise ValueError(f"第{line}行elapsed_seconds不能小于0")
    return parsed


def load_history(path: str | Path) -> list[HistoryRow]:
    """读取并验证一个已经完成64k训练的history.csv。"""

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"history.csv不存在：{resolved}")

    with resolved.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_fields = set(reader.fieldnames or ())
        missing = REQUIRED_HISTORY_FIELDS - actual_fields
        if missing:
            raise ValueError(
                "history.csv缺少字段："
                + ", ".join(sorted(missing))
            )
        rows = [
            _parse_history_row(row, line=index)
            for index, row in enumerate(reader, start=2)
        ]

    if not rows:
        raise ValueError("history.csv不能是空文件")

    iterations = [row.iteration for row in rows]
    if any(
        current <= previous
        for previous, current in zip(iterations, iterations[1:])
    ):
        raise ValueError("history.csv中的iteration必须严格递增")
    if rows[-1].iteration != 64000:
        raise ValueError(
            "正式实验最后iteration必须为64000，"
            f"实际为{rows[-1].iteration}"
        )
    if not rows[-1].is_partial_epoch:
        raise ValueError("64000 iteration记录必须是最后部分epoch")
    return rows


def load_config(path: str | Path) -> dict[str, object]:
    """读取实验配置，并检查配置文件的基本结构。"""

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"config.yaml不存在：{resolved}")

    with resolved.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("config.yaml根对象必须是字典")
    for section in ("model", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"config.yaml中的{section}必须是字典")
    return config


def validate_experiment_config(
    config: dict[str, object],
    expected_model: str,
) -> None:
    """确认实验模型与所有影响公平比较的训练设置完全一致。"""

    model = config.get("model")
    training = config.get("training")
    if not isinstance(model, dict):
        raise ValueError("config中的model必须是字典")
    if not isinstance(training, dict):
        raise ValueError("config中的training必须是字典")

    actual_model = model.get("name")
    if actual_model != expected_model:
        raise ValueError(
            f"model.name期望为{expected_model!r}，实际为{actual_model!r}"
        )

    for field, expected in EXPECTED_TRAINING.items():
        actual = training.get(field)
        if actual != expected:
            raise ValueError(
                f"training.{field}期望为{expected!r}，实际为{actual!r}"
            )


def summarize_experiment(
    model: str,
    rows: list[HistoryRow],
    parameters: int,
) -> ExperimentSummary:
    """分别汇总最佳测试点与训练结束点。"""

    if not rows:
        raise ValueError("实验记录不能为空")
    if parameters <= 0:
        raise ValueError("parameters必须大于0")

    best = max(rows, key=lambda row: row.test_accuracy)
    final = rows[-1]
    return ExperimentSummary(
        model=model,
        parameters=parameters,
        best_epoch=best.epoch,
        best_iteration=best.iteration,
        best_test_accuracy_percent=100.0 * best.test_accuracy,
        best_test_error_percent=100.0 * (1.0 - best.test_accuracy),
        final_train_accuracy_percent=100.0 * final.train_accuracy,
        final_train_error_percent=100.0 * (1.0 - final.train_accuracy),
        final_test_accuracy_percent=100.0 * final.test_accuracy,
        final_test_error_percent=100.0 * (1.0 - final.test_accuracy),
        total_minutes=sum(row.elapsed_seconds for row in rows) / 60.0,
        completed_iterations=final.iteration,
    )


def write_summary_csv(
    summaries: list[ExperimentSummary],
    path: str | Path,
) -> None:
    """使用固定列顺序和临时文件原子写入汇总结果。"""

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for summary in summaries:
            row: dict[str, object] = {}
            for field in SUMMARY_FIELDS:
                value = getattr(summary, field)
                row[field] = (
                    f"{value:.6f}"
                    if isinstance(value, float)
                    else value
                )
            writer.writerow(row)

    temporary.replace(resolved)


def plot_error_curves(
    histories: dict[str, list[HistoryRow]],
    path: str | Path,
) -> None:
    """绘制四组实验未经平滑的训练和测试错误率曲线。"""

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=160)

    for model in MODELS:
        rows = histories[model]
        iterations = [row.iteration for row in rows]
        train_errors = [
            100.0 * (1.0 - row.train_accuracy) for row in rows
        ]
        test_errors = [
            100.0 * (1.0 - row.test_accuracy) for row in rows
        ]
        axes[0].plot(
            iterations,
            train_errors,
            label=model,
            color=COLORS[model],
            linewidth=1.8,
        )
        axes[1].plot(
            iterations,
            test_errors,
            label=model,
            color=COLORS[model],
            linewidth=1.8,
        )

    for axis, title in zip(
        axes,
        ("Training error", "Test error"),
    ):
        axis.axvline(
            32_000,
            color="#888888",
            linestyle="--",
            linewidth=1,
            label="LR milestone" if axis is axes[0] else None,
        )
        axis.axvline(
            48_000,
            color="#888888",
            linestyle="--",
            linewidth=1,
        )
        axis.axvline(
            64_000,
            color="#222222",
            linestyle=":",
            linewidth=1,
            label="Training end" if axis is axes[0] else None,
        )
        axis.set_title(title)
        axis.set_xlabel("Iteration")
        axis.set_ylabel("Error (%)")
        axis.grid(alpha=0.25)
        axis.legend(ncol=3)

    figure.suptitle("CIFAR-10 PlainNet vs ResNet (64k schedule)")
    figure.tight_layout()
    figure.savefig(resolved, bbox_inches="tight")
    plt.close(figure)


def plot_paper_comparison(
    summaries: list[ExperimentSummary],
    path: str | Path,
) -> None:
    """对比ResNet-20/56论文错误率与本次复现最佳错误率。"""

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    summary_by_model = {summary.model: summary for summary in summaries}
    models = ("resnet20", "resnet56")
    paper_values = [PAPER_TEST_ERRORS[model] for model in models]
    reproduced_values = [
        summary_by_model[model].best_test_error_percent
        for model in models
    ]
    positions = range(len(models))
    width = 0.36

    figure, axis = plt.subplots(figsize=(8, 5), dpi=160)
    paper_bars = axis.bar(
        [position - width / 2 for position in positions],
        paper_values,
        width,
        label="Paper",
        color="#999999",
    )
    reproduced_bars = axis.bar(
        [position + width / 2 for position in positions],
        reproduced_values,
        width,
        label="Reproduction",
        color=[COLORS[model] for model in models],
    )
    axis.set_xticks(list(positions), ["ResNet-20", "ResNet-56"])
    axis.set_ylabel("Best test error (%)")
    axis.set_title("CIFAR-10 paper comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(paper_bars, fmt="%.2f%%", padding=3)
    axis.bar_label(reproduced_bars, fmt="%.2f%%", padding=3)
    upper = max(paper_values + reproduced_values)
    axis.set_ylim(0, upper * 1.18)
    figure.tight_layout()
    figure.savefig(resolved, bbox_inches="tight")
    plt.close(figure)


def run_analysis(
    results_root: str | Path,
    output_dir: str | Path,
) -> list[ExperimentSummary]:
    """校验四组正式实验并生成汇总表与图。"""

    root = Path(results_root)
    destination = Path(output_dir)
    histories: dict[str, list[HistoryRow]] = {}
    summaries: list[ExperimentSummary] = []

    for model in MODELS:
        experiment = root / model / "seed42_paper64k"
        config = load_config(experiment / "config.yaml")
        validate_experiment_config(config, model)
        history = load_history(experiment / "history.csv")
        histories[model] = history
        summaries.append(
            summarize_experiment(model, history, PARAMETERS[model])
        )

    write_summary_csv(summaries, destination / "cifar_summary.csv")
    plot_error_curves(histories, destination / "cifar_error_curves.png")
    plot_paper_comparison(
        summaries,
        destination / "paper_comparison.png",
    )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="校验并分析四组CIFAR-10正式64k实验",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="包含plain20/resnet20/plain56/resnet56的outputs目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/results"),
        help="汇总CSV和PNG图表的输出目录",
    )
    args = parser.parse_args()

    summaries = run_analysis(args.results_root, args.output_dir)
    for summary in summaries:
        print(
            f"{summary.model} "
            f"best_test_error={summary.best_test_error_percent:.2f}%"
        )


if __name__ == "__main__":
    main()
