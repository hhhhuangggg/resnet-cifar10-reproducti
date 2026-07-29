"""验证、汇总并绘制CIFAR-10论文复现实验结果。"""
import csv
from dataclasses import dataclass
from pathlib import Path

import yaml


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
