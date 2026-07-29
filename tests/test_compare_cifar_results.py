"""CIFAR-10正式实验结果分析测试。"""
import csv
from pathlib import Path

import pytest

from analysis.compare_cifar_results import load_history


FIELDS = [
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
            "epoch": 1,
            "iteration": 391,
            "train_batches": 391,
            "is_partial_epoch": False,
            "train_loss": 1.5,
            "train_accuracy": 0.4,
            "test_loss": 1.4,
            "test_accuracy": 0.5,
            "learning_rate": 0.1,
            "elapsed_seconds": 20.0,
            "best_test_accuracy": 0.5,
        },
        {
            "epoch": 164,
            "iteration": 64000,
            "train_batches": 267,
            "is_partial_epoch": True,
            "train_loss": 0.1,
            "train_accuracy": 0.98,
            "test_loss": 0.3,
            "test_accuracy": 0.91,
            "learning_rate": 0.001,
            "elapsed_seconds": 15.0,
            "best_test_accuracy": 0.91,
        },
    ]


def test_load_history_parses_completed_run(tmp_path: Path) -> None:
    """数值或布尔字段解析错误时，本测试应当失败。"""

    path = tmp_path / "history.csv"
    write_history(path, completed_rows())

    rows = load_history(path)

    assert [row.iteration for row in rows] == [391, 64000]
    assert rows[-1].is_partial_epoch is True
    assert rows[-1].test_accuracy == pytest.approx(0.91)


def test_load_history_rejects_missing_field(tmp_path: Path) -> None:
    """缺失关键指标仍被接受时，本测试应当失败。"""

    path = tmp_path / "history.csv"
    fields = [field for field in FIELDS if field != "test_accuracy"]
    write_history(path, completed_rows(), fields)

    with pytest.raises(ValueError, match="缺少字段"):
        load_history(path)


def test_load_history_rejects_unfinished_run(tmp_path: Path) -> None:
    """未到64000次更新仍被汇总时，本测试应当失败。"""

    path = tmp_path / "history.csv"
    rows = completed_rows()
    rows[-1]["iteration"] = 63999
    write_history(path, rows)

    with pytest.raises(ValueError, match="64000"):
        load_history(path)
