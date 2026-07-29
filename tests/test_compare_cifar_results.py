"""CIFAR-10正式实验结果分析测试。"""
import csv
from pathlib import Path

import pytest
import yaml

from analysis.compare_cifar_results import (
    load_config,
    load_history,
    run_analysis,
    summarize_experiment,
    validate_experiment_config,
    write_summary_csv,
)


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
    training = config["training"]
    assert isinstance(training, dict)
    training[field] = bad_value

    with pytest.raises(ValueError, match=field):
        validate_experiment_config(config, "plain20")


def test_summary_keeps_best_and_final_separate(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    rows = completed_rows()
    rows[0]["test_accuracy"] = 0.92
    rows[0]["best_test_accuracy"] = 0.92
    rows[-1]["test_accuracy"] = 0.91
    rows[-1]["best_test_accuracy"] = 0.92
    write_history(path, rows)

    summary = summarize_experiment(
        "plain20",
        load_history(path),
        269722,
    )

    assert summary.best_test_error_percent == pytest.approx(8.0)
    assert summary.final_test_error_percent == pytest.approx(9.0)
    assert summary.final_train_error_percent == pytest.approx(2.0)
    assert summary.best_iteration == 391
    assert summary.total_minutes == pytest.approx(35.0 / 60.0)


def test_write_summary_csv_uses_fixed_header(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    write_history(path, completed_rows())
    summary = summarize_experiment(
        "plain20",
        load_history(path),
        269722,
    )
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
        "plain20",
        "resnet20",
        "plain56",
        "resnet56",
    ]
    for filename in (
        "cifar_summary.csv",
        "cifar_error_curves.png",
        "paper_comparison.png",
    ):
        artifact = output_dir / filename
        assert artifact.is_file()
        assert artifact.stat().st_size > 0
