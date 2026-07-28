"""论文64k iteration训练编排测试。"""

import argparse
import csv
import copy
from pathlib import Path

import pytest
import torch

import train
from engine import BatchProgress, EpochMetrics
from schedules import PaperSchedule
from train import (
    PAPER_HISTORY_FIELDS,
    build_experiment_config,
    build_experiment_paths,
    build_paper_checkpoint,
    build_paper_schedule,
    build_run_name,
    load_checkpoint,
    restore_paper_training_state,
    run_paper_training,
    validate_args,
    validate_paper_resume_checkpoint,
    write_history,
    write_paper_learning_rate_trace,
    write_paper_tensorboard_metrics,
)


def _make_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "model": "resnet20",
        "epochs": 5,
        "batch_size": 128,
        "learning_rate": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "seed": 42,
        "num_workers": 0,
        "run_name": None,
        "output_dir": "outputs",
        "log_dir": "runs",
        "device": "cuda",
        "schedule": "paper",
        "max_iterations": 64000,
        "resume": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_default_paper_arguments_are_valid() -> None:
    parser = argparse.ArgumentParser()

    validate_args(parser, _make_args())


def test_paper_run_name_is_isolated_from_short_run() -> None:
    paper_args = _make_args()
    short_args = _make_args(schedule="short")

    assert build_run_name(paper_args) == "seed42_paper64k"
    assert build_run_name(short_args) == "seed42_5epochs"


def test_paper_schedule_uses_cli_initial_learning_rate() -> None:
    schedule = build_paper_schedule(
        _make_args(
            max_iterations=70000,
            learning_rate=0.2,
        )
    )

    assert schedule.max_iterations == 70000
    assert schedule.milestones == (32000, 48000)
    assert schedule.learning_rates == pytest.approx((0.2, 0.02, 0.002))


def test_paper_config_records_exact_iteration_schedule() -> None:
    args = _make_args()
    output_path, log_path = build_experiment_paths(args)

    config = build_experiment_config(args, output_path, log_path)

    assert config["training"]["epochs"] is None
    assert config["training"]["max_iterations"] == 64000
    assert config["training"]["learning_rate_milestones"] == [32000, 48000]
    assert config["training"]["learning_rates"] == [0.1, 0.01, 0.001]
    assert output_path == Path("outputs/resnet20/seed42_paper64k")
    assert log_path == Path("runs/resnet20/seed42_paper64k")


@pytest.mark.parametrize("max_iterations", [1, 32000, 48000])
def test_paper_arguments_require_both_learning_rate_milestones(
    max_iterations: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = argparse.ArgumentParser()

    with pytest.raises(SystemExit):
        validate_args(parser, _make_args(max_iterations=max_iterations))

    assert "48000" in capsys.readouterr().err


class FakeWriter:
    def __init__(self, log_dir: str | None = None) -> None:
        self.log_dir = log_dir
        self.scalars: list[tuple[str, float, int]] = []
        self.flush_count = 0
        self.closed = False

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, value, step))

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.closed = True


def _paper_history_row() -> dict[str, object]:
    return {
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


def test_paper_history_uses_iteration_and_partial_epoch_columns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.csv"

    write_history(
        path,
        [_paper_history_row()],
        fields=PAPER_HISTORY_FIELDS,
    )

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == PAPER_HISTORY_FIELDS
    assert rows[0]["iteration"] == "7"
    assert rows[0]["train_batches"] == "3"
    assert rows[0]["is_partial_epoch"] == "True"
    assert b"\r\r\n" not in path.read_bytes()


def test_paper_tensorboard_metrics_use_iteration_as_step() -> None:
    writer = FakeWriter()

    write_paper_tensorboard_metrics(writer, _paper_history_row())

    assert [tag for tag, _, _ in writer.scalars] == [
        "Loss/train",
        "Loss/test",
        "Accuracy/train",
        "Accuracy/test",
        "Time/epoch_seconds",
        "Progress/epoch",
    ]
    assert all(step == 7 for _, _, step in writer.scalars)
    assert writer.flush_count == 1


def test_paper_learning_rate_trace_marks_exact_boundaries() -> None:
    writer = FakeWriter()
    schedule = PaperSchedule(
        max_iterations=7,
        milestones=(3, 5),
        learning_rates=(0.1, 0.01, 0.001),
    )

    write_paper_learning_rate_trace(writer, schedule)

    assert writer.scalars == [
        ("LearningRate", 0.1, 0),
        ("LearningRate", 0.01, 3),
        ("LearningRate", 0.001, 5),
        ("LearningRate", 0.001, 7),
    ]
    assert writer.flush_count == 1


def _paper_checkpoint_parts() -> tuple[
    argparse.Namespace,
    PaperSchedule,
    torch.nn.Module,
    torch.optim.Optimizer,
    dict[str, object],
    list[dict[str, object]],
]:
    args = _make_args(
        device="cpu",
        max_iterations=64000,
        learning_rate=0.1,
    )
    schedule = PaperSchedule(7, (3, 5), (0.1, 0.01, 0.001))
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001,
    )
    output_path, log_path = build_experiment_paths(args)
    config = build_experiment_config(args, output_path, log_path)
    config["training"]["max_iterations"] = 7
    config["training"]["learning_rate_milestones"] = [3, 5]
    config["training"]["learning_rates"] = [0.1, 0.01, 0.001]
    history = [
        {
            **_paper_history_row(),
            "epoch": 1,
            "iteration": 4,
            "train_batches": 4,
            "is_partial_epoch": False,
            "learning_rate": 0.01,
        }
    ]
    return args, schedule, model, optimizer, config, history


def test_paper_checkpoint_records_global_iteration_and_schedule() -> None:
    args, schedule, model, optimizer, config, history = (
        _paper_checkpoint_parts()
    )

    checkpoint = build_paper_checkpoint(
        model_name=args.model,
        epoch=1,
        global_iteration=4,
        model=model,
        optimizer=optimizer,
        best_test_accuracy=0.6,
        history=history,
        config=config,
        schedule=schedule,
    )

    assert checkpoint["global_iteration"] == 4
    assert checkpoint["current_learning_rate"] == pytest.approx(0.01)
    assert checkpoint["schedule_state"] == {
        "max_iterations": 7,
        "milestones": [3, 5],
        "learning_rates": [0.1, 0.01, 0.001],
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("global_iteration", 8, "iteration"),
        (
            "schedule_state",
            {
                "max_iterations": 8,
                "milestones": [3, 5],
                "learning_rates": [0.1, 0.01, 0.001],
            },
            "日程",
        ),
        ("current_learning_rate", 0.1, "学习率"),
    ],
)
def test_paper_checkpoint_rejects_conflicting_resume_state(
    field: str,
    value: object,
    message: str,
) -> None:
    args, schedule, model, optimizer, config, history = (
        _paper_checkpoint_parts()
    )
    checkpoint = build_paper_checkpoint(
        model_name=args.model,
        epoch=1,
        global_iteration=4,
        model=model,
        optimizer=optimizer,
        best_test_accuracy=0.6,
        history=history,
        config=config,
        schedule=schedule,
    )
    checkpoint[field] = value

    with pytest.raises(ValueError, match=message):
        validate_paper_resume_checkpoint(
            checkpoint,
            args,
            config,
            schedule,
        )


def test_paper_checkpoint_rejects_history_iteration_mismatch() -> None:
    args, schedule, model, optimizer, config, history = (
        _paper_checkpoint_parts()
    )
    checkpoint = build_paper_checkpoint(
        model_name=args.model,
        epoch=1,
        global_iteration=4,
        model=model,
        optimizer=optimizer,
        best_test_accuracy=0.6,
        history=history,
        config=config,
        schedule=schedule,
    )
    checkpoint["history"] = copy.deepcopy(history)
    checkpoint["history"][-1]["iteration"] = 3

    with pytest.raises(ValueError, match="history.*iteration"):
        validate_paper_resume_checkpoint(
            checkpoint,
            args,
            config,
            schedule,
        )


def test_restore_paper_training_state_returns_next_epoch_and_iteration() -> None:
    args, schedule, model, optimizer, config, history = (
        _paper_checkpoint_parts()
    )
    original = [parameter.detach().clone() for parameter in model.parameters()]
    checkpoint = build_paper_checkpoint(
        model_name=args.model,
        epoch=1,
        global_iteration=4,
        model=model,
        optimizer=optimizer,
        best_test_accuracy=0.6,
        history=history,
        config=config,
        schedule=schedule,
    )
    for parameter in model.parameters():
        parameter.data.zero_()

    next_epoch, iteration, best, restored_history = (
        restore_paper_training_state(
            checkpoint,
            model,
            optimizer,
            torch.device("cpu"),
        )
    )

    assert next_epoch == 2
    assert iteration == 4
    assert best == pytest.approx(0.6)
    assert restored_history == history
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(original, model.parameters())
    )


def _paper_run_args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        **vars(_make_args(device="cpu")),
        "run_name": "paper_integration",
        "output_dir": str(tmp_path / "outputs"),
        "log_dir": str(tmp_path / "runs"),
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_paper_training_stops_exactly_and_marks_partial_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = PaperSchedule(7, (3, 5), (0.1, 0.01, 0.001))
    learning_rates_used: list[float] = []
    evaluate_results = iter(
        [
            EpochMetrics(1.0, 0.5, 5, 10, 1),
            EpochMetrics(0.8, 0.6, 6, 10, 1),
        ]
    )

    monkeypatch.setattr(
        train,
        "create_cifar10_loaders",
        lambda **kwargs: (list(range(4)), ["test"]),
    )
    monkeypatch.setattr(
        train,
        "create_model",
        lambda name, num_classes=10: torch.nn.Linear(2, 2),
    )

    def fake_train_one_epoch(
        model: torch.nn.Module,
        data_loader: object,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        max_batches: int | None = None,
        on_batch_end: object | None = None,
    ) -> EpochMetrics:
        assert max_batches is not None
        for batch in range(1, max_batches + 1):
            learning_rates_used.append(optimizer.param_groups[0]["lr"])
            on_batch_end(
                BatchProgress(
                    batch=batch,
                    loss=1.0,
                    accuracy=0.5,
                    correct=batch,
                    samples=batch * 2,
                )
            )
        return EpochMetrics(
            loss=1.0,
            accuracy=0.5,
            correct=max_batches,
            samples=max_batches * 2,
            batches=max_batches,
        )

    monkeypatch.setattr(train, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(
        train,
        "evaluate",
        lambda *args, **kwargs: next(evaluate_results),
    )
    writer = FakeWriter()

    history = run_paper_training(
        _paper_run_args(tmp_path),
        schedule=schedule,
        show_progress=False,
        writer_factory=lambda **kwargs: writer,
    )

    assert learning_rates_used == pytest.approx(
        [0.1, 0.1, 0.1, 0.01, 0.01, 0.001, 0.001]
    )
    assert [row["iteration"] for row in history] == [4, 7]
    assert [row["train_batches"] for row in history] == [4, 3]
    assert [row["is_partial_epoch"] for row in history] == [False, True]
    output = tmp_path / "outputs" / "resnet20" / "paper_integration"
    assert load_checkpoint(output / "latest.pt")["global_iteration"] == 7
    assert load_checkpoint(output / "best.pt")["global_iteration"] == 7
    assert writer.closed is True
    metric_steps = [
        step
        for tag, _, step in writer.scalars
        if tag == "Accuracy/test"
    ]
    assert metric_steps == [4, 7]


def test_paper_training_resumes_from_last_evaluated_iteration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = PaperSchedule(7, (3, 5), (0.1, 0.01, 0.001))
    args = _paper_run_args(tmp_path)
    monkeypatch.setattr(
        train,
        "create_cifar10_loaders",
        lambda **kwargs: (list(range(4)), ["test"]),
    )
    monkeypatch.setattr(
        train,
        "create_model",
        lambda name, num_classes=10: torch.nn.Linear(2, 2),
    )
    monkeypatch.setattr(
        train,
        "evaluate",
        lambda *args, **kwargs: EpochMetrics(0.8, 0.6, 6, 10, 1),
    )

    train_calls = 0

    def interrupt_on_second_epoch(
        model: torch.nn.Module,
        data_loader: object,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        max_batches: int | None = None,
        on_batch_end: object | None = None,
    ) -> EpochMetrics:
        nonlocal train_calls
        train_calls += 1
        if train_calls == 2:
            raise KeyboardInterrupt
        assert max_batches == 4
        for batch in range(1, 5):
            on_batch_end(BatchProgress(batch, 1.0, 0.5, batch, batch * 2))
        return EpochMetrics(1.0, 0.5, 4, 8, 4)

    monkeypatch.setattr(
        train,
        "train_one_epoch",
        interrupt_on_second_epoch,
    )
    first_writer = FakeWriter()

    interrupted_history = run_paper_training(
        args,
        schedule=schedule,
        show_progress=False,
        writer_factory=lambda **kwargs: first_writer,
    )

    output = tmp_path / "outputs" / "resnet20" / "paper_integration"
    latest_path = output / "latest.pt"
    assert [row["iteration"] for row in interrupted_history] == [4]
    assert load_checkpoint(latest_path)["global_iteration"] == 4
    assert first_writer.closed is True

    resumed_learning_rates: list[float] = []

    def finish_partial_epoch(
        model: torch.nn.Module,
        data_loader: object,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        max_batches: int | None = None,
        on_batch_end: object | None = None,
    ) -> EpochMetrics:
        assert max_batches == 3
        for batch in range(1, 4):
            resumed_learning_rates.append(optimizer.param_groups[0]["lr"])
            on_batch_end(BatchProgress(batch, 1.0, 0.5, batch, batch * 2))
        return EpochMetrics(1.0, 0.5, 3, 6, 3)

    monkeypatch.setattr(train, "train_one_epoch", finish_partial_epoch)
    resume_args = _paper_run_args(
        tmp_path,
        run_name=None,
        resume=str(latest_path),
    )
    second_writer = FakeWriter()

    resumed_history = run_paper_training(
        resume_args,
        schedule=schedule,
        show_progress=False,
        writer_factory=lambda **kwargs: second_writer,
    )

    assert resumed_learning_rates == pytest.approx([0.01, 0.001, 0.001])
    assert [row["iteration"] for row in resumed_history] == [4, 7]
    assert load_checkpoint(latest_path)["global_iteration"] == 7
    assert second_writer.closed is True
