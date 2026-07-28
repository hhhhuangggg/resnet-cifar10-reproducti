"""第七阶段训练编排功能测试。"""

import argparse
from pathlib import Path
import csv
import random

import numpy as np
import pytest
import torch
import yaml

import train
from engine import EpochMetrics
from train import (
    HISTORY_FIELDS,
    atomic_save_checkpoint,
    build_checkpoint,
    capture_rng_state,
    load_checkpoint,
    prepare_experiment_directories,
    resolve_device,
    restore_rng_state,
    restore_training_state,
    run_training,
    set_random_seed,
    validate_args,
    validate_resume_checkpoint,
    write_config,
    write_history,
    write_tensorboard_metrics,
)


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


def test_resolve_device_returns_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_cuda_refuses_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_new_run_accepts_empty_directories(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    log = tmp_path / "runs"
    output.mkdir()
    log.mkdir()

    prepare_experiment_directories(output, log, resume=False)

    assert output.is_dir()
    assert log.is_dir()


@pytest.mark.parametrize("occupied_name", ["output", "log"])
def test_new_run_rejects_nonempty_experiment_directory(
    tmp_path: Path,
    occupied_name: str,
) -> None:
    output = tmp_path / "outputs"
    log = tmp_path / "runs"
    output.mkdir()
    log.mkdir()
    occupied = output if occupied_name == "output" else log
    (occupied / "old.txt").write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError, match="run-name|resume"):
        prepare_experiment_directories(output, log, resume=False)


def test_resume_accepts_existing_experiment_directories(
    tmp_path: Path,
) -> None:
    output = tmp_path / "outputs"
    log = tmp_path / "runs"
    output.mkdir()
    log.mkdir()
    (output / "latest.pt").write_bytes(b"checkpoint")
    (log / "events.out").write_text("event", encoding="utf-8")

    prepare_experiment_directories(output, log, resume=True)


class FakeWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.flush_count = 0

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, value, step))

    def flush(self) -> None:
        self.flush_count += 1


def _history_row(epoch: int = 1) -> dict[str, object]:
    return {
        "epoch": epoch,
        "train_loss": 1.2,
        "train_accuracy": 0.5,
        "test_loss": 1.1,
        "test_accuracy": 0.6,
        "learning_rate": 0.1,
        "elapsed_seconds": 2.5,
        "best_test_accuracy": 0.6,
    }


def test_write_config_is_utf8_round_trippable_and_atomic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"
    config = {"说明": "残差网络", "epochs": 5}

    write_config(path, config)

    assert yaml.safe_load(path.read_text(encoding="utf-8")) == config
    assert not (tmp_path / "config.yaml.tmp").exists()


def test_write_history_uses_fixed_columns_and_replaces_atomically(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.csv"

    write_history(path, [_history_row()])

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == HISTORY_FIELDS
    assert rows[0]["epoch"] == "1"
    assert float(rows[0]["test_accuracy"]) == pytest.approx(0.6)
    assert b"\r\r\n" not in path.read_bytes()
    assert not (tmp_path / "history.csv.tmp").exists()


def test_write_tensorboard_metrics_writes_six_tags_and_flushes() -> None:
    writer = FakeWriter()

    write_tensorboard_metrics(writer, _history_row(epoch=3))

    assert [tag for tag, _, _ in writer.scalars] == [
        "Loss/train",
        "Loss/test",
        "Accuracy/train",
        "Accuracy/test",
        "LearningRate",
        "Time/epoch_seconds",
    ]
    assert all(step == 3 for _, _, step in writer.scalars)
    assert writer.flush_count == 1


def _checkpoint_parts() -> tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    dict[str, object],
    list[dict[str, object]],
]:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    config: dict[str, object] = {
        "model": {"name": "resnet20"},
        "training": {
            "schedule": "short",
            "epochs": 5,
            "batch_size": 128,
            "learning_rate": 0.1,
            "momentum": 0.9,
            "weight_decay": 0.0001,
            "seed": 42,
            "device": "cpu",
        },
    }
    history = [_history_row()]
    return model, optimizer, config, history


def test_checkpoint_round_trip_is_atomic_and_has_required_fields(
    tmp_path: Path,
) -> None:
    model, optimizer, config, history = _checkpoint_parts()
    checkpoint = build_checkpoint(
        model_name="resnet20",
        epoch=1,
        model=model,
        optimizer=optimizer,
        best_test_accuracy=0.6,
        history=history,
        config=config,
    )
    path = tmp_path / "latest.pt"

    atomic_save_checkpoint(path, checkpoint)
    loaded = load_checkpoint(path)

    assert set(loaded) == {
        "format_version",
        "model_name",
        "epoch",
        "model_state",
        "optimizer_state",
        "best_test_accuracy",
        "history",
        "config",
        "rng_state",
    }
    assert loaded["format_version"] == 1
    assert not (tmp_path / "latest.pt.tmp").exists()


def test_validate_resume_checkpoint_rejects_model_conflict() -> None:
    model, optimizer, config, history = _checkpoint_parts()
    checkpoint = build_checkpoint(
        model_name="plain20",
        epoch=1,
        model=model,
        optimizer=optimizer,
        best_test_accuracy=0.6,
        history=history,
        config=config,
    )

    with pytest.raises(ValueError, match="模型"):
        validate_resume_checkpoint(checkpoint, "resnet20", config, 5)


def test_restore_training_state_restores_model_optimizer_and_epoch() -> None:
    model, optimizer, config, history = _checkpoint_parts()
    original = [parameter.detach().clone() for parameter in model.parameters()]
    checkpoint = build_checkpoint(
        model_name="resnet20",
        epoch=1,
        model=model,
        optimizer=optimizer,
        best_test_accuracy=0.6,
        history=history,
        config=config,
    )
    for parameter in model.parameters():
        parameter.data.zero_()

    start_epoch, best, restored_history = restore_training_state(
        checkpoint,
        model,
        optimizer,
        torch.device("cpu"),
    )

    assert start_epoch == 2
    assert best == pytest.approx(0.6)
    assert restored_history == history
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(original, model.parameters())
    )


def test_rng_state_restores_python_numpy_and_torch_sequence() -> None:
    set_random_seed(17)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(2))

    random.random()
    np.random.rand()
    torch.rand(2)
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(2))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


class ClosableFakeWriter(FakeWriter):
    instances: list["ClosableFakeWriter"] = []

    def __init__(self, log_dir: str) -> None:
        super().__init__()
        self.log_dir = log_dir
        self.closed = False
        self.instances.append(self)

    def close(self) -> None:
        self.closed = True


def _run_args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "model": "resnet20",
        "epochs": 2,
        "batch_size": 4,
        "learning_rate": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "seed": 42,
        "num_workers": 0,
        "run_name": "integration",
        "output_dir": str(tmp_path / "outputs"),
        "log_dir": str(tmp_path / "runs"),
        "device": "cpu",
        "schedule": "short",
        "max_iterations": 64000,
        "resume": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_run_training_records_two_complete_epochs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ClosableFakeWriter.instances.clear()
    train_results = iter(
        [
            EpochMetrics(1.5, 0.4, 4, 10, 1),
            EpochMetrics(1.0, 0.6, 6, 10, 1),
        ]
    )
    test_results = iter(
        [
            EpochMetrics(1.4, 0.5, 5, 10, 1),
            EpochMetrics(0.9, 0.7, 7, 10, 1),
        ]
    )
    monkeypatch.setattr(
        train,
        "create_cifar10_loaders",
        lambda **kwargs: (["train"], ["test"]),
    )
    monkeypatch.setattr(
        train,
        "create_model",
        lambda name, num_classes=10: torch.nn.Linear(2, 2),
    )
    monkeypatch.setattr(
        train,
        "train_one_epoch",
        lambda *args, **kwargs: next(train_results),
    )
    monkeypatch.setattr(
        train,
        "evaluate",
        lambda *args, **kwargs: next(test_results),
    )
    monkeypatch.setattr(train, "SummaryWriter", ClosableFakeWriter)

    history = run_training(_run_args(tmp_path), show_progress=False)

    output = tmp_path / "outputs" / "resnet20" / "integration"
    assert [row["epoch"] for row in history] == [1, 2]
    assert history[-1]["test_accuracy"] == pytest.approx(0.7)
    assert (output / "config.yaml").is_file()
    assert (output / "history.csv").is_file()
    assert load_checkpoint(output / "latest.pt")["epoch"] == 2
    assert load_checkpoint(output / "best.pt")["epoch"] == 2
    writer = ClosableFakeWriter.instances[-1]
    assert writer.closed is True
    assert len(writer.scalars) == 12


def test_main_calls_run_training_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _run_args(tmp_path)
    calls: list[argparse.Namespace] = []
    monkeypatch.setattr(train, "parse_args", lambda: args)
    monkeypatch.setattr(train, "run_training", calls.append)

    train.main()

    assert calls == [args]


def test_validate_args_rejects_paper_schedule_before_training(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = argparse.ArgumentParser()
    args = _run_args(Path("."), schedule="paper")

    with pytest.raises(SystemExit):
        validate_args(parser, args)

    assert "paper 64k iteration" in capsys.readouterr().err
