"""Tests for experiment configuration and argument validation."""

import argparse
from pathlib import Path

import pytest

from train import build_experiment_config, validate_args


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
        "schedule": "short",
        "max_iterations": 64000,
        "resume": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_experiment_config_records_resume_and_paths() -> None:
    args = _make_args()
    output_path = Path("outputs/resnet20/seed42_5epochs")
    log_path = Path("runs/resnet20/seed42_5epochs")

    config = build_experiment_config(args, output_path, log_path)

    assert config["training"]["resume_from"] is None
    assert config["paths"] == {
        "output": str(output_path),
        "tensorboard": str(log_path),
    }


def test_resume_and_run_name_conflict_is_reported_first(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = argparse.ArgumentParser()
    args = _make_args(
        resume="missing.pt",
        run_name="new-name",
    )

    with pytest.raises(SystemExit):
        validate_args(parser, args)

    assert "使用--resume时不能同时指定--run-name" in capsys.readouterr().err
