"""论文64k iteration训练编排测试。"""

import argparse
from pathlib import Path

import pytest

from train import (
    build_experiment_config,
    build_experiment_paths,
    build_paper_schedule,
    build_run_name,
    validate_args,
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
