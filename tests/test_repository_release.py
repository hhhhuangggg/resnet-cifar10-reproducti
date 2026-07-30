"""Static checks for a portable GitHub reproduction release."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_environment_file_is_machine_independent() -> None:
    raw = read_text("environment.yml")
    environment = yaml.safe_load(raw)

    assert environment["name"] == "resnet-paper"
    assert "prefix:" not in raw
    assert "python=3.11.15" in environment["dependencies"]
    pip_dependencies = next(
        item["pip"]
        for item in environment["dependencies"]
        if isinstance(item, dict) and "pip" in item
    )
    assert "torch==2.5.1+cu121" in pip_dependencies
    assert "torchvision==0.20.1+cu121" in pip_dependencies


def test_lock_file_records_verified_core_versions() -> None:
    lock = read_text("requirements-lock.txt")

    assert "torch==2.5.1+cu121" in lock
    assert "torchvision==0.20.1+cu121" in lock
    assert "pytest==8.3.3" in lock


def test_gitignore_excludes_local_data_and_training_artifacts() -> None:
    ignored = read_text(".gitignore")

    for pattern in (
        ".worktrees/",
        "data/",
        "runs/",
        "*.pt",
        "outputs/",
        ".pytest-tmp/",
    ):
        assert pattern in ignored
