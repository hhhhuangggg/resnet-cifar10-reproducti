"""Static checks for a portable GitHub reproduction release."""

from pathlib import Path
import re

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


def test_readme_describes_authorized_clean_clone_workflow() -> None:
    readme = read_text("README.md")

    required = (
        "Private",
        "git@github.com:hhhhuangggg/resnet-cifar10-reproducti.git",
        "https://github.com/hhhhuangggg/resnet-cifar10-reproducti.git",
        "conda env create -f environment.yml",
        "torch.cuda.is_available()",
        "python verify_reproduction.py",
        "python -m pytest",
        "--epochs 1",
        "--epochs 5",
        "--schedule paper",
        "tensorboard --logdir runs",
        "--include-imagenet",
        "ImageNet",
        "outputs/",
        "runs/",
    )
    for text in required:
        assert text in readme
    assert r"E:\文档" not in readme
    assert "ImageNet 结构验证不代表完成 ImageNet 训练" in readme


def test_readme_relative_links_exist() -> None:
    readme = read_text("README.md")
    targets = re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme)
    relative_targets = [
        target.split("#", 1)[0]
        for target in targets
        if target
        and not target.startswith(("http://", "https://", "mailto:"))
        and not target.startswith("#")
    ]

    assert relative_targets
    assert all((ROOT / target).exists() for target in relative_targets)


def test_project_report_has_project_specific_structure() -> None:
    report = read_text("docs/ResNet复现报告.md")
    headings = re.findall(r"^## (.+)$", report, flags=re.MULTILINE)

    assert headings == [
        "一、复现目标与学习收获",
        "二、ResNet 核心原理",
        "三、项目实现过程",
        "四、实验环境与训练配置",
        "五、实验结果",
        "六、结果分析与论文对比",
        "七、从私有 GitHub 仓库复现",
        "八、局限性与后续工作",
        "九、结论",
    ]
    for value in ("9.30%", "8.07%", "12.03%", "6.92%"):
        assert value in report
    assert "Private" in report
    assert "只有仓库所有者和受邀协作者" in report
    assert "DQN" not in report
    assert "Breakout" not in report


def test_project_report_relative_images_exist() -> None:
    report_path = ROOT / "docs" / "ResNet复现报告.md"
    report = report_path.read_text(encoding="utf-8")
    image_targets = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report)

    assert len(image_targets) == 2
    assert all((report_path.parent / target).exists() for target in image_targets)
