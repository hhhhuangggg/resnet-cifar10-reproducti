"""最终复现项目文档入口测试。"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
REPORT = ROOT / "docs" / "ResNet论文复现完整报告.md"


def test_final_documentation_entrypoints_exist() -> None:
    """最终报告、阶段总结和版本化分析产物必须存在且非空。"""

    paths = [
        README,
        REPORT,
        ROOT / "docs" / "阶段十一-完整复现分支整合总结.md",
        ROOT / "docs" / "阶段十三-四模型最终结果分析总结.md",
        ROOT / "analysis" / "results" / "cifar_summary.csv",
        ROOT / "analysis" / "results" / "cifar_error_curves.png",
        ROOT / "analysis" / "results" / "paper_comparison.png",
    ]

    for path in paths:
        assert path.is_file(), f"缺少最终项目入口：{path}"
        assert path.stat().st_size > 0, f"最终项目入口为空：{path}"


def test_readme_exposes_workflows() -> None:
    """README必须提供快速训练、正式训练、日志和分析入口。"""

    text = README.read_text(encoding="utf-8")
    for value in [
        "python train.py --model plain20 --epochs 5",
        "python train.py --model resnet20 --schedule paper",
        "tensorboard --logdir",
        "analysis/compare_cifar_results.py",
        "models/imagenet_resnet.py",
        "ResNet论文复现完整报告.md",
    ]:
        assert value in text


def test_report_states_scope_and_results() -> None:
    """报告必须记录四模型结论、ImageNet结构和实验边界。"""

    text = REPORT.read_text(encoding="utf-8")
    for value in [
        "PlainNet-20",
        "ResNet-20",
        "PlainNet-56",
        "ResNet-56",
        "2.73",
        "1.15",
        "5.11",
        "ResNet-18/34/50/101/152",
        "单个随机种子",
        "未训练",
    ]:
        assert value in text
