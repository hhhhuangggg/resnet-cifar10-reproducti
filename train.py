"""CIFAR-10 单模型训练入口。

后续负责：
1. 读取命令行参数。
2. 创建指定模型。
3. 加载 CIFAR-10。
4. 执行训练和测试。
5. 记录 CSV 与 TensorBoard 日志。
6. 保存和恢复 checkpoint。

本文件不直接定义 PlainNet 或 ResNet。
"""

import argparse
import copy
import csv
import io
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from data import create_cifar10_loaders
from engine import BatchProgress, evaluate, train_one_epoch
from models import MODEL_CONFIGS, SUPPORTED_MODELS, create_model
from schedules import PaperSchedule

SHORT_HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "train_accuracy",
    "test_loss",
    "test_accuracy",
    "learning_rate",
    "elapsed_seconds",
    "best_test_accuracy",
]
HISTORY_FIELDS = SHORT_HISTORY_FIELDS

PAPER_HISTORY_FIELDS = [
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


def atomic_write_text(path: Path, text: str) -> None:
    """先完整写临时文件，再原子替换目标文本文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def write_config(path: Path, config: dict[str, object]) -> None:
    """以UTF-8 YAML原子保存实验配置。"""

    validate_config_round_trip(config)
    atomic_write_text(path, config_to_yaml(config))


def write_history(
    path: Path,
    history: list[dict[str, object]],
    *,
    fields: list[str] = SHORT_HISTORY_FIELDS,
) -> None:
    """以固定字段原子重写完整epoch历史。"""

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fields,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(history)
    atomic_write_text(path, buffer.getvalue())


def write_tensorboard_metrics(
    writer: object,
    row: dict[str, object],
) -> None:
    """写入一轮训练的六项TensorBoard标量并立即刷新。"""

    epoch = int(row["epoch"])
    metrics = (
        ("Loss/train", row["train_loss"]),
        ("Loss/test", row["test_loss"]),
        ("Accuracy/train", row["train_accuracy"]),
        ("Accuracy/test", row["test_accuracy"]),
        ("LearningRate", row["learning_rate"]),
        ("Time/epoch_seconds", row["elapsed_seconds"]),
    )
    for tag, value in metrics:
        writer.add_scalar(tag, float(value), epoch)
    writer.flush()


def write_paper_tensorboard_metrics(
    writer: object,
    row: dict[str, object],
) -> None:
    """以全局iteration为横轴写入paper评估指标。"""

    step = int(row["iteration"])
    metrics = (
        ("Loss/train", row["train_loss"]),
        ("Loss/test", row["test_loss"]),
        ("Accuracy/train", row["train_accuracy"]),
        ("Accuracy/test", row["test_accuracy"]),
        ("Time/epoch_seconds", row["elapsed_seconds"]),
        ("Progress/epoch", row["epoch"]),
    )
    for tag, value in metrics:
        writer.add_scalar(tag, float(value), step)
    writer.flush()


def write_paper_learning_rate_trace(
    writer: object,
    schedule: PaperSchedule,
) -> None:
    """在起点、里程碑和终点记录paper学习率阶梯。"""

    first, second = schedule.milestones
    points = (
        (0, schedule.learning_rates[0]),
        (first, schedule.learning_rates[1]),
        (second, schedule.learning_rates[2]),
        (schedule.max_iterations, schedule.learning_rates[2]),
    )
    for step, learning_rate in points:
        writer.add_scalar("LearningRate", float(learning_rate), step)
    writer.flush()


def capture_rng_state() -> dict[str, object]:
    """捕获可用于epoch边界恢复的全部随机状态。"""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def restore_rng_state(state: dict[str, object]) -> None:
    """恢复Python、NumPy、CPU PyTorch及可用的CUDA随机状态。"""

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(state, dict) or set(state) != required:
        raise ValueError("checkpoint中的RNG状态不完整")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint包含CUDA随机状态，但CUDA当前不可用")
        torch.cuda.set_rng_state_all(cuda_state)


def build_checkpoint(
    *,
    model_name: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_test_accuracy: float,
    history: list[dict[str, object]],
    config: dict[str, object],
) -> dict[str, object]:
    """整理一个完整epoch结束后的可恢复训练状态。"""

    return {
        "format_version": 1,
        "model_name": model_name,
        "epoch": epoch,
        "model_state": copy.deepcopy(model.state_dict()),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "best_test_accuracy": best_test_accuracy,
        "history": copy.deepcopy(history),
        "config": copy.deepcopy(config),
        "rng_state": capture_rng_state(),
    }


def atomic_save_checkpoint(
    path: Path,
    checkpoint: dict[str, object],
) -> None:
    """原子保存checkpoint，避免中断留下半个目标文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(path: Path) -> dict[str, object]:
    """从CPU读取本项目生成的可信checkpoint。"""

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint顶层必须是字典")
    return checkpoint


def _resume_config_values(config: dict[str, object]) -> tuple[object, ...]:
    model = config.get("model")
    training = config.get("training")
    if not isinstance(model, dict) or not isinstance(training, dict):
        raise ValueError("checkpoint配置缺少model或training")
    return (
        model.get("name"),
        training.get("schedule"),
        training.get("epochs"),
        training.get("batch_size"),
        training.get("learning_rate"),
        training.get("momentum"),
        training.get("weight_decay"),
        training.get("seed"),
        training.get("device"),
    )


def validate_resume_checkpoint(
    checkpoint: dict[str, object],
    model_name: str,
    config: dict[str, object],
    target_epochs: int,
) -> None:
    """在修改模型前完整校验恢复文件的结构和实验含义。"""

    required = {
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
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"checkpoint缺少字段：{sorted(missing)}")
    if checkpoint["format_version"] != 1:
        raise ValueError("不支持的checkpoint格式版本")
    if checkpoint["model_name"] != model_name:
        raise ValueError("checkpoint模型与命令行模型不一致")
    epoch = checkpoint["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("checkpoint epoch必须是非负整数")
    if epoch > target_epochs:
        raise ValueError("checkpoint epoch超过目标epochs")
    best = checkpoint["best_test_accuracy"]
    if not isinstance(best, (int, float)) or not 0.0 <= float(best) <= 1.0:
        raise ValueError("checkpoint最佳准确率必须位于0到1")
    history = checkpoint["history"]
    if not isinstance(history, list) or len(history) != epoch:
        raise ValueError("checkpoint history长度必须等于epoch")
    if [row.get("epoch") for row in history if isinstance(row, dict)] != list(
        range(1, epoch + 1)
    ):
        raise ValueError("checkpoint history的epoch必须从1连续递增")
    saved_config = checkpoint["config"]
    if not isinstance(saved_config, dict):
        raise ValueError("checkpoint config必须是字典")
    if _resume_config_values(saved_config) != _resume_config_values(config):
        raise ValueError("checkpoint配置与当前实验配置冲突")
    if not isinstance(checkpoint["model_state"], dict):
        raise ValueError("checkpoint model_state必须是字典")
    if not isinstance(checkpoint["optimizer_state"], dict):
        raise ValueError("checkpoint optimizer_state必须是字典")
    if not isinstance(checkpoint["rng_state"], dict):
        raise ValueError("checkpoint rng_state必须是字典")


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def restore_training_state(
    checkpoint: dict[str, object],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float, list[dict[str, object]]]:
    """加载模型、优化器和RNG，并返回下一轮及历史信息。"""

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    _move_optimizer_state(optimizer, device)
    restore_rng_state(checkpoint["rng_state"])
    history = list(checkpoint["history"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint["best_test_accuracy"]),
        history,
    )


def set_random_seed(seed: int) -> None:
    """固定Python、NumPy和PyTorch随机状态。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    """解析训练设备；请求CUDA失败时不静默回退CPU。"""

    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("请求使用CUDA，但当前PyTorch无法使用CUDA")


def prepare_experiment_directories(
    output_path: Path,
    log_path: Path,
    *,
    resume: bool,
) -> None:
    """创建实验目录，并防止新实验覆盖已有结果或日志。"""

    if not resume:
        for path in (output_path, log_path):
            if path.exists() and any(path.iterdir()):
                raise FileExistsError(
                    "实验目录已经存在，请更换--run-name或使用--resume"
                )
    output_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

def validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    """检查训练参数是否处于合理范围."""

    if args.epochs <= 0:
        parser.error("--epochs必须大于0")

    if args.batch_size <= 0:
        parser.error("--batch-size必须大于0")

    if args.learning_rate <= 0:
        parser.error("--learning-rate必须大于0")

    if not 0 <= args.momentum < 1:
        parser.error("--momentum必须满足0 <= momentum < 1")

    if args.weight_decay < 0:
        parser.error("--weight-decay不能小于0")

    if args.seed < 0:
        parser.error("--seed不能小于0")

    if args.num_workers < 0:
        parser.error("--num-workers不能小于0")
    if args.run_name is not None:
        run_name = args.run_name.strip()

        if not run_name:
            parser.error("--run-name不能为空")

        if Path(run_name).name != run_name:
            parser.error(
                "--run-name只能是名称，不能包含目录或路径分隔符"
            )
    if args.max_iterations <= 0:
        parser.error("--max-iterations必须大于0")
    if (
        args.schedule == "paper"
        and args.max_iterations <= 48000
    ):
        parser.error(
            "paper日程的--max-iterations必须大于48000，"
            "否则无法经过论文的第二个学习率节点"
        )
    if (
        args.resume is not None
        and args.run_name is not None
    ):
        parser.error(
            "使用--resume时不能同时指定--run-name"
        )

    if args.resume is not None:
        resume_path = Path(args.resume)

        if not resume_path.exists():
            parser.error(
                f"checkpoint不存在：{resume_path}"
            )

        if not resume_path.is_file():
            parser.error(
                f"checkpoint必须是文件：{resume_path}"
            )

        if resume_path.suffix.lower() != ".pt":
            parser.error(
                "checkpoint文件必须使用.pt扩展名"
            )
def parse_args() -> argparse.Namespace:
    """读取并检查命令行中的训练参数."""
    

    parser = argparse.ArgumentParser(
        description="在CIFAR-10上训练PlainNet或ResNet"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=SUPPORTED_MODELS,
        help="需要训练的模型名称",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="训练轮数，预实验默认5轮",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="每个batch包含的图片数量",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="初始学习率",
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD动量",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
        help="权重衰减系数",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader子进程数量，Windows先使用0",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="实验名称；不指定时根据seed和epochs自动生成",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="checkpoint、CSV和结果的根目录",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs",
        help="TensorBoard日志根目录",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="训练使用的计算设备",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="short",
        choices=("short", "paper"),
        help="short用于epoch预实验，paper用于论文64k正式训练",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=64000,
        help="paper日程的最大参数更新次数",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从指定的.pt checkpoint继续训练",
    )
    args = parser.parse_args()
    validate_args(parser, args)
    
    return args
def build_run_name(args: argparse.Namespace) -> str:
    """生成本次实验的名称."""

    if args.run_name is not None:
        return args.run_name.strip()

    if args.schedule == "paper":
        return (
            f"seed{args.seed}_"
            f"paper{args.max_iterations // 1000}k"
        )

    return f"seed{args.seed}_{args.epochs}epochs"


def build_paper_schedule(args: argparse.Namespace) -> PaperSchedule:
    """根据命令行初始学习率构造论文iteration日程。"""

    return PaperSchedule(
        max_iterations=args.max_iterations,
        milestones=(32000, 48000),
        learning_rates=(
            args.learning_rate,
            args.learning_rate / 10,
            args.learning_rate / 100,
        ),
    )


def build_experiment_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    """生成本次实验的输出目录和TensorBoard目录."""

    if args.resume is not None:
        resume_path = Path(args.resume)
        output_path = resume_path.parent

        run_name = output_path.name
        log_path = (
            Path(args.log_dir)
            / args.model
            / run_name
        )

        return output_path, log_path

    run_name = build_run_name(args)

    output_path = (
        Path(args.output_dir)
        / args.model
        / run_name
    )

    log_path = (
        Path(args.log_dir)
        / args.model
        / run_name
    )

    return output_path, log_path
def build_experiment_config(
    args: argparse.Namespace,
    output_path: Path,
    log_path: Path,
) -> dict[str, object]:
    """整理本次实验需要记录的完整配置."""

    model_config = MODEL_CONFIGS[args.model]

    return {
        "model": {
            "name": args.model,
            "family": model_config["family"],
            "depth": model_config["depth"],
            "n": model_config["n"],
            "num_classes": 10,
        },
        "data": {
            "dataset": "cifar10",
            "preprocessing": "per_pixel_mean",
            "data_augmentation": True,
        },
        "training": {
            "schedule": args.schedule,
            "epochs": (
                args.epochs
                if args.schedule == "short"
                else None
            ),
            "max_iterations": (
                args.max_iterations
                if args.schedule == "paper"
                else None
            ),
            "learning_rate_milestones": (
                [32000, 48000]
                if args.schedule == "paper"
                else []
            ),
            "learning_rates": (
                list(build_paper_schedule(args).learning_rates)
                if args.schedule == "paper"
                else [args.learning_rate]
            ),
            "batch_size": args.batch_size,
            "optimizer": "sgd",
            "learning_rate": args.learning_rate,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "num_workers": args.num_workers,
            "device": args.device,
            "resume_from": (
                str(Path(args.resume))
                if args.resume is not None
                else None
            ),
        },
        "paths": {
            "output": str(output_path),
            "tensorboard": str(log_path),
        },
    }
def config_to_yaml(
    config: dict[str, object],
) -> str:
    """将实验配置转换成便于阅读的YAML文本."""

    return yaml.safe_dump(
        config,
        allow_unicode=True,
        sort_keys=False,
    )
def validate_config_round_trip(
    config: dict[str, object],
) -> None:
    """验证配置转换成YAML后可以无损读回."""

    yaml_text = config_to_yaml(config)
    loaded_config = yaml.safe_load(yaml_text)

    if loaded_config != config:
        raise ValueError(
            "配置经过YAML转换后内容发生了变化"
        )

def _make_progress_callback(
    description: str,
    total: int | None,
    *,
    enabled: bool,
) -> tuple[object | None, object | None]:
    """创建tqdm及其BatchProgress回调；关闭显示时返回两个None。"""

    if not enabled:
        return None, None
    progress_bar = tqdm(total=total, desc=description, unit="batch")

    def on_batch_end(progress: BatchProgress) -> None:
        progress_bar.update(1)
        progress_bar.set_postfix(
            loss=f"{progress.loss:.4f}",
            accuracy=f"{progress.accuracy * 100:.2f}%",
        )

    return progress_bar, on_batch_end


def _print_epoch_summary(
    epoch: int,
    target_epochs: int,
    row: dict[str, object],
) -> None:
    print(f"\nEpoch {epoch}/{target_epochs}")
    print(f"  train_loss: {float(row['train_loss']):.4f}")
    print(f"  train_accuracy: {float(row['train_accuracy']) * 100:.2f}%")
    print(f"  test_loss: {float(row['test_loss']):.4f}")
    print(f"  test_accuracy: {float(row['test_accuracy']) * 100:.2f}%")
    print(f"  learning_rate: {float(row['learning_rate']):.6f}")
    print(f"  elapsed: {float(row['elapsed_seconds']):.2f}s")
    print(
        "  best_test_accuracy: "
        f"{float(row['best_test_accuracy']) * 100:.2f}%"
    )


def run_training(
    args: argparse.Namespace,
    *,
    show_progress: bool = True,
    max_train_batches: int | None = None,
    max_test_batches: int | None = None,
) -> list[dict[str, object]]:
    """执行short日程训练，并保存每个完整epoch的全部实验状态。"""

    if args.schedule != "short":
        raise ValueError(
            "paper 64k iteration日程将在后续正式复现阶段实现"
        )

    device = resolve_device(args.device)
    set_random_seed(args.seed)
    output_path, log_path = build_experiment_paths(args)
    is_resume = args.resume is not None
    prepare_experiment_directories(
        output_path,
        log_path,
        resume=is_resume,
    )
    config = build_experiment_config(args, output_path, log_path)

    if is_resume:
        checkpoint = load_checkpoint(Path(args.resume))
        validate_resume_checkpoint(
            checkpoint,
            args.model,
            config,
            args.epochs,
        )
    else:
        checkpoint = None
        write_config(output_path / "config.yaml", config)

    train_loader, test_loader = create_cifar10_loaders(
        data_dir="data",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=True,
        pin_memory=device.type == "cuda",
    )
    model = create_model(args.model, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if checkpoint is None:
        start_epoch = 1
        best_test_accuracy = -1.0
        history: list[dict[str, object]] = []
    else:
        start_epoch, best_test_accuracy, history = restore_training_state(
            checkpoint,
            model,
            optimizer,
            device,
        )

    print(f"device: {device.type}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(device)}")
    print(f"output: {output_path}")
    print(f"tensorboard: {log_path}")

    writer = SummaryWriter(log_dir=str(log_path))
    try:
        if start_epoch > args.epochs:
            print("checkpoint已经完成目标epochs，无需重复训练")
            return history

        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.perf_counter()
            train_bar, train_callback = _make_progress_callback(
                f"Epoch {epoch}/{args.epochs} Train",
                len(train_loader) if hasattr(train_loader, "__len__") else None,
                enabled=show_progress,
            )
            try:
                train_metrics = train_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                    max_batches=max_train_batches,
                    on_batch_end=train_callback,
                )
            finally:
                if train_bar is not None:
                    train_bar.close()

            test_bar, test_callback = _make_progress_callback(
                f"Epoch {epoch}/{args.epochs} Test",
                len(test_loader) if hasattr(test_loader, "__len__") else None,
                enabled=show_progress,
            )
            try:
                test_metrics = evaluate(
                    model,
                    test_loader,
                    criterion,
                    device,
                    max_batches=max_test_batches,
                    on_batch_end=test_callback,
                )
            finally:
                if test_bar is not None:
                    test_bar.close()

            learning_rate = float(optimizer.param_groups[0]["lr"])
            elapsed_seconds = time.perf_counter() - epoch_start
            improved = test_metrics.accuracy > best_test_accuracy
            if improved:
                best_test_accuracy = test_metrics.accuracy
            row: dict[str, object] = {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "test_loss": test_metrics.loss,
                "test_accuracy": test_metrics.accuracy,
                "learning_rate": learning_rate,
                "elapsed_seconds": elapsed_seconds,
                "best_test_accuracy": best_test_accuracy,
            }
            history.append(row)
            write_history(output_path / "history.csv", history)
            write_tensorboard_metrics(writer, row)
            current_checkpoint = build_checkpoint(
                model_name=args.model,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_test_accuracy=best_test_accuracy,
                history=history,
                config=config,
            )
            atomic_save_checkpoint(
                output_path / "latest.pt",
                current_checkpoint,
            )
            if improved:
                atomic_save_checkpoint(
                    output_path / "best.pt",
                    current_checkpoint,
                )
            _print_epoch_summary(epoch, args.epochs, row)
    except KeyboardInterrupt:
        completed_epochs = len(history)
        print(f"\n训练已中断；已完整保存 {completed_epochs} 个epoch。")
        if completed_epochs:
            print(
                "恢复命令：python train.py "
                f"--model {args.model} --epochs {args.epochs} "
                f'--resume "{output_path / "latest.pt"}"'
            )
        else:
            print("第一轮尚未完成，没有checkpoint，请重新开始本次实验。")
    finally:
        writer.close()

    if history and len(history) >= args.epochs:
        print("\n训练完成")
        print(f"最佳测试准确率: {best_test_accuracy * 100:.2f}%")
        print(f"best.pt: {output_path / 'best.pt'}")
        print(f"latest.pt: {output_path / 'latest.pt'}")
        print(f"history.csv: {output_path / 'history.csv'}")
        print(f"TensorBoard日志: {log_path}")
    return history


def main() -> None:
    """解析命令行并启动短日程训练。"""

    run_training(parse_args())
   


if __name__ == "__main__":
   main()
    

    

