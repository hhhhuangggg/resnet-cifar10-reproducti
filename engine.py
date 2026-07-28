"""模型训练、评估与分类指标统计工具。"""
from dataclasses import dataclass
from collections.abc import Callable, Iterable

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass(frozen=True)
class EpochMetrics:
    """一次训练或评估遍历得到的不可修改指标。"""

    loss: float
    accuracy: float
    correct: int
    samples: int
    batches: int


@dataclass(frozen=True)
class BatchProgress:
    """一个epoch内处理完当前batch后的累计指标快照。"""

    batch: int
    loss: float
    accuracy: float
    correct: int
    samples: int


def move_batch_to_device(
    batch: object,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """校验分类batch并把图片和标签移动到指定设备。"""

    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise TypeError("batch必须是(images, labels)")
    images, labels = batch
    if not isinstance(images, torch.Tensor):
        raise TypeError("images必须是Tensor")
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels必须是Tensor")
    if images.ndim != 4:
        raise ValueError("images必须是四维[N,3,H,W]")
    if images.shape[1] != 3:
        raise ValueError("images通道数必须为3")
    if not images.is_floating_point():
        raise ValueError("images必须是浮点Tensor")
    if labels.ndim != 1:
        raise ValueError("labels必须是一维[N]")
    if labels.dtype != torch.int64:
        raise ValueError("labels必须是torch.int64")
    if images.shape[0] != labels.shape[0]:
        raise ValueError("images和labels样本数必须相同")
    if images.shape[0] == 0:
        raise ValueError("batch不能为空")

    resolved_device = torch.device(device)
    return (
        images.to(resolved_device, non_blocking=True),
        labels.to(resolved_device, non_blocking=True),
    )


def compute_batch_statistics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[int, int]:
    """统计分类logits中的正确预测数和样本数。"""

    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits必须是Tensor")
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels必须是Tensor")
    if logits.ndim != 2:
        raise ValueError("logits必须是二维[N,C]")
    if logits.shape[0] == 0:
        raise ValueError("logits不能为空")
    if logits.shape[1] <= 1:
        raise ValueError("分类数C必须大于1")
    if not logits.is_floating_point():
        raise ValueError("logits必须是浮点Tensor")
    if not torch.isfinite(logits).all():
        raise ValueError("logits必须全部为有限数")
    if labels.ndim != 1:
        raise ValueError("labels必须是一维[N]")
    if labels.dtype != torch.int64:
        raise ValueError("labels必须是torch.int64")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("logits和labels样本数必须相同")
    if logits.device != labels.device:
        raise ValueError("logits和labels必须位于同一设备")
    if torch.any(labels < 0) or torch.any(labels >= logits.shape[1]):
        raise ValueError("labels超出logits类别范围")

    predictions = logits.argmax(dim=1)
    correct = int((predictions == labels).sum().item())
    return correct, int(labels.shape[0])


def _validate_max_batches(max_batches: int | None) -> None:
    """校验可选的batch数量限制。"""

    if max_batches is None:
        return
    if isinstance(max_batches, bool) or not isinstance(max_batches, int):
        raise TypeError("max_batches必须是正整数或None")
    if max_batches <= 0:
        raise ValueError("max_batches必须大于0")


def _validate_loss(
    loss: torch.Tensor,
    *,
    require_grad: bool,
) -> None:
    """校验criterion产生的标量loss。"""

    if not isinstance(loss, torch.Tensor):
        raise TypeError("criterion必须返回Tensor")
    if loss.numel() != 1:
        raise ValueError("loss必须只包含一个元素")
    if not loss.is_floating_point():
        raise ValueError("loss必须是浮点Tensor")
    if not torch.isfinite(loss).all():
        raise FloatingPointError("loss出现NaN或无穷大")
    if require_grad and not loss.requires_grad:
        raise ValueError("训练loss必须能够反向传播")


def _build_metrics(
    loss_sum: float,
    correct: int,
    samples: int,
    batches: int,
) -> EpochMetrics:
    """将累加量整理成最终指标。"""

    if samples == 0 or batches == 0:
        raise ValueError("DataLoader没有产生任何batch")
    return EpochMetrics(
        loss=loss_sum / samples,
        accuracy=correct / samples,
        correct=correct,
        samples=samples,
        batches=batches,
    )


def train_one_epoch(
    model: nn.Module,
    data_loader: Iterable,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: str | torch.device,
    max_batches: int | None = None,
    on_batch_end: Callable[[BatchProgress], None] | None = None,
) -> EpochMetrics:
    """训练一个完整epoch，或最多训练指定数量的batch。"""

    _validate_max_batches(max_batches)
    model.train()
    loss_sum = 0.0
    correct_sum = 0
    sample_sum = 0
    batch_count = 0

    for batch in data_loader:
        images, labels = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        correct, samples = compute_batch_statistics(logits, labels)
        loss = criterion(logits, labels)
        _validate_loss(loss, require_grad=True)
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.item()) * samples
        correct_sum += correct
        sample_sum += samples
        batch_count += 1

        if on_batch_end is not None:
            on_batch_end(
                BatchProgress(
                    batch=batch_count,
                    loss=loss_sum / sample_sum,
                    accuracy=correct_sum / sample_sum,
                    correct=correct_sum,
                    samples=sample_sum,
                )
            )

        if max_batches is not None and batch_count >= max_batches:
            break

    return _build_metrics(
        loss_sum,
        correct_sum,
        sample_sum,
        batch_count,
    )


def evaluate(
    model: nn.Module,
    data_loader: Iterable,
    criterion: nn.Module,
    device: str | torch.device,
    max_batches: int | None = None,
    on_batch_end: Callable[[BatchProgress], None] | None = None,
) -> EpochMetrics:
    """在不计算梯度的情况下评估完整数据或有限batch。"""

    _validate_max_batches(max_batches)
    model.eval()
    loss_sum = 0.0
    correct_sum = 0
    sample_sum = 0
    batch_count = 0

    with torch.no_grad():
        for batch in data_loader:
            images, labels = move_batch_to_device(batch, device)
            logits = model(images)
            correct, samples = compute_batch_statistics(logits, labels)
            loss = criterion(logits, labels)
            _validate_loss(loss, require_grad=False)

            loss_sum += float(loss.item()) * samples
            correct_sum += correct
            sample_sum += samples
            batch_count += 1

            if on_batch_end is not None:
                on_batch_end(
                    BatchProgress(
                        batch=batch_count,
                        loss=loss_sum / sample_sum,
                        accuracy=correct_sum / sample_sum,
                        correct=correct_sum,
                        samples=sample_sum,
                    )
                )

            if max_batches is not None and batch_count >= max_batches:
                break

    return _build_metrics(
        loss_sum,
        correct_sum,
        sample_sum,
        batch_count,
    )
