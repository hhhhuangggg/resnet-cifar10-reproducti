"""训练与评估核心引擎测试。"""

import importlib.util
from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import engine

def test_engine_module_exists() -> None:
    """项目应当提供独立训练引擎模块。"""

    assert importlib.util.find_spec("engine") is not None


def test_epoch_metrics_is_immutable() -> None:
    """已完成的指标记录不应被意外改写。"""

    metrics = engine.EpochMetrics(
        loss=1.2,
        accuracy=0.5,
        correct=2,
        samples=4,
        batches=1,
    )

    with pytest.raises(FrozenInstanceError):
        metrics.loss = 0.0


def test_batch_progress_is_immutable() -> None:
    """Batch进度是只读快照，回调不能改写引擎统计。"""

    progress = engine.BatchProgress(
        batch=1,
        loss=1.0,
        accuracy=0.5,
        correct=2,
        samples=4,
    )

    with pytest.raises(FrozenInstanceError):
        progress.batch = 2


def test_move_batch_to_device_preserves_values_and_shapes() -> None:
    """合法图片和标签应完整移动到目标设备。"""

    images = torch.randn(2, 3, 8, 8)
    labels = torch.tensor([0, 1], dtype=torch.int64)

    moved_images, moved_labels = engine.move_batch_to_device(
        (images, labels),
        "cpu",
    )

    assert moved_images.device.type == "cpu"
    assert moved_labels.device.type == "cpu"
    assert torch.equal(moved_images, images)
    assert torch.equal(moved_labels, labels)


@pytest.mark.parametrize(
    "batch",
    [
        torch.zeros(2),
        (torch.zeros(2),),
        ("images", torch.zeros(2, dtype=torch.int64)),
        (torch.zeros(2, 8, 8), torch.zeros(2, dtype=torch.int64)),
        (torch.zeros(2, 1, 8, 8), torch.zeros(2, dtype=torch.int64)),
        (
            torch.zeros(2, 3, 8, 8, dtype=torch.int64),
            torch.zeros(2, dtype=torch.int64),
        ),
        (torch.zeros(2, 3, 8, 8), torch.zeros(2, 1, dtype=torch.int64)),
        (torch.zeros(2, 3, 8, 8), torch.zeros(2)),
        (torch.zeros(2, 3, 8, 8), torch.zeros(3, dtype=torch.int64)),
        (torch.zeros(0, 3, 8, 8), torch.zeros(0, dtype=torch.int64)),
    ],
)
def test_move_batch_to_device_rejects_invalid_batch(
    batch: object,
) -> None:
    """错误结构、形状、dtype或空batch必须被拒绝。"""

    with pytest.raises((TypeError, ValueError)):
        engine.move_batch_to_device(batch, "cpu")


def test_compute_batch_statistics_counts_known_predictions() -> None:
    """argmax预测应与标签逐项比较并返回正确数和样本数。"""

    logits = torch.tensor(
        [
            [5.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [0.0, 1.0, 4.0],
        ]
    )
    labels = torch.tensor([0, 2, 2], dtype=torch.int64)

    correct, samples = engine.compute_batch_statistics(logits, labels)

    assert correct == 2
    assert samples == 3
    assert isinstance(correct, int)
    assert isinstance(samples, int)


@pytest.mark.parametrize(
    ("logits", "labels"),
    [
        (torch.zeros(3), torch.zeros(3, dtype=torch.int64)),
        (torch.zeros(3, 1), torch.zeros(3, dtype=torch.int64)),
        (torch.zeros(0, 3), torch.zeros(0, dtype=torch.int64)),
        (torch.zeros(3, 3, dtype=torch.int64), torch.zeros(3, dtype=torch.int64)),
        (torch.zeros(3, 3), torch.zeros(3, 1, dtype=torch.int64)),
        (torch.zeros(3, 3), torch.zeros(3)),
        (torch.zeros(3, 3), torch.zeros(2, dtype=torch.int64)),
        (
            torch.full((3, 3), float("nan")),
            torch.zeros(3, dtype=torch.int64),
        ),
        (torch.zeros(3, 3), torch.tensor([0, 1, 3], dtype=torch.int64)),
    ],
)
def test_compute_batch_statistics_rejects_invalid_inputs(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    """预测统计应拒绝错误形状、dtype、数值和标签范围。"""

    with pytest.raises((TypeError, ValueError)):
        engine.compute_batch_statistics(logits, labels)


@pytest.mark.parametrize("value", [None, 1, 5])
def test_validate_max_batches_accepts_valid_values(
    value: int | None,
) -> None:
    """None和正整数是合法batch限制。"""

    engine._validate_max_batches(value)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "5"])
def test_validate_max_batches_rejects_invalid_values(
    value: object,
) -> None:
    """max_batches必须是None或正整数，bool不视为整数。"""

    with pytest.raises((TypeError, ValueError)):
        engine._validate_max_batches(value)


class TinyClassifier(nn.Module):
    """测试训练引擎用的轻量图片分类模型。"""

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 4 * 4, 3)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fc(self.flatten(images))


def _make_loader(
    sample_count: int = 10,
    batch_size: int = 4,
) -> DataLoader:
    generator = torch.Generator().manual_seed(7)
    images = torch.randn(
        sample_count,
        3,
        4,
        4,
        generator=generator,
    )
    labels = torch.arange(sample_count, dtype=torch.int64) % 3
    return DataLoader(
        TensorDataset(images, labels),
        batch_size=batch_size,
        shuffle=False,
    )


def test_train_one_epoch_updates_parameters_and_respects_limit() -> None:
    """训练应更新参数，并在指定batch数后停止。"""

    torch.manual_seed(11)
    model = TinyClassifier()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    metrics = engine.train_one_epoch(
        model,
        _make_loader(),
        nn.CrossEntropyLoss(),
        optimizer,
        "cpu",
        max_batches=2,
    )

    after = list(model.parameters())
    assert model.training is True
    assert metrics.batches == 2
    assert metrics.samples == 8
    assert metrics.correct <= metrics.samples
    assert 0.0 <= metrics.accuracy <= 1.0
    assert torch.isfinite(torch.tensor(metrics.loss))
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, after)
    )


def test_train_one_epoch_none_processes_all_batches() -> None:
    """max_batches=None应遍历包括最后小batch在内的全部数据。"""

    model = TinyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    metrics = engine.train_one_epoch(
        model,
        _make_loader(sample_count=10, batch_size=4),
        nn.CrossEntropyLoss(),
        optimizer,
        "cpu",
        max_batches=None,
    )

    assert metrics.batches == 3
    assert metrics.samples == 10


class BatchSizeLoss(nn.Module):
    """返回当前batch大小，用于验证按样本加权平均。"""

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return logits.sum() * 0 + float(labels.shape[0])


def test_train_one_epoch_weights_loss_by_sample_count() -> None:
    """大小不同的batch必须按样本数而非batch数平均。"""

    model = TinyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    metrics = engine.train_one_epoch(
        model,
        _make_loader(sample_count=3, batch_size=2),
        BatchSizeLoss(),
        optimizer,
        "cpu",
    )

    assert metrics.loss == pytest.approx(5 / 3)


@pytest.mark.parametrize("operation", ["train", "evaluate"])
def test_epoch_operation_reports_running_batch_progress(
    operation: str,
) -> None:
    """训练和评估应逐batch报告从epoch开头累计的运行指标。"""

    reports: list[engine.BatchProgress] = []
    model = TinyClassifier()
    loader = _make_loader(sample_count=10, batch_size=4)
    criterion = BatchSizeLoss()

    if operation == "train":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        metrics = engine.train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            "cpu",
            on_batch_end=reports.append,
        )
    else:
        metrics = engine.evaluate(
            model,
            loader,
            criterion,
            "cpu",
            on_batch_end=reports.append,
        )

    assert [item.batch for item in reports] == [1, 2, 3]
    assert [item.samples for item in reports] == [4, 8, 10]
    assert reports[-1].loss == pytest.approx(metrics.loss)
    assert reports[-1].accuracy == pytest.approx(metrics.accuracy)
    assert reports[-1].correct == metrics.correct


def test_train_one_epoch_rejects_empty_loader() -> None:
    """没有产生任何样本时不能返回除零指标。"""

    model = TinyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with pytest.raises(ValueError, match="没有产生任何batch"):
        engine.train_one_epoch(
            model,
            [],
            nn.CrossEntropyLoss(),
            optimizer,
            "cpu",
        )


def test_evaluate_preserves_parameters_and_disables_gradients() -> None:
    """评估应切换模式，且不创建梯度或修改模型参数。"""

    model = TinyClassifier()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.grad = None

    metrics = engine.evaluate(
        model,
        _make_loader(),
        nn.CrossEntropyLoss(),
        "cpu",
        max_batches=2,
    )

    assert model.training is False
    assert metrics.batches == 2
    assert metrics.samples == 8
    assert torch.isfinite(torch.tensor(metrics.loss))
    assert 0.0 <= metrics.accuracy <= 1.0
    assert all(
        torch.equal(old, new)
        for old, new in zip(before, model.parameters())
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_evaluate_none_processes_all_batches() -> None:
    """评估max_batches=None应处理最后一个小batch。"""

    metrics = engine.evaluate(
        TinyClassifier(),
        _make_loader(sample_count=10, batch_size=4),
        nn.CrossEntropyLoss(),
        "cpu",
    )

    assert metrics.batches == 3
    assert metrics.samples == 10


def test_evaluate_rejects_empty_loader() -> None:
    """空评估集不能产生有意义的指标。"""

    with pytest.raises(ValueError, match="没有产生任何batch"):
        engine.evaluate(
            TinyClassifier(),
            [],
            nn.CrossEntropyLoss(),
            "cpu",
        )
