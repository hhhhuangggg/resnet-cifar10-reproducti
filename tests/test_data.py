"""CIFAR-10 数据管道测试。"""

import importlib.util
import importlib
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import (
    Dataset,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)
from torchvision import datasets, transforms

import data as data_module


def test_data_module_exists() -> None:
    """项目应当提供独立的数据处理模块。"""

    assert importlib.util.find_spec("data") is not None


def test_validate_per_pixel_mean_exists() -> None:
    """数据模块应提供逐像素平均图像校验函数。"""

    data_module = importlib.import_module("data")

    assert hasattr(data_module, "validate_per_pixel_mean")


def test_validate_per_pixel_mean_accepts_valid_tensor() -> None:
    """合法的CIFAR-10逐像素平均图像应通过校验。"""

    data_module.validate_per_pixel_mean(
        torch.full((3, 32, 32), 0.5, dtype=torch.float32)
    )


@pytest.mark.parametrize(
    ("mean_image", "exception_type"),
    [
        ("not-a-tensor", TypeError),
        (torch.zeros(3), ValueError),
        (torch.zeros(1, 3, 32, 32), ValueError),
        (torch.zeros(3, 32, 32, dtype=torch.int64), ValueError),
        (
            torch.full((3, 32, 32), float("nan")),
            ValueError,
        ),
        (
            torch.full((3, 32, 32), float("inf")),
            ValueError,
        ),
        (torch.full((3, 32, 32), -0.1), ValueError),
        (torch.full((3, 32, 32), 1.1), ValueError),
    ],
)
def test_validate_per_pixel_mean_rejects_invalid_values(
    mean_image: object,
    exception_type: type[Exception],
) -> None:
    """平均图像的类型、形状、dtype和数值都必须合法。"""

    with pytest.raises(exception_type):
        data_module.validate_per_pixel_mean(mean_image)


class TensorImageDataset(Dataset):
    """测试用的小型Tensor图片数据集。"""

    def __init__(self, images: list[torch.Tensor]) -> None:
        self.images = images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], index


def test_compute_per_pixel_mean_averages_training_images() -> None:
    """平均图像应在样本维度对每个通道和位置求平均。"""

    dataset = TensorImageDataset(
        [
            torch.full((3, 32, 32), 0.2),
            torch.full((3, 32, 32), 0.6),
        ]
    )

    mean_image = data_module.compute_per_pixel_mean(dataset)

    assert mean_image.shape == (3, 32, 32)
    assert mean_image.dtype == torch.float32
    assert torch.allclose(mean_image, torch.full_like(mean_image, 0.4))


def test_compute_per_pixel_mean_rejects_empty_dataset() -> None:
    """空数据集无法产生有意义的平均图像。"""

    with pytest.raises(ValueError, match="数据集不能为空"):
        data_module.compute_per_pixel_mean(TensorImageDataset([]))


def test_compute_per_pixel_mean_rejects_wrong_image_shape() -> None:
    """平均图像只能由CIFAR尺寸的RGB Tensor计算。"""

    dataset = TensorImageDataset([torch.zeros(3, 16, 16)])

    with pytest.raises(ValueError, match="图片形状"):
        data_module.compute_per_pixel_mean(dataset)


def test_mean_cache_computes_and_saves_when_missing(
    tmp_path: Path,
) -> None:
    """首次运行应计算平均图像并写入缓存。"""

    dataset = TensorImageDataset(
        [
            torch.full((3, 32, 32), 0.25),
            torch.full((3, 32, 32), 0.75),
        ]
    )
    cache_path = tmp_path / "nested" / "mean.pt"

    mean_image = data_module.load_or_compute_per_pixel_mean(
        dataset,
        cache_path,
    )

    assert cache_path.is_file()
    assert torch.allclose(mean_image, torch.full_like(mean_image, 0.5))
    assert torch.equal(
        torch.load(cache_path, weights_only=True),
        mean_image,
    )


class UnreadableDataset(Dataset):
    """一旦读取就失败，用来证明缓存路径没有重复计算。"""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        raise AssertionError("缓存存在时不应读取dataset")


def test_mean_cache_loads_existing_file_without_dataset_access(
    tmp_path: Path,
) -> None:
    """缓存存在时应直接读取，不再遍历训练集。"""

    cache_path = tmp_path / "mean.pt"
    expected = torch.full((3, 32, 32), 0.4)
    torch.save(expected, cache_path)

    actual = data_module.load_or_compute_per_pixel_mean(
        UnreadableDataset(),
        cache_path,
    )

    assert torch.equal(actual, expected)


def test_mean_cache_rejects_corrupt_tensor(tmp_path: Path) -> None:
    """形状错误的缓存不能进入后续数据处理。"""

    cache_path = tmp_path / "mean.pt"
    torch.save(torch.zeros(3), cache_path)

    with pytest.raises(ValueError, match="mean_image形状"):
        data_module.load_or_compute_per_pixel_mean(
            UnreadableDataset(),
            cache_path,
        )


def test_subtract_per_pixel_mean_returns_centered_copy() -> None:
    """逐像素变换应执行image-mean且不修改输入。"""

    mean_image = torch.full((3, 32, 32), 0.25)
    transform = data_module.SubtractPerPixelMean(mean_image)
    image = torch.full((3, 32, 32), 0.75)
    original = image.clone()

    output = transform(image)

    assert torch.equal(output, torch.full_like(output, 0.5))
    assert torch.equal(image, original)


def test_subtract_per_pixel_mean_keeps_private_mean_copy() -> None:
    """外部Tensor被修改后不应改变变换内部的平均图像。"""

    mean_image = torch.full((3, 32, 32), 0.25)
    transform = data_module.SubtractPerPixelMean(mean_image)
    mean_image.fill_(1.0)

    output = transform(torch.full((3, 32, 32), 0.75))

    assert torch.equal(output, torch.full_like(output, 0.5))


@pytest.mark.parametrize(
    "image",
    [
        "not-a-tensor",
        torch.zeros(3),
        torch.zeros(3, 32, 32, dtype=torch.int64),
    ],
)
def test_subtract_per_pixel_mean_rejects_invalid_images(
    image: object,
) -> None:
    """均值减法只接受浮点CIFAR图片Tensor。"""

    transform = data_module.SubtractPerPixelMean(
        torch.zeros(3, 32, 32)
    )

    with pytest.raises((TypeError, ValueError)):
        transform(image)


def test_build_cifar10_transforms_has_paper_order() -> None:
    """训练和测试预处理的组成与顺序应符合设计。"""

    train_transform, test_transform = (
        data_module.build_cifar10_transforms(
            torch.full((3, 32, 32), 0.5)
        )
    )

    assert isinstance(train_transform, transforms.Compose)
    assert isinstance(test_transform, transforms.Compose)
    assert [type(step) for step in train_transform.transforms] == [
        transforms.RandomCrop,
        transforms.RandomHorizontalFlip,
        transforms.ToTensor,
        data_module.SubtractPerPixelMean,
    ]
    assert [type(step) for step in test_transform.transforms] == [
        transforms.ToTensor,
        data_module.SubtractPerPixelMean,
    ]


def test_cifar10_transforms_keep_image_shape() -> None:
    """训练增强和测试预处理都应输出标准CIFAR Tensor。"""

    train_transform, test_transform = (
        data_module.build_cifar10_transforms(
            torch.full((3, 32, 32), 0.5)
        )
    )
    image = Image.new("RGB", (32, 32), color=(128, 64, 32))

    assert train_transform(image).shape == (3, 32, 32)
    assert test_transform(image).shape == (3, 32, 32)


def test_test_transform_is_deterministic() -> None:
    """测试图片重复预处理应得到完全相同的结果。"""

    _, test_transform = data_module.build_cifar10_transforms(
        torch.full((3, 32, 32), 0.5)
    )
    image = Image.new("RGB", (32, 32), color=(128, 64, 32))

    first = test_transform(image)
    second = test_transform(image)

    assert torch.equal(first, second)


class FakeCIFAR10(Dataset):
    """记录CIFAR10构造参数并返回两张内存图片。"""

    calls: list[dict[str, object]] = []

    def __init__(
        self,
        root: str,
        train: bool,
        transform: object,
        download: bool,
    ) -> None:
        self.train = train
        self.transform = transform
        self.calls.append(
            {
                "root": root,
                "train": train,
                "transform": transform,
                "download": download,
            }
        )
        self.images = [
            Image.new("RGB", (32, 32), color=(64, 128, 192)),
            Image.new("RGB", (32, 32), color=(192, 128, 64)),
        ]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[object, int]:
        image: object = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, index


def test_create_cifar10_datasets_builds_raw_train_and_formal_sets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """应先创建原始训练集，再创建正式训练集和测试集。"""

    FakeCIFAR10.calls.clear()
    monkeypatch.setattr(datasets, "CIFAR10", FakeCIFAR10)

    train_dataset, test_dataset = data_module.create_cifar10_datasets(
        tmp_path,
        download=False,
    )

    assert len(FakeCIFAR10.calls) == 3
    raw_call, train_call, test_call = FakeCIFAR10.calls
    assert raw_call["train"] is True
    assert isinstance(raw_call["transform"], transforms.ToTensor)
    assert train_call["train"] is True
    assert test_call["train"] is False
    assert all(call["download"] is False for call in FakeCIFAR10.calls)
    assert train_dataset is not test_dataset
    assert (tmp_path / "cifar10_per_pixel_mean.pt").is_file()


def test_create_cifar10_datasets_shares_training_mean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式训练集和测试集必须减去相同的训练集平均图像。"""

    FakeCIFAR10.calls.clear()
    monkeypatch.setattr(datasets, "CIFAR10", FakeCIFAR10)

    train_dataset, test_dataset = data_module.create_cifar10_datasets(
        tmp_path,
        download=False,
    )
    train_mean = train_dataset.transform.transforms[-1].mean_image
    test_mean = test_dataset.transform.transforms[-1].mean_image

    assert torch.equal(train_mean, test_mean)
    assert train_mean.shape == (3, 32, 32)


def test_create_cifar10_datasets_returns_valid_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式Dataset应返回中心化Tensor与整数标签。"""

    FakeCIFAR10.calls.clear()
    monkeypatch.setattr(datasets, "CIFAR10", FakeCIFAR10)
    train_dataset, test_dataset = data_module.create_cifar10_datasets(
        tmp_path,
        download=False,
    )

    train_image, train_label = train_dataset[0]
    test_image, test_label = test_dataset[0]

    assert train_image.shape == (3, 32, 32)
    assert test_image.shape == (3, 32, 32)
    assert train_image.dtype == torch.float32
    assert test_image.dtype == torch.float32
    assert isinstance(train_label, int)
    assert isinstance(test_label, int)


def _small_tensor_datasets() -> tuple[TensorDataset, TensorDataset]:
    train = TensorDataset(
        torch.randn(10, 3, 32, 32),
        torch.arange(10, dtype=torch.int64) % 10,
    )
    test = TensorDataset(
        torch.randn(6, 3, 32, 32),
        torch.arange(6, dtype=torch.int64),
    )
    return train, test


def test_create_cifar10_loaders_builds_expected_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DataLoader应将单张样本堆叠成图片和标签batch。"""

    monkeypatch.setattr(
        data_module,
        "create_cifar10_datasets",
        lambda *args, **kwargs: _small_tensor_datasets(),
    )

    train_loader, test_loader = data_module.create_cifar10_loaders(
        "unused",
        batch_size=4,
        num_workers=0,
        download=False,
        pin_memory=False,
    )
    train_images, train_labels = next(iter(train_loader))
    test_images, test_labels = next(iter(test_loader))

    assert train_images.shape == (4, 3, 32, 32)
    assert train_labels.shape == (4,)
    assert test_images.shape == (4, 3, 32, 32)
    assert test_labels.shape == (4,)
    assert train_images.dtype == torch.float32
    assert train_labels.dtype == torch.int64


def test_create_cifar10_loaders_uses_correct_sampling_and_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """训练集应打乱，测试集应固定，并保留最后不足一批的数据。"""

    monkeypatch.setattr(
        data_module,
        "create_cifar10_datasets",
        lambda *args, **kwargs: _small_tensor_datasets(),
    )

    train_loader, test_loader = data_module.create_cifar10_loaders(
        "unused",
        batch_size=4,
        num_workers=0,
        download=False,
        pin_memory=True,
    )

    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)
    assert train_loader.drop_last is False
    assert test_loader.drop_last is False
    assert train_loader.pin_memory is True
    assert test_loader.pin_memory is True
    assert len(list(train_loader)) == 3
    assert len(list(test_loader)) == 2
    assert list(train_loader)[-1][0].shape[0] == 2
    assert list(test_loader)[-1][0].shape[0] == 2


def test_create_cifar10_loaders_auto_detects_pin_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pin_memory=None时应根据CUDA可用性自动决定。"""

    monkeypatch.setattr(
        data_module,
        "create_cifar10_datasets",
        lambda *args, **kwargs: _small_tensor_datasets(),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    train_loader, test_loader = data_module.create_cifar10_loaders(
        "unused",
        pin_memory=None,
    )

    assert train_loader.pin_memory is True
    assert test_loader.pin_memory is True


@pytest.mark.parametrize(
    ("overrides", "exception_type"),
    [
        ({"batch_size": 0}, ValueError),
        ({"batch_size": 1.5}, TypeError),
        ({"batch_size": True}, TypeError),
        ({"num_workers": -1}, ValueError),
        ({"num_workers": 1.5}, TypeError),
        ({"num_workers": True}, TypeError),
        ({"pin_memory": "yes"}, TypeError),
        ({"download": "yes"}, TypeError),
    ],
)
def test_create_cifar10_loaders_rejects_invalid_arguments(
    overrides: dict[str, object],
    exception_type: type[Exception],
) -> None:
    """DataLoader配置必须具有正确类型和数值范围。"""

    with pytest.raises(exception_type):
        data_module.create_cifar10_loaders(
            "unused",
            **overrides,
        )
