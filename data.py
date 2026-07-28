"""CIFAR-10 数据预处理、Dataset 与 DataLoader 构建工具。"""
from collections.abc import Sized
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def validate_per_pixel_mean(mean_image: torch.Tensor) -> None:
    """校验逐像素平均图像。"""

    if not isinstance(mean_image, torch.Tensor):
        raise TypeError("mean_image必须是Tensor")
    if mean_image.shape != (3, 32, 32):
        raise ValueError(
            "mean_image形状必须是(3, 32, 32)，"
            f"实际收到{tuple(mean_image.shape)}"
        )
    if not mean_image.is_floating_point():
        raise ValueError("mean_image必须是浮点Tensor")
    if not torch.isfinite(mean_image).all():
        raise ValueError("mean_image必须全部为有限数")
    if torch.any(mean_image < 0) or torch.any(mean_image > 1):
        raise ValueError("mean_image数值必须位于[0, 1]")


def compute_per_pixel_mean(dataset: Dataset) -> torch.Tensor:
    """从未增强的训练集计算CIFAR-10逐像素平均图像。"""

    if not isinstance(dataset, Sized):
        raise TypeError("dataset必须支持len()")
    if len(dataset) == 0:
        raise ValueError("数据集不能为空")

    total = torch.zeros((3, 32, 32), dtype=torch.float64)

    for sample in dataset:
        if not isinstance(sample, (tuple, list)) or len(sample) != 2:
            raise ValueError("数据样本必须是(image, label)")
        image, _ = sample
        if not isinstance(image, torch.Tensor):
            raise TypeError("图片必须是Tensor")
        if image.shape != (3, 32, 32):
            raise ValueError(
                "图片形状必须是(3, 32, 32)，"
                f"实际收到{tuple(image.shape)}"
            )
        if not image.is_floating_point():
            raise ValueError("图片必须是浮点Tensor")
        if not torch.isfinite(image).all():
            raise ValueError("图片必须全部为有限数")
        if torch.any(image < 0) or torch.any(image > 1):
            raise ValueError("原始图片数值必须位于[0, 1]")
        total += image.to(dtype=torch.float64)

    mean_image = (total / len(dataset)).to(dtype=torch.float32)
    validate_per_pixel_mean(mean_image)
    return mean_image


def load_or_compute_per_pixel_mean(
    raw_train_dataset: Dataset,
    cache_path: str | Path,
) -> torch.Tensor:
    """读取平均图像缓存；缓存不存在时计算并保存。"""

    resolved_cache_path = Path(cache_path)

    if resolved_cache_path.exists():
        mean_image = torch.load(
            resolved_cache_path,
            map_location="cpu",
            weights_only=True,
        )
        validate_per_pixel_mean(mean_image)
        return mean_image

    mean_image = compute_per_pixel_mean(raw_train_dataset)
    resolved_cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mean_image, resolved_cache_path)
    return mean_image


class SubtractPerPixelMean:
    """从CIFAR图片Tensor中减去训练集逐像素平均图像。"""

    def __init__(self, mean_image: torch.Tensor) -> None:
        validate_per_pixel_mean(mean_image)
        self.mean_image = mean_image.detach().clone()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError("image必须是Tensor")
        if image.shape != (3, 32, 32):
            raise ValueError(
                "image形状必须是(3, 32, 32)，"
                f"实际收到{tuple(image.shape)}"
            )
        if not image.is_floating_point():
            raise ValueError("image必须是浮点Tensor")

        return image - self.mean_image.to(
            device=image.device,
            dtype=image.dtype,
        )


def build_cifar10_transforms(
    mean_image: torch.Tensor,
) -> tuple[transforms.Compose, transforms.Compose]:
    """创建论文式CIFAR-10训练和测试预处理。"""

    validate_per_pixel_mean(mean_image)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            SubtractPerPixelMean(mean_image),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            SubtractPerPixelMean(mean_image),
        ]
    )
    return train_transform, test_transform


def create_cifar10_datasets(
    data_dir: str | Path,
    download: bool = True,
) -> tuple[Dataset, Dataset]:
    """创建共享训练集逐像素均值的正式训练和测试Dataset。"""

    if not isinstance(download, bool):
        raise TypeError("download必须是布尔值")

    resolved_data_dir = Path(data_dir)
    raw_train_dataset = datasets.CIFAR10(
        root=str(resolved_data_dir),
        train=True,
        transform=transforms.ToTensor(),
        download=download,
    )
    mean_image = load_or_compute_per_pixel_mean(
        raw_train_dataset,
        resolved_data_dir / "cifar10_per_pixel_mean.pt",
    )
    train_transform, test_transform = build_cifar10_transforms(mean_image)

    train_dataset = datasets.CIFAR10(
        root=str(resolved_data_dir),
        train=True,
        transform=train_transform,
        download=download,
    )
    test_dataset = datasets.CIFAR10(
        root=str(resolved_data_dir),
        train=False,
        transform=test_transform,
        download=download,
    )
    return train_dataset, test_dataset


def create_cifar10_loaders(
    data_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 0,
    download: bool = True,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader]:
    """创建论文复现实验使用的训练和测试DataLoader。"""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size必须是整数")
    if batch_size <= 0:
        raise ValueError("batch_size必须大于0")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise TypeError("num_workers必须是整数")
    if num_workers < 0:
        raise ValueError("num_workers不能小于0")
    if not isinstance(download, bool):
        raise TypeError("download必须是布尔值")
    if pin_memory is not None and not isinstance(pin_memory, bool):
        raise TypeError("pin_memory必须是布尔值或None")

    resolved_pin_memory = (
        torch.cuda.is_available()
        if pin_memory is None
        else pin_memory
    )
    train_dataset, test_dataset = create_cifar10_datasets(
        data_dir,
        download=download,
    )
    common_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": resolved_pin_memory,
        "drop_last": False,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **common_options,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_options,
    )
    return train_loader, test_loader
