"""Verify that a clean clone can construct the reproduction models."""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from models import create_model
from models.imagenet_resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)


CIFAR_MODEL_NAMES = ("plain20", "resnet20", "plain56", "resnet56")
IMAGENET_FACTORIES: dict[str, Callable[..., nn.Module]] = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}


@dataclass(frozen=True)
class CheckResult:
    """One successful model forward check."""

    name: str
    shape: tuple[int, ...]
    device: str


def resolve_device(device_name: str) -> torch.device:
    """Resolve an explicit or automatic PyTorch device."""

    normalized = device_name.strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError("--device 只能是 auto、cpu 或 cuda")
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 无法使用 CUDA")
    return torch.device(normalized)


def _forward_check(
    name: str,
    model: nn.Module,
    input_shape: tuple[int, int, int, int],
    device: torch.device,
) -> CheckResult:
    model = model.to(device=device).eval()
    inputs = torch.randn(*input_shape, device=device)
    with torch.inference_mode():
        outputs = model(inputs)
    result = CheckResult(
        name=name,
        shape=tuple(outputs.shape),
        device=outputs.device.type,
    )
    del outputs, inputs, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_checks(
    device_name: str = "auto",
    include_imagenet: bool = False,
) -> list[CheckResult]:
    """Run CIFAR checks and optionally the five ImageNet structure checks."""

    device = resolve_device(device_name)
    results = [
        _forward_check(
            name,
            create_model(name, num_classes=10),
            (1, 3, 32, 32),
            device,
        )
        for name in CIFAR_MODEL_NAMES
    ]
    if include_imagenet:
        for name, factory in IMAGENET_FACTORIES.items():
            results.append(
                _forward_check(
                    name,
                    factory(num_classes=1000),
                    (1, 3, 224, 224),
                    device,
                )
            )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="验证克隆后的 CIFAR 与 ImageNet ResNet 结构",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="运行设备，默认自动选择 CUDA 或 CPU",
    )
    parser.add_argument(
        "--include-imagenet",
        action="store_true",
        help="同时验证 ImageNet ResNet-18/34/50/101/152",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = run_checks(
        device_name=args.device,
        include_imagenet=args.include_imagenet,
    )
    for result in results:
        print(
            f"[PASS] {result.name}: "
            f"shape={list(result.shape)}, device={result.device}"
        )
    print(f"全部 {len(results)} 项结构验证通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
