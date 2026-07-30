"""Tests for the clean-clone reproduction verification entry point."""

from importlib.util import find_spec

import pytest
import torch
from torch import nn


def test_verifier_module_exists() -> None:
    assert find_spec("verify_reproduction") is not None


def test_cpu_checks_cover_four_formal_cifar_models() -> None:
    from verify_reproduction import run_checks

    results = run_checks(device_name="cpu", include_imagenet=False)

    assert [result.name for result in results] == [
        "plain20",
        "resnet20",
        "plain56",
        "resnet56",
    ]
    assert all(result.shape == (1, 10) for result in results)
    assert all(result.device == "cpu" for result in results)


def test_imagenet_checks_are_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import verify_reproduction

    class TinyImageNetModel(nn.Module):
        def __init__(self, num_classes: int = 1000) -> None:
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(3, num_classes)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.fc(self.pool(inputs).flatten(1))

    factories = {
        name: TinyImageNetModel
        for name in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
    }
    monkeypatch.setattr(verify_reproduction, "IMAGENET_FACTORIES", factories)

    results = verify_reproduction.run_checks(
        device_name="cpu",
        include_imagenet=True,
    )

    assert [result.name for result in results[-5:]] == list(factories)
    assert all(result.shape == (1, 1000) for result in results[-5:])


def test_cuda_request_reports_unavailable_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import verify_reproduction

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        verify_reproduction.resolve_device("cuda")


def test_invalid_device_name_is_rejected() -> None:
    from verify_reproduction import resolve_device

    with pytest.raises(ValueError, match="auto、cpu 或 cuda"):
        resolve_device("tpu")


def test_main_prints_each_successful_check(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import verify_reproduction

    monkeypatch.setattr(
        verify_reproduction,
        "run_checks",
        lambda **_: [
            verify_reproduction.CheckResult("plain20", (1, 10), "cpu"),
        ],
    )
    monkeypatch.setattr(
        "sys.argv",
        ["verify_reproduction.py", "--device", "cpu"],
    )

    assert verify_reproduction.main() == 0
    assert "[PASS] plain20" in capsys.readouterr().out
