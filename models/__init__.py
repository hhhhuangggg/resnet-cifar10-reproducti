"""模型包的统一入口。

后续对外提供：

    create_model(name: str, num_classes: int = 10)

支持的模型名称：

    plain20
    plain32
    plain44
    plain56
    resnet20
    resnet32
    resnet44
    resnet56
"""
from torch import nn

from .plainnet import PlainNet
from .resnet import ResNet

MODEL_CONFIGS = {
    "plain20": {"family": "plain", "depth": 20, "n": 3},
    "plain32": {"family": "plain", "depth": 32, "n": 5},
    "plain44": {"family": "plain", "depth": 44, "n": 7},
    "plain56": {"family": "plain", "depth": 56, "n": 9},
    "resnet20": {"family": "resnet", "depth": 20, "n": 3},
    "resnet32": {"family": "resnet", "depth": 32, "n": 5},
    "resnet44": {"family": "resnet", "depth": 44, "n": 7},
    "resnet56": {"family": "resnet", "depth": 56, "n": 9},
}
SUPPORTED_MODELS = tuple(MODEL_CONFIGS.keys())
def create_model(name: str, num_classes: int = 10) -> nn.Module:
    """根据模型名称创建CIFAR分类模型。

    Args:
        name: 模型名称，例如 "plain20" 或 "resnet56"。
        num_classes: 分类类别数，CIFAR-10默认为10。

    Returns:
        创建完成的PyTorch模型。

    Raises:
        TypeError: name或num_classes类型错误。
        ValueError: 模型名称不存在或类别数不合法。
    """
    if not isinstance(name, str):
        raise TypeError(
            f"name必须是字符串，实际收到{type(name).__name__}"
        )
    
    normalized_name = name.strip().lower()

    if normalized_name not in MODEL_CONFIGS:
        supported = ", ".join(SUPPORTED_MODELS)
        raise ValueError(
            f"不支持的模型名称：{name!r}。支持的模型：{supported}"
        )
    if not isinstance(num_classes, int):
        raise TypeError(
            "num_classes必须是整数，"
            f"实际收到{type(num_classes).__name__}"
        )

    if num_classes <= 0:
        raise ValueError(
            f"num_classes必须大于0，实际收到{num_classes}"
        )
    config = MODEL_CONFIGS[normalized_name]

    if config["family"] == "plain":
        return PlainNet(
            n=config["n"],
            num_classes=num_classes,
        )

    return ResNet(
        n=config["n"],
        num_classes=num_classes,
    )
