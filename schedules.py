"""训练过程中与模型无关的学习率和 iteration 日程。"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PaperSchedule:
    """CIFAR-10论文使用的分段常数iteration日程。"""

    max_iterations: int = 64000
    milestones: tuple[int, int] = (32000, 48000)
    learning_rates: tuple[float, float, float] = (0.1, 0.01, 0.001)

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
        ):
            raise TypeError("max_iterations必须是整数")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations必须大于0")

        if not isinstance(self.milestones, tuple) or len(self.milestones) != 2:
            raise TypeError("milestones必须是包含两个整数的tuple")
        first, second = self.milestones
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.milestones
        ):
            raise TypeError("milestones必须只包含整数")
        if not 0 < first < second < self.max_iterations:
            raise ValueError(
                "milestones必须在0和max_iterations之间严格递增"
            )

        if (
            not isinstance(self.learning_rates, tuple)
            or len(self.learning_rates) != 3
        ):
            raise TypeError("learning_rates必须是包含三个数值的tuple")
        for learning_rate in self.learning_rates:
            if (
                isinstance(learning_rate, bool)
                or not isinstance(learning_rate, (int, float))
            ):
                raise TypeError("learning_rates必须只包含数值")
            if not math.isfinite(learning_rate) or learning_rate <= 0:
                raise ValueError("learning_rates必须是有限正数")

    def _validate_completed(self, completed_iterations: int) -> None:
        if (
            isinstance(completed_iterations, bool)
            or not isinstance(completed_iterations, int)
        ):
            raise TypeError("completed_iterations必须是整数")
        if not 0 <= completed_iterations <= self.max_iterations:
            raise ValueError(
                "completed_iterations必须位于0和max_iterations之间"
            )

    @staticmethod
    def _validate_loader_batches(loader_batches: int) -> None:
        if isinstance(loader_batches, bool) or not isinstance(
            loader_batches,
            int,
        ):
            raise TypeError("loader_batches必须是整数")
        if loader_batches <= 0:
            raise ValueError("loader_batches必须大于0")

    def learning_rate_after(self, completed_iterations: int) -> float:
        """返回已经完成指定更新数后，下一次更新应使用的学习率。"""

        self._validate_completed(completed_iterations)
        first, second = self.milestones
        if completed_iterations < first:
            return float(self.learning_rates[0])
        if completed_iterations < second:
            return float(self.learning_rates[1])
        return float(self.learning_rates[2])

    def remaining_iterations(self, completed_iterations: int) -> int:
        """返回距离严格停止点还剩多少次参数更新。"""

        self._validate_completed(completed_iterations)
        return self.max_iterations - completed_iterations

    def max_batches_for_epoch(
        self,
        completed_iterations: int,
        loader_batches: int,
    ) -> int:
        """限制当前数据轮次，避免训练超过最终iteration。"""

        self._validate_loader_batches(loader_batches)
        return min(
            loader_batches,
            self.remaining_iterations(completed_iterations),
        )

    def is_complete(self, completed_iterations: int) -> bool:
        """判断是否已经严格完成整个日程。"""

        self._validate_completed(completed_iterations)
        return completed_iterations == self.max_iterations
