"""论文 iteration 学习率日程测试。"""

from dataclasses import FrozenInstanceError

import pytest

from schedules import PaperSchedule


def test_default_schedule_drives_paper_boundary_behavior() -> None:
    schedule = PaperSchedule()

    assert schedule.max_iterations == 64000
    assert schedule.milestones == (32000, 48000)
    assert schedule.learning_rates == (0.1, 0.01, 0.001)
    assert schedule.remaining_iterations(0) == 64000


def test_paper_schedule_is_immutable() -> None:
    schedule = PaperSchedule()

    with pytest.raises(FrozenInstanceError):
        schedule.max_iterations = 1


@pytest.mark.parametrize(
    ("completed", "expected"),
    [
        (0, 0.1),
        (31999, 0.1),
        (32000, 0.01),
        (47999, 0.01),
        (48000, 0.001),
        (63999, 0.001),
        (64000, 0.001),
    ],
)
def test_learning_rate_changes_after_exact_completed_update(
    completed: int,
    expected: float,
) -> None:
    schedule = PaperSchedule()

    assert schedule.learning_rate_after(completed) == expected


def test_short_schedule_limits_full_and_partial_epochs() -> None:
    schedule = PaperSchedule(
        max_iterations=7,
        milestones=(3, 5),
        learning_rates=(0.1, 0.01, 0.001),
    )

    assert schedule.remaining_iterations(4) == 3
    assert schedule.max_batches_for_epoch(0, 4) == 4
    assert schedule.max_batches_for_epoch(4, 4) == 3
    assert schedule.is_complete(6) is False
    assert schedule.is_complete(7) is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iterations": 0},
        {"max_iterations": -1},
        {"max_iterations": True},
        {"max_iterations": 7.5},
        {"max_iterations": 7, "milestones": (0, 5)},
        {"max_iterations": 7, "milestones": (3, 3)},
        {"max_iterations": 7, "milestones": (5, 3)},
        {"max_iterations": 7, "milestones": (3, 7)},
        {"max_iterations": 7, "milestones": (3.0, 5)},
        {"learning_rates": (0.1, 0.01)},
        {"learning_rates": (0.1, 0.01, 0.0)},
        {"learning_rates": (0.1, float("inf"), 0.001)},
        {"learning_rates": (0.1, float("nan"), 0.001)},
        {"learning_rates": (0.1, True, 0.001)},
    ],
)
def test_schedule_rejects_invalid_configuration(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        PaperSchedule(**kwargs)


@pytest.mark.parametrize("completed", [-1, 64001, True, 1.5, "3"])
def test_completed_iterations_must_be_an_in_range_integer(
    completed: object,
) -> None:
    schedule = PaperSchedule()

    with pytest.raises((TypeError, ValueError)):
        schedule.learning_rate_after(completed)


@pytest.mark.parametrize("loader_batches", [0, -1, True, 3.5, "4"])
def test_loader_batches_must_be_a_positive_integer(
    loader_batches: object,
) -> None:
    schedule = PaperSchedule()

    with pytest.raises((TypeError, ValueError)):
        schedule.max_batches_for_epoch(0, loader_batches)
