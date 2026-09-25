import pytest

from arm_control import (
    target_to_servo_positions,
    NEGATIVE_ELBOW,
    POSITIVE_ELBOW,
)


def test_both_branches_are_valid():
    result = target_to_servo_positions(
        z=10,
        y=10,
        L1=10,
        L2=10,
    )

    assert result["reachable"] is True
    assert result["geometrically_reachable"] is True

    assert (
        result["solutions"][NEGATIVE_ELBOW]["valid"]
        is True
    )

    assert (
        result["solutions"][POSITIVE_ELBOW]["valid"]
        is True
    )


def test_default_prefers_positive_elbow():
    result = target_to_servo_positions(
        z=10,
        y=10,
        L1=10,
        L2=10,
    )

    assert (
        result["selected_branch"]
        == POSITIVE_ELBOW
    )

    assert (
        result["shoulder_servo_position"]
        == 2030
    )

    assert (
        result["elbow_servo_position"]
        == 1061
    )


def test_can_prefer_negative_elbow():
    result = target_to_servo_positions(
        z=10,
        y=10,
        L1=10,
        L2=10,
        preferred_branch=NEGATIVE_ELBOW,
    )

    assert (
        result["selected_branch"]
        == NEGATIVE_ELBOW
    )

    assert (
        result["shoulder_servo_position"]
        == 3054
    )

    assert (
        result["elbow_servo_position"]
        == 3109
    )


def test_falls_back_to_only_valid_branch():
    result = target_to_servo_positions(
        z=15,
        y=0,
        L1=10,
        L2=10,
        preferred_branch=POSITIVE_ELBOW,
    )

    assert result["reachable"] is True

    # Positive-elbow solution requires a negative
    # shoulder angle, which our real arm cannot use.
    assert (
        result["solutions"][POSITIVE_ELBOW]["valid"]
        is False
    )

    assert (
        result["solutions"][NEGATIVE_ELBOW]["valid"]
        is True
    )

    assert (
        result["selected_branch"]
        == NEGATIVE_ELBOW
    )


def test_geometrically_reachable_but_physically_invalid():
    result = target_to_servo_positions(
        z=10,
        y=5,
        L1=10,
        L2=10,
    )

    assert result["geometrically_reachable"] is True
    assert result["reachable"] is False

    assert result["selected_branch"] is None


def test_geometrically_unreachable_target():
    result = target_to_servo_positions(
        z=25,
        y=0,
        L1=10,
        L2=10,
    )

    assert result["geometrically_reachable"] is False
    assert result["reachable"] is False

    assert result["selected_branch"] is None


def test_invalid_preferred_branch():
    with pytest.raises(ValueError):
        target_to_servo_positions(
            z=10,
            y=10,
            L1=10,
            L2=10,
            preferred_branch="banana",
        )