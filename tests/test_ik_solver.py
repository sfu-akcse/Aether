import math
import pytest

from ik_solver import solve_ik, forward_kinematics


def forward_kinematics(theta1, theta2, L1, L2):
    """
    Calculate end-effector position from joint angles.

    Coordinate plane:
        z = forward/back
        y = up/down
    """

    z = (
        L1 * math.cos(theta1)
        + L2 * math.cos(theta1 + theta2)
    )

    y = (
        L1 * math.sin(theta1)
        + L2 * math.sin(theta1 + theta2)
    )

    return z, y


@pytest.mark.parametrize("elbow_up", [True, False])
def test_reachable_target(elbow_up):
    z_target = 9
    y_target = 10

    L1 = 10
    L2 = 10

    result = solve_ik(
        z_target,
        y_target,
        L1,
        L2,
        elbow_up=elbow_up,
    )

    assert result["reachable"] is True

    z_result, y_result = forward_kinematics(
        result["theta1_rad"],
        result["theta2_rad"],
        L1,
        L2,
    )

    assert math.isclose(
        z_result,
        z_target,
        abs_tol=1e-9,
    )

    assert math.isclose(
        y_result,
        y_target,
        abs_tol=1e-9,
    )


def test_fully_extended_arm():
    result = solve_ik(
        z=20,
        y=0,
        L1=10,
        L2=10,
    )

    assert result["reachable"] is True

    assert math.isclose(
        result["theta1_deg"],
        0,
        abs_tol=1e-9,
    )

    assert math.isclose(
        result["theta2_deg"],
        0,
        abs_tol=1e-9,
    )


def test_target_too_far():
    result = solve_ik(
        z=25,
        y=0,
        L1=10,
        L2=10,
    )

    assert result["reachable"] is False

    assert result["theta1_rad"] is None
    assert result["theta2_rad"] is None


def test_target_too_close():
    result = solve_ik(
        z=2,
        y=0,
        L1=10,
        L2=4,
    )

    assert result["reachable"] is False


def test_invalid_L1():
    with pytest.raises(ValueError):
        solve_ik(
            z=5,
            y=5,
            L1=0,
            L2=10,
        )


def test_invalid_L2():
    with pytest.raises(ValueError):
        solve_ik(
            z=5,
            y=5,
            L1=10,
            L2=-1,
        )


@pytest.mark.parametrize(
    "z_target,y_target",
    [
        (10, 5),
        (5, 10),
        (15, 0),
        (0, 15),
        (8, -4),
        (-5, 10),
    ],
)
@pytest.mark.parametrize(
    "elbow_up",
    [True, False],
)
def test_forward_kinematics_reconstructs_target(
    z_target,
    y_target,
    elbow_up,
):
    L1 = 10
    L2 = 10

    result = solve_ik(
        z_target,
        y_target,
        L1,
        L2,
        elbow_up=elbow_up,
    )

    assert result["reachable"] is True

    z_result, y_result = forward_kinematics(
        result["theta1_rad"],
        result["theta2_rad"],
        L1,
        L2,
    )

    assert math.isclose(
        z_result,
        z_target,
        abs_tol=1e-9,
    )

    assert math.isclose(
        y_result,
        y_target,
        abs_tol=1e-9,
    )