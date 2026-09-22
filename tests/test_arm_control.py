import math

from arm_control import target_to_servo_positions


def test_reachable_target():
    result = target_to_servo_positions(
        z=9,
        y=10,
        L1=10,
        L2=10,
        elbow_up=True,
    )

    assert result["reachable"] is True

    assert result["shoulder_servo_position"] is not None
    assert result["elbow_servo_position"] is not None


def test_unreachable_target():
    result = target_to_servo_positions(
        z=25,
        y=0,
        L1=10,
        L2=10,
    )

    assert result["reachable"] is False

    assert result["shoulder_angle_deg"] is None
    assert result["elbow_angle_deg"] is None

    assert result["shoulder_servo_position"] is None
    assert result["elbow_servo_position"] is None


def test_straight_arm():
    result = target_to_servo_positions(
        z=20,
        y=0,
        L1=10,
        L2=10,
    )

    assert result["reachable"] is True

    assert math.isclose(
        result["shoulder_angle_deg"],
        0,
        abs_tol=1e-9,
    )

    assert math.isclose(
        result["elbow_angle_deg"],
        0,
        abs_tol=1e-9,
    )

    assert result["shoulder_servo_position"] == 2047
    assert result["elbow_servo_position"] == 2047


def test_reversed_elbow_direction():
    result = target_to_servo_positions(
        z=20,
        y=0,
        L1=10,
        L2=10,
        elbow_direction=-1,
    )

    assert result["reachable"] is True

    assert result["elbow_servo_position"] == 2047