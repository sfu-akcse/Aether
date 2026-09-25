import math
import pytest

from servo_conversion import (
    angle_to_servo_position,
    servo_position_to_angle,
    shoulder_angle_to_position,
    elbow_angle_to_position,
)


def test_zero_degrees_is_center():
    assert angle_to_servo_position(0) == 2047


def test_positive_45_degrees():
    assert angle_to_servo_position(45) == 2559


def test_negative_45_degrees():
    assert angle_to_servo_position(-45) == 1535


def test_positive_90_degrees():
    assert angle_to_servo_position(90) == 3071


def test_negative_90_degrees():
    assert angle_to_servo_position(-90) == 1023


def test_reverse_direction_positive_angle():
    assert angle_to_servo_position(
        45,
        direction=-1,
    ) == 1535


def test_reverse_direction_negative_angle():
    assert angle_to_servo_position(
        -45,
        direction=-1,
    ) == 2559


def test_custom_center():
    assert angle_to_servo_position(
        45,
        center=2100,
    ) == 2612


def test_invalid_direction():
    with pytest.raises(ValueError):
        angle_to_servo_position(
            45,
            direction=0,
        )


def test_angle_outside_servo_range():
    with pytest.raises(ValueError):
        angle_to_servo_position(
            200,
        )


def test_invalid_position_range():
    with pytest.raises(ValueError):
        angle_to_servo_position(
            0,
            min_position=3000,
            max_position=2000,
        )


@pytest.mark.parametrize(
    "angle",
    [
        -90,
        -45,
        -10,
        0,
        10,
        45,
        90,
    ],
)

def test_angle_position_round_trip(angle):
    position = angle_to_servo_position(angle)

    calculated_angle = servo_position_to_angle(
        position
    )

    # One encoder count is about 0.088 degrees,
    # so allow a small rounding error.
    assert math.isclose(
        calculated_angle,
        angle,
        abs_tol=0.05,
    )

def test_calibrated_shoulder_zero():
    assert shoulder_angle_to_position(0) == 2030


def test_calibrated_elbow_zero():
    assert elbow_angle_to_position(0) == 2085


def test_calibrated_shoulder_90_degrees():
    assert shoulder_angle_to_position(90) == 3054


def test_calibrated_elbow_positive_90():
    assert elbow_angle_to_position(90) == 1061


def test_calibrated_elbow_negative_90():
    assert elbow_angle_to_position(-90) == 3109


def test_shoulder_rejects_negative_angle():
    with pytest.raises(ValueError):
        shoulder_angle_to_position(-10)


def test_elbow_rejects_angle_outside_safe_range():
    with pytest.raises(ValueError):
        elbow_angle_to_position(120)   