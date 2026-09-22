import math
import pytest

from servo_conversion import (
    angle_to_servo_position,
    servo_position_to_angle,
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