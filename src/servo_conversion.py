# ST3215 servo position conversion
#
# Converts theoretical joint angles in degrees
# into ST3215 servo position counts.
#
# NOTE:
# center and direction are provisional until
# the physical robot arm can be assembled/calibrated.

SERVO_COUNTS_PER_REV = 4096
DEGREES_PER_REV = 360.0

COUNTS_PER_DEGREE = (
    SERVO_COUNTS_PER_REV / DEGREES_PER_REV
)

# servo theoretical range is 0-4095, but the physical servo
# may not be able to reach the full range.  The physical
# limits will be determined after the robot arm is assembled and calibrated.
SERVO_MIN_POSITION = 0
SERVO_MAX_POSITION = 4095

# values may need to change after the physical robot arm is assembled and calibrated
def angle_to_servo_position(
    angle_deg,
    center=2047,
    direction=1,
    min_position=SERVO_MIN_POSITION,
    max_position=SERVO_MAX_POSITION,
):
    """
    Convert a joint angle in degrees to an ST3215 position.

    angle_deg:
        The theoretical joint angle.

    center:
        Servo position corresponding to 0 degrees.
        Currently assumed to be 2047.

    direction:
        +1 = positive angle increases servo count
        -1 = positive angle decreases servo count

    min_position / max_position:
        Allowed servo position range.

    Returns:
        Integer servo position.
    """

    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")

    if min_position > max_position:
        raise ValueError(
            "min_position cannot be greater than max_position"
        )

    if center < min_position or center > max_position:
        raise ValueError(
            "center must be inside the allowed servo range"
        )

    position = round(
        center
        + direction
        * angle_deg
        * COUNTS_PER_DEGREE
    )

    # check final position is within allowed range
    if position < min_position or position > max_position:
        raise ValueError(
            f"Calculated servo position {position} "
            f"is outside allowed range "
            f"{min_position}-{max_position}"
        )

    return position

# convert servo position back to theoretical joint angle in degrees
def servo_position_to_angle(
    position,
    center=2047,
    direction=1,
):
    """
    Convert an ST3215 servo position back to
    a theoretical joint angle in degrees.
    """

    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")

    if position < SERVO_MIN_POSITION or position > SERVO_MAX_POSITION:
        raise ValueError(
            "position must be between 0 and 4095"
        )

    angle_deg = (
        (position - center)
        / COUNTS_PER_DEGREE
        * direction
    )

    return angle_deg