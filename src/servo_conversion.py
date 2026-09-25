# ST3215 servo position conversion
#
# Converts theoretical joint angles in degrees
# into ST3215 servo position counts.

SERVO_COUNTS_PER_REV = 4096
DEGREES_PER_REV = 360.0

COUNTS_PER_DEGREE = (
    SERVO_COUNTS_PER_REV / DEGREES_PER_REV
)

# ST3215 theoretical position range
SERVO_MIN_POSITION = 0
SERVO_MAX_POSITION = 4095


# --------------------------------------------------
# Physical Aether prototype calibration
# --------------------------------------------------

SHOULDER_CENTER = 2030
SHOULDER_DIRECTION = +1
SHOULDER_MIN = 2030
SHOULDER_MAX = 4000

ELBOW_CENTER = 2085
ELBOW_DIRECTION = -1
ELBOW_MIN = 950
ELBOW_MAX = 3150


def angle_to_servo_position(
    angle_deg,
    center=2047,
    direction=1,
    min_position=SERVO_MIN_POSITION,
    max_position=SERVO_MAX_POSITION,
):
    """
    Convert a theoretical joint angle in degrees
    to an ST3215 servo position.

    angle_deg:
        Joint angle in degrees.

    center:
        Servo position corresponding to 0 degrees.

    direction:
        +1 = positive angle increases servo count
        -1 = positive angle decreases servo count

    min_position / max_position:
        Allowed servo command range.

    Returns:
        Integer servo position.
    """

    if direction not in (-1, 1):
        raise ValueError(
            "direction must be +1 or -1"
        )

    if min_position > max_position:
        raise ValueError(
            "min_position cannot be greater than max_position"
        )

    position = round(
        center
        + direction
        * angle_deg
        * COUNTS_PER_DEGREE
    )

    if position < min_position or position > max_position:
        raise ValueError(
            f"Calculated servo position {position} "
            f"is outside allowed range "
            f"{min_position}-{max_position}"
        )

    return position


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
        raise ValueError(
            "direction must be +1 or -1"
        )

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


def shoulder_angle_to_position(angle_deg):
    """
    Convert a theoretical shoulder angle to the
    calibrated Aether shoulder servo position.
    """

    return angle_to_servo_position(
        angle_deg,
        center=SHOULDER_CENTER,
        direction=SHOULDER_DIRECTION,
        min_position=SHOULDER_MIN,
        max_position=SHOULDER_MAX,
    )


def elbow_angle_to_position(angle_deg):
    """
    Convert a theoretical elbow angle to the
    calibrated Aether elbow servo position.
    """

    return angle_to_servo_position(
        angle_deg,
        center=ELBOW_CENTER,
        direction=ELBOW_DIRECTION,
        min_position=ELBOW_MIN,
        max_position=ELBOW_MAX,
    )