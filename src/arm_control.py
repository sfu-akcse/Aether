from ik_solver import solve_ik
from servo_conversion import angle_to_servo_position


def target_to_servo_positions(
    z,
    y,
    L1,
    L2,
    elbow_up=True,
    shoulder_center=2047,
    shoulder_direction=1,
    elbow_center=2047,
    elbow_direction=1,
):
    """
    Convert a target (z, y) position into theoretical joint
    angles and corresponding ST3215 servo positions.

    NOTE:
    Servo centers and directions are provisional until
    the physical robot arm is assembled and calibrated.
    """

    ik_result = solve_ik(
        z=z,
        y=y,
        L1=L1,
        L2=L2,
        elbow_up=elbow_up,
    )

    if not ik_result["reachable"]:
        return {
            "reachable": False,
            "shoulder_angle_deg": None,
            "elbow_angle_deg": None,
            "shoulder_servo_position": None,
            "elbow_servo_position": None,
        }

    shoulder_angle = ik_result["theta1_deg"]
    elbow_angle = ik_result["theta2_deg"]

    shoulder_position = angle_to_servo_position(
        shoulder_angle,
        center=shoulder_center,
        direction=shoulder_direction,
    )

    elbow_position = angle_to_servo_position(
        elbow_angle,
        center=elbow_center,
        direction=elbow_direction,
    )

    return {
        "reachable": True,
        "shoulder_angle_deg": shoulder_angle,
        "elbow_angle_deg": elbow_angle,
        "shoulder_servo_position": shoulder_position,
        "elbow_servo_position": elbow_position,
    }


if __name__ == "__main__":
    result = target_to_servo_positions(
        z=9,
        y=10,
        L1=10,
        L2=10,
        elbow_up=True,
    )

    print(result)