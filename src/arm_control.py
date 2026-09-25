from ik_solver import solve_ik
from servo_conversion import (shoulder_angle_to_position, elbow_angle_to_position,)


NEGATIVE_ELBOW = "negative_elbow"
POSITIVE_ELBOW = "positive_elbow"


def _calculate_candidate(
    z,
    y,
    L1,
    L2,
    branch,
):
    """
    Calculate one IK branch and check whether it fits
    within the calibrated physical servo limits.
    """

    if branch == NEGATIVE_ELBOW:
        # solve_ik() uses negative theta2 when elbow_up=True
        elbow_up = True

    elif branch == POSITIVE_ELBOW:
        # solve_ik() uses positive theta2 when elbow_up=False
        elbow_up = False

    else:
        raise ValueError(
            "branch must be "
            "'negative_elbow' or 'positive_elbow'"
        )

    ik_result = solve_ik(
        z=z,
        y=y,
        L1=L1,
        L2=L2,
        elbow_up=elbow_up,
    )

    # Geometrically unreachable targets cannot have a valid hardware solution
    if not ik_result["reachable"]:
        return {
            "branch": branch,
            "geometrically_reachable": False,
            "valid": False,
            "shoulder_angle_deg": None,
            "elbow_angle_deg": None,
            "shoulder_servo_position": None,
            "elbow_servo_position": None,
            "errors": [
                "Target is geometrically unreachable."
            ],
        }

    shoulder_angle = ik_result["theta1_deg"]
    elbow_angle = ik_result["theta2_deg"]

    shoulder_position = None
    elbow_position = None

    errors = []

    # --------------------------------------------------
    # Check shoulder against calibrated hardware limits
    # --------------------------------------------------

    try:
        shoulder_position = (
            shoulder_angle_to_position(
                shoulder_angle
            )
        )

    except ValueError as error:
        errors.append(
            f"Shoulder limit: {error}"
        )

    # --------------------------------------------------
    # Check elbow against calibrated hardware limits
    # --------------------------------------------------

    try:
        elbow_position = (
            elbow_angle_to_position(
                elbow_angle
            )
        )

    except ValueError as error:
        errors.append(
            f"Elbow limit: {error}"
        )

    return {
        "branch": branch,
        "geometrically_reachable": True,
        "valid": len(errors) == 0,
        "shoulder_angle_deg": shoulder_angle,
        "elbow_angle_deg": elbow_angle,
        "shoulder_servo_position": shoulder_position,
        "elbow_servo_position": elbow_position,
        "errors": errors,
    }


def target_to_servo_positions(
    z,
    y,
    L1,
    L2,
    preferred_branch=POSITIVE_ELBOW,
):
    """
    Convert a target position into a physically valid
    robot-arm configuration.

    Both 2-link IK branches are calculated:

        negative_elbow -> theta2 < 0
        positive_elbow -> theta2 > 0

    Each branch is checked against the calibrated
    shoulder and elbow servo limits.

    If both are valid, preferred_branch is selected.

    If only one is valid, that branch is selected
    automatically.

    If neither is valid, reachable=False.
    """

    if preferred_branch not in (
        NEGATIVE_ELBOW,
        POSITIVE_ELBOW,
    ):
        raise ValueError(
            "preferred_branch must be "
            "'negative_elbow' or 'positive_elbow'"
        )

    negative_solution = _calculate_candidate(
        z=z,
        y=y,
        L1=L1,
        L2=L2,
        branch=NEGATIVE_ELBOW,
    )

    positive_solution = _calculate_candidate(
        z=z,
        y=y,
        L1=L1,
        L2=L2,
        branch=POSITIVE_ELBOW,
    )

    solutions = {
        NEGATIVE_ELBOW: negative_solution,
        POSITIVE_ELBOW: positive_solution,
    }

    # --------------------------------------------------
    # Target is outside the basic 2-link workspace
    # --------------------------------------------------

    if not negative_solution["geometrically_reachable"]:
        return {
            "reachable": False,
            "geometrically_reachable": False,
            "selected_branch": None,
            "shoulder_angle_deg": None,
            "elbow_angle_deg": None,
            "shoulder_servo_position": None,
            "elbow_servo_position": None,
            "reason": (
                "Target is outside the geometric "
                "2-link workspace."
            ),
            "solutions": solutions,
        }

    # --------------------------------------------------
    # Determine which hardware-valid branches exist
    # --------------------------------------------------

    valid_branches = [
        branch
        for branch, solution in solutions.items()
        if solution["valid"]
    ]

    if not valid_branches:
        return {
            "reachable": False,
            "geometrically_reachable": True,
            "selected_branch": None,
            "shoulder_angle_deg": None,
            "elbow_angle_deg": None,
            "shoulder_servo_position": None,
            "elbow_servo_position": None,
            "reason": (
                "Target is mathematically reachable, "
                "but neither IK branch satisfies the "
                "calibrated joint limits."
            ),
            "solutions": solutions,
        }

    # --------------------------------------------------
    # Prefer requested branch if it is physically valid
    #
    # Otherwise automatically fall back to the other valid solution
    # --------------------------------------------------

    if preferred_branch in valid_branches:
        selected_branch = preferred_branch

    else:
        selected_branch = valid_branches[0]

    selected = solutions[selected_branch]

    return {
        "reachable": True,
        "geometrically_reachable": True,
        "selected_branch": selected_branch,
        "shoulder_angle_deg": (
            selected["shoulder_angle_deg"]
        ),
        "elbow_angle_deg": (
            selected["elbow_angle_deg"]
        ),
        "shoulder_servo_position": (
            selected["shoulder_servo_position"]
        ),
        "elbow_servo_position": (
            selected["elbow_servo_position"]
        ),
        "reason": None,
        "solutions": solutions,
    }


if __name__ == "__main__":
    result = target_to_servo_positions(
        z=10,
        y=10,
        L1=10,
        L2=10,
    )

    print(result)