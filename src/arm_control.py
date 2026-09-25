from ik_solver import solve_ik

from servo_conversion import (
    shoulder_angle_to_position,
    elbow_angle_to_position,
)


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
    Calculate one IK branch and determine whether
    it satisfies the calibrated physical joint limits.
    """

    if branch == NEGATIVE_ELBOW:
        elbow_up = True

    elif branch == POSITIVE_ELBOW:
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

    # Check shoulder limits
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

    # Check elbow limits
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


def _movement_cost(
    solution,
    current_shoulder_position,
    current_elbow_position,
):
    """
    Estimate how much servo movement is required
    to reach an IK solution.

    Lower cost = less total servo movement.
    """

    shoulder_difference = abs(
        solution["shoulder_servo_position"]
        - current_shoulder_position
    )

    elbow_difference = abs(
        solution["elbow_servo_position"]
        - current_elbow_position
    )

    return (
        shoulder_difference
        + elbow_difference
    )


def target_to_servo_positions(
    z,
    y,
    L1,
    L2,
    preferred_branch=POSITIVE_ELBOW,
    current_shoulder_position=None,
    current_elbow_position=None,
):
    """
    Convert a target position into a physically valid
    robot-arm configuration.

    Both IK branches are calculated and checked
    against the calibrated joint limits.

    If current servo positions are provided and both
    branches are valid, the branch requiring the
    least servo movement is selected.

    If current positions are not provided, the
    preferred branch is used when possible.
    """

    if preferred_branch not in (
        NEGATIVE_ELBOW,
        POSITIVE_ELBOW,
    ):
        raise ValueError(
            "preferred_branch must be "
            "'negative_elbow' or 'positive_elbow'"
        )

    # Either provide BOTH current positions or neither.
    if (
        (current_shoulder_position is None)
        !=
        (current_elbow_position is None)
    ):
        raise ValueError(
            "current_shoulder_position and "
            "current_elbow_position must either "
            "both be provided or both be None"
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

    # Target outside basic geometric workspace
    if not negative_solution["geometrically_reachable"]:
        return {
            "reachable": False,
            "geometrically_reachable": False,
            "selected_branch": None,
            "selection_reason": None,
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

    valid_branches = [
        branch
        for branch, solution in solutions.items()
        if solution["valid"]
    ]

    # Geometrically reachable, but hardware cannot
    # safely perform either configuration.
    if not valid_branches:
        return {
            "reachable": False,
            "geometrically_reachable": True,
            "selected_branch": None,
            "selection_reason": None,
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
    # Only one physical solution exists
    # --------------------------------------------------

    if len(valid_branches) == 1:
        selected_branch = valid_branches[0]

        selection_reason = (
            "Only one IK branch satisfies "
            "the calibrated joint limits."
        )

    # --------------------------------------------------
    # Both physical solutions exist
    # --------------------------------------------------

    else:
        # If we know the current pose, choose the
        # solution requiring the least movement.
        if current_shoulder_position is not None:

            negative_cost = _movement_cost(
                negative_solution,
                current_shoulder_position,
                current_elbow_position,
            )

            positive_cost = _movement_cost(
                positive_solution,
                current_shoulder_position,
                current_elbow_position,
            )

            if negative_cost < positive_cost:
                selected_branch = NEGATIVE_ELBOW

            elif positive_cost < negative_cost:
                selected_branch = POSITIVE_ELBOW

            else:
                # Exact tie: use configured preference.
                selected_branch = preferred_branch

            selection_reason = (
                "Both IK branches are valid; "
                "selected the solution requiring "
                "the least servo movement."
            )

        else:
            # No current pose available yet.
            selected_branch = preferred_branch

            selection_reason = (
                "Both IK branches are valid; "
                "no current pose was provided, "
                "so preferred_branch was used."
            )

    selected = solutions[selected_branch]

    return {
        "reachable": True,
        "geometrically_reachable": True,
        "selected_branch": selected_branch,
        "selection_reason": selection_reason,
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
        z=120,
        y=100,
        L1=135,
        L2=85,
        current_shoulder_position=2598,
        current_elbow_position=3002,
    )

    print(result)