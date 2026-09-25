from arm_control import (
    target_to_servo_positions,
    NEGATIVE_ELBOW,
    POSITIVE_ELBOW,
)


# --------------------------------------------------
# Current physical prototype dimensions
# --------------------------------------------------

L1 = 135.0  # shoulder axis -> elbow axis, mm
L2 = 85.0   # elbow axis -> current endpoint, mm


# --------------------------------------------------
# Targets to test
#
# z = forward/back
# y = up/down
# --------------------------------------------------

TARGETS = [
    (220, 0),     # almost/full extension forward
    (200, 0),
    (180, 40),
    (160, 60),
    (120, 100),
    (100, 120),
    (80, 140),
    (0, 180),     # straight above shoulder area
    (-50, 150),   # behind shoulder: negative z
    (100, 0),     # geometrically reachable, may violate joint limits
    (250, 0),     # geometrically unreachable
]


def print_candidate(solution):
    """
    Print one IK branch in a readable format.
    """

    branch = solution["branch"]

    print(f"    {branch}:")

    if not solution["geometrically_reachable"]:
        print("        geometrically reachable: NO")
        return

    print("        geometrically reachable: YES")
    print(f"        physically valid: {solution['valid']}")

    print(
        f"        shoulder angle: "
        f"{solution['shoulder_angle_deg']:.2f} deg"
    )

    print(
        f"        elbow angle: "
        f"{solution['elbow_angle_deg']:.2f} deg"
    )

    print(
        f"        shoulder servo: "
        f"{solution['shoulder_servo_position']}"
    )

    print(
        f"        elbow servo: "
        f"{solution['elbow_servo_position']}"
    )

    if solution["errors"]:
        print("        errors:")

        for error in solution["errors"]:
            print(f"          - {error}")


def main():
    print("==========================================")
    print("Aether 2-Link IK Workspace Check")
    print("==========================================")

    print(f"L1 = {L1} mm")
    print(f"L2 = {L2} mm")

    print()
    print(
        f"Maximum geometric reach = "
        f"{L1 + L2:.1f} mm"
    )

    print(
        f"Minimum geometric reach = "
        f"{abs(L1 - L2):.1f} mm"
    )

    print()

    for z, y in TARGETS:
        print("==========================================")
        print(f"Target: z={z} mm, y={y} mm")
        print("==========================================")

        result = target_to_servo_positions(
            z=z,
            y=y,
            L1=L1,
            L2=L2,
            preferred_branch=POSITIVE_ELBOW,
        )

        print_candidate(
            result["solutions"][NEGATIVE_ELBOW]
        )

        print()

        print_candidate(
            result["solutions"][POSITIVE_ELBOW]
        )

        print()

        if result["reachable"]:
            print("RESULT: VALID")

            print(
                f"Selected branch: "
                f"{result['selected_branch']}"
            )

            print(
                f"Shoulder command: "
                f"{result['shoulder_servo_position']}"
            )

            print(
                f"Elbow command: "
                f"{result['elbow_servo_position']}"
            )

        else:
            print("RESULT: NOT VALID FOR ROBOT")

            print(
                f"Reason: {result['reason']}"
            )

        print()

    print("==========================================")
    print("Workspace check complete.")
    print("NO commands were sent to the physical arm.")
    print("==========================================")


if __name__ == "__main__":
    main()