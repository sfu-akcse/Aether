# input: x,y,L1,L2
# output: reachable, theta1, theta2

import math

def solve_ik(z, y, L1, L2, elbow_up=True):
    """
    Solves 2-D inverse kinematics

    z = forward/back target coordinate
    y = up/down target coordinate
    L1 = shoulder-to-elbow link length
    L2 = elbow-to-wrist/end-effector link length
    """

    # TODO 1: calculate r
    r = None

    # TODO 2: check reachability
    min_reach = None
    max_reach = None

    if False:  # replace this condition
        return {
            "reachable": False,
            "theta1_rad": None,
            "theta2_rad": None,
            "theta1_deg": None,
            "theta2_deg": None,
        }

    # TODO 3: calculate cos(theta2)
    cos_theta2 = None

    # TODO 4: clamp cos_theta2 between -1 and 1
    cos_theta2 = None

    # TODO 5: calculate theta2 magnitude using acos
    theta2_mag = None

    # TODO 6: choose elbow-up or elbow-down
    if elbow_up:
        theta2 = None
    else:
        theta2 = None

    # TODO 7: calculate phi
    phi = None

    # TODO 8: calculate offset angle a
    a = None

    # TODO 9: calculate theta1
    theta1 = None

    # TODO 10: convert both to degrees
    theta1_deg = None
    theta2_deg = None

    return {
        "reachable": True,
        "theta1_rad": theta1,
        "theta2_rad": theta2,
        "theta1_deg": theta1_deg,
        "theta2_deg": theta2_deg,
    }

# test values (below)
result = solve_ik(z=10,y=10,L1=10,L2=10,elbow_up=True)
print(result)