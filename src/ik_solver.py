# 2D Inverse Kinematics Solver
# Coord plane: z-y 

# input: 
#   target position: z,y 
#   link lengths: L1,L2
# output:
#   reachability
#   theoretical shoulder/elbow angles

import math

def solve_ik(z, y, L1, L2, elbow_up, debug):
    """
    Solves 2-D inverse kinematics

    z = forward/back target coordinate
    y = up/down target coordinate
    L1 = shoulder-to-elbow link length
    L2 = elbow-to-wrist/end-effector link length
    """

    # calculate r
    r = math.sqrt(z**2 + y**2)

    # check reachability
    min_reach = abs(L1 - L2)
    max_reach = L1 + L2

    if (r < min_reach) or (r > max_reach):  # if too far / too close
        return {
            "reachable": False,
            "theta1_rad": None,
            "theta2_rad": None,
            "theta1_deg": None,
            "theta2_deg": None,
        }

    # calculate cos(theta2)
    cos_theta2 = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)

    # clamp cos_theta2 between -1 and 1
    cos_theta2 = max(-1, min(1,cos_theta2))
    # we do this b/c of tiny floating-point errors
    # EX. if cos_theta = 1.0000001, would break math.acos()

    # calculate theta2 magnitude using acos
    theta2_mag = math.acos(cos_theta2)

    # choose elbow-up or elbow-down
    if elbow_up:
        theta2 = -theta2_mag
    else:
        theta2 = theta2_mag

    # calculate phi
    # use atan2 to calculate correct angle w.r.t correct quadrant
    # **** math.atan2(vertical, horizontal) ****
    phi = math.atan2(y,z)

    # calculate offset angle a
    # arg1: perpendicular component (the up/down part)
    # arg2: forward component (forward/backward)
    a = math.atan2((L2 * math.sin(theta2)), (L1 + L2 * math.cos(theta2)))

    # calculate theta1
    theta1 = phi - a

    # convert both to degrees
    theta1_deg = math.degrees(theta1)
    theta2_deg = math.degrees(theta2)
    
    if debug:
        print("r:", r)
        print("cos_theta2:", cos_theta2)
        print("theta2_deg:", math.degrees(theta2))
        print("phi_deg:", math.degrees(phi))
        print("a_deg:", math.degrees(a))
        print("theta1_deg:", theta1_deg)

    return {
        "reachable": True,
        "theta1_rad": theta1,
        "theta2_rad": theta2,
        "theta1_deg": theta1_deg,
        "theta2_deg": theta2_deg,
    }

## TESTING VALUES (manual input, for now)
z = 9
y = 10
L1 = L2 = 10
elbow_up = True
debug = True # turn debug mode on/off, allowing you to see print statements w/values

result = solve_ik(z,y,L1,L2,elbow_up,debug)
print(result)
