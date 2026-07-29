import math
import numpy as np

# Left/right: angle of the wrist→middle-knuckle vector relative to vertical (compute_roll_degrees).
# Up/down: angle of the wrist→palm-center vector using only its Y and Z components (compute_palm_pitch_degrees).

# Increase this value if you want the hand to tilt more before detecting
# left/right. Decrease it to make left/right detection trigger sooner.
LEFT_RIGHT_TILT_THRESHOLD_DEGREES = 12.0

# Increase this value if you want the hand to tilt more before detecting
# up/down. Decrease it to make up/down detection trigger sooner.
UP_DOWN_DETECTION_THRESHOLD_DEGREES = 8.0

def to_vector(landmark):
    return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float64)


def compute_roll_degrees(hand_landmarks):
    """Measure left/right wrist tilt from the wrist-to-middle-finger direction."""
    wrist = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]
    delta_x = middle_mcp.x - wrist.x
    delta_y = middle_mcp.y - wrist.y

    # 0 degrees means the hand is upright. Positive means tilted right,
    # negative means tilted left.
    return math.degrees(math.atan2(delta_x, -delta_y))


def compute_palm_pitch_degrees(hand_landmarks):
    """Approximate up/down pitch using a wrist-to-palm-center axis."""
    wrist = to_vector(hand_landmarks[0])
    palm_center = (
        to_vector(hand_landmarks[5]) +
        to_vector(hand_landmarks[9]) +
        to_vector(hand_landmarks[13]) +
        to_vector(hand_landmarks[17])
    ) / 4.0

    # Keep the measurement inside the palm by using only the MCP row rather
    # than a normal influenced by finger spread. Comparing depth vs vertical
    # movement of this palm axis makes wrist pitch less sensitive to finger
    # pose and grab motion.
    palm_axis = palm_center - wrist
    yz_axis = np.array([0.0, palm_axis[1], palm_axis[2]], dtype=np.float64)
    magnitude = np.linalg.norm(yz_axis)
    if magnitude == 0:
        return 0.0

    yz_axis = yz_axis / magnitude
    return math.degrees(math.atan2(float(yz_axis[2]), float(-yz_axis[1])))


def classify_roll(roll_delta_degrees):
    # Compare current left/right tilt against the calibrated base pose.
    if roll_delta_degrees > LEFT_RIGHT_TILT_THRESHOLD_DEGREES:
        return "Right"
    if roll_delta_degrees < -LEFT_RIGHT_TILT_THRESHOLD_DEGREES:
        return "Left"
    return "Center"


def classify_pitch(pitch_delta_degrees):
    # Match left/right behavior: stay neutral near the base pose and only
    # switch to Up/Down after crossing the threshold.
    if abs(pitch_delta_degrees) < UP_DOWN_DETECTION_THRESHOLD_DEGREES:
        return "Neutral"

    return "Up" if pitch_delta_degrees > 0 else "Down"


def calibrate_wrist_base(hand_landmarks):
    # Save the current wrist/palm angles as the neutral reference pose.
    base_roll = compute_roll_degrees(hand_landmarks)
    base_pitch = compute_palm_pitch_degrees(hand_landmarks)

    return base_roll, base_pitch


def compute_wrist_state(hand_landmarks, base_roll_degrees, base_pitch_degrees):
    # Compute the new pose, compare it against the saved base pose, and turn
    # those angle differences into simple labels the rest of the app can use.
    current_roll = compute_roll_degrees(hand_landmarks)
    current_pitch = compute_palm_pitch_degrees(hand_landmarks)

    roll_delta = current_roll - base_roll_degrees
    pitch_delta = current_pitch - base_pitch_degrees
    roll_direction = classify_roll(roll_delta)
    pitch_direction = classify_pitch(pitch_delta)

    return {
        "roll_direction": roll_direction,
        "pitch_direction": pitch_direction,
        "roll_delta": roll_delta,
        "pitch_delta": pitch_delta,
    }


