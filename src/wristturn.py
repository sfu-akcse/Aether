from enum import Enum


class WristDirection(Enum):
    NEUTRAL = "NEUTRAL"
    LEFT    = "LEFT"
    RIGHT   = "RIGHT"

# x_diff threshold — abs(pinky_x - index_x) above this = neutral, below = turned
NEUTRAL_X_DIFF_THRESHOLD = 0.1


# determine hand side from landmark positions
def get_hand_side(hand_landmarks) -> str:
    thumb = hand_landmarks[1]
    pinky = hand_landmarks[17]

    if pinky.x < thumb.x:
        return "Left"
    else:
        return "Right"

# detect wrist direction for a single hand
def detect_wrist(hand_landmarks, hand_side: str) -> WristDirection:
    pinky = hand_landmarks[17]
    index = hand_landmarks[5]

    x_diff = abs(pinky.x - index.x)

    if x_diff > NEUTRAL_X_DIFF_THRESHOLD:
        return WristDirection.NEUTRAL
    else:
        if hand_side == "Left":
            return WristDirection.RIGHT
        else:
            return WristDirection.LEFT


# run detect_wrist for every detected hand
def detect_all_wrists(detection_result, detectors: dict):
    results = []

    for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
        hand_side = get_hand_side(hand_landmarks)
        direction = detect_wrist(hand_landmarks, hand_side)
        results.append((hand_side, direction))
    
    return results