from enum import Enum


class WristDirection(Enum):
    NEUTRAL = "NEUTRAL"
    LEFT    = "LEFT"
    RIGHT   = "RIGHT"

# x_diff threshold — abs(pinky_x - index_x) above this = neutral, below = turned
NEUTRAL_X_DIFF_THRESHOLD = 0.1

# EMA smoothing factor — 0.7~0.9 is a common range for smoothing.
# Higher values make the algorithm trust the previous values more (smoother but slower response)
# Lower values make the algorithm trust the current values more (faster but more jitter)
EMA_ALPHA = 0.8

# Hand-specific smoothed x_diff storage {"Left": float, "Right": float}
_smoothed_x_diff: dict[str, float] = {}

# determine hand side from landmark positions
def get_hand_side(hand_landmarks) -> str:
    thumb = hand_landmarks[1]
    pinky = hand_landmarks[17]

    if pinky.x < thumb.x:
        return "Left"
    else:
        return "Right"


def _apply_ema(hand_side: str, current_x_diff: float) -> float:
    """ Apply EMA(Exponential Moving Average).
    Use current value for the first frame, then apply smoothing."""
    if hand_side not in _smoothed_x_diff:
        _smoothed_x_diff[hand_side] = current_x_diff
    else:
        _smoothed_x_diff[hand_side] = (
            EMA_ALPHA * _smoothed_x_diff[hand_side]
            + (1 - EMA_ALPHA) * current_x_diff
        )
    return _smoothed_x_diff[hand_side]


def detect_wrist(hand_landmarks, hand_side: str) -> WristDirection:
    pinky = hand_landmarks[17]
    index = hand_landmarks[5]

    raw_x_diff = abs(pinky.x - index.x)

    # Apply EMA smoothing to the raw x_diff to reduce jitter
    smoothed = _apply_ema(hand_side, raw_x_diff)

    if smoothed > NEUTRAL_X_DIFF_THRESHOLD:
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