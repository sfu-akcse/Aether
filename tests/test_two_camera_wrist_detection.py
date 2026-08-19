import pytest

from TwoCameraWristDetection import (
    calibrate_wrist_base,
    compute_side_pitch_degrees,
    compute_wrist_state,
    snap_degrees,
)


class FakeLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def make_landmarks(overrides):
    landmarks = [FakeLandmark(0.0, 0.0, 0.0) for _ in range(21)]
    for index, (x, y) in overrides.items():
        landmarks[index] = FakeLandmark(x, y)
    return landmarks


# Level side view: palm-top beside the wrist, no vertical offset.
LEVEL_SIDE = make_landmarks({0: (0.0, 0.0), 9: (1.0, 0.0)})
# Neutral front view: palm-top straight above the wrist (upright hand).
NEUTRAL_FRONT = make_landmarks({0: (0.0, 0.0), 9: (0.0, -1.0)})


def test_compute_side_pitch_degrees_zero_when_level():
    assert compute_side_pitch_degrees(LEVEL_SIDE) == pytest.approx(0.0)


def test_compute_side_pitch_degrees_positive_when_palm_raised():
    raised = make_landmarks({0: (0.0, 0.0), 9: (1.0, -1.0)})
    assert compute_side_pitch_degrees(raised) == pytest.approx(45.0)


def test_compute_side_pitch_degrees_negative_when_palm_lowered():
    lowered = make_landmarks({0: (0.0, 0.0), 9: (1.0, 1.0)})
    assert compute_side_pitch_degrees(lowered) == pytest.approx(-45.0)


def test_calibrate_wrist_base_uses_front_roll_and_side_pitch():
    base_roll, base_pitch = calibrate_wrist_base(NEUTRAL_FRONT, LEVEL_SIDE)
    assert base_roll == pytest.approx(0.0)
    assert base_pitch == pytest.approx(0.0)


def test_compute_wrist_state_classifies_right_and_up():
    front_tilted_right = make_landmarks({0: (0.0, 0.0), 9: (0.5, -1.0)})
    side_palm_raised = make_landmarks({0: (0.0, 0.0), 9: (1.0, -0.5)})

    base_roll, base_pitch = calibrate_wrist_base(NEUTRAL_FRONT, LEVEL_SIDE)
    state = compute_wrist_state(front_tilted_right, side_palm_raised, base_roll, base_pitch)

    assert state["roll_direction"] == "Right"
    assert state["pitch_direction"] == "Up"
    assert state["roll_delta"] == pytest.approx(26.565, abs=0.01)
    assert state["pitch_delta"] == pytest.approx(26.565, abs=0.01)
    assert state["roll_deg"] == 10
    assert state["pitch_deg"] == 10


def test_compute_wrist_state_stays_neutral_within_threshold():
    # Small tilt, under both classify_roll/classify_pitch thresholds.
    front_slight = make_landmarks({0: (0.0, 0.0), 9: (0.05, -1.0)})
    side_slight = make_landmarks({0: (0.0, 0.0), 9: (1.0, -0.05)})

    base_roll, base_pitch = calibrate_wrist_base(NEUTRAL_FRONT, LEVEL_SIDE)
    state = compute_wrist_state(front_slight, side_slight, base_roll, base_pitch)

    assert state["roll_direction"] == "Center"
    assert state["pitch_direction"] == "Neutral"


# --- Single-camera fallback: only one camera connected shouldn't block the other axis. ---

def test_calibrate_wrist_base_only_front_connected_preserves_existing_pitch():
    # base_pitch was already set earlier; only the front hand is in view now.
    base_roll, base_pitch = calibrate_wrist_base(NEUTRAL_FRONT, None,
                                                   base_roll_degrees=None, base_pitch_degrees=99.0)
    assert base_roll == pytest.approx(0.0)
    assert base_pitch == pytest.approx(99.0)  # untouched, not clobbered to None


def test_calibrate_wrist_base_only_side_connected_preserves_existing_roll():
    base_roll, base_pitch = calibrate_wrist_base(None, LEVEL_SIDE,
                                                   base_roll_degrees=12.0, base_pitch_degrees=None)
    assert base_roll == pytest.approx(12.0)
    assert base_pitch == pytest.approx(0.0)


def test_compute_wrist_state_only_front_camera_available():
    front_tilted_right = make_landmarks({0: (0.0, 0.0), 9: (0.5, -1.0)})
    base_roll, base_pitch = calibrate_wrist_base(NEUTRAL_FRONT, LEVEL_SIDE)

    # No side hand this frame (e.g. side camera not connected at all).
    state = compute_wrist_state(front_tilted_right, None, base_roll, base_pitch)

    assert state["roll_direction"] == "Right"
    assert state["pitch_direction"] is None
    assert state["pitch_delta"] is None
    assert state["pitch_deg"] is None


def test_compute_wrist_state_no_baseline_yet_returns_none():
    front_tilted_right = make_landmarks({0: (0.0, 0.0), 9: (0.5, -1.0)})
    state = compute_wrist_state(front_tilted_right, LEVEL_SIDE, None, None)

    assert state["roll_direction"] is None
    assert state["pitch_direction"] is None


# --- snap_degrees: same 0/10/.../50 scoring used for both roll and pitch. ---

def test_snap_degrees_rounds_to_nearest_ten():
    assert snap_degrees(0.0) == 0
    assert snap_degrees(18.0) == 10
    assert snap_degrees(-18.0) == -10


def test_snap_degrees_caps_at_max():
    assert snap_degrees(100.0) == 50
    assert snap_degrees(-100.0) == -50