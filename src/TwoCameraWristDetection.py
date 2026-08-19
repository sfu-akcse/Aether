import math
import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from WristDetection import (
    LatestFrameReader,
    classify_pitch,
    classify_roll,
    compute_roll_degrees,
    draw_hand_landmarks,
    open_camera_capture,
)

# Left/right comes from the front camera (reuses WristDetection's roll calc).
# Up/down comes from the side camera below — it sees vertical movement
# directly, no depth estimate needed.


def is_headless_environment() -> bool:
    if os.name == "posix" and hasattr(os, "uname") and os.uname().sysname == "Darwin":
        return False
    return not os.getenv("DISPLAY")


def resolve_camera_source(env_var, default):
    raw_source = os.getenv(env_var, default).strip()
    try:
        return int(raw_source)
    except ValueError:
        return raw_source


def resolve_front_camera_source():
    return resolve_camera_source("CAMERA_SOURCE_FRONT", "http://host.docker.internal:8080/video.mjpg")


def resolve_side_camera_source():
    return resolve_camera_source("CAMERA_SOURCE_SIDE", "http://host.docker.internal:8081/video.mjpg")


# Same 0/10/.../50 scoring used elsewhere (main_single_hand.py, motion_recorder.py).
SNAP_BAND = 18  # degrees per output step
SNAP_MAX = 50   # max output value


def snap_degrees(value):
    """Snap to 0, 10, 20, … SNAP_MAX with each output step spanning SNAP_BAND degrees."""
    abs_val = abs(value)
    sign = 1 if value >= 0 else -1
    step_count = round(abs_val / SNAP_BAND)
    return int(sign * min(SNAP_MAX, step_count * 10))


def compute_side_pitch_degrees(hand_landmarks):
    """Angle of the wrist-to-palm-top vector in the side camera's image.

    0 = palm level with wrist. Positive = raised (Up), negative = lowered (Down).
    """
    wrist = hand_landmarks[0]
    palm_top = hand_landmarks[9]
    delta_x = palm_top.x - wrist.x
    delta_y = palm_top.y - wrist.y
    return math.degrees(math.atan2(-delta_y, delta_x))


def calibrate_wrist_base(front_hand_landmarks, side_hand_landmarks,
                          base_roll_degrees=None, base_pitch_degrees=None):
    """Save the current angles as the neutral pose.

    Only updates an axis if its camera has a hand in view, so calibrating
    with one camera doesn't wipe out the other axis's baseline.
    """
    if front_hand_landmarks is not None:
        base_roll_degrees = compute_roll_degrees(front_hand_landmarks)
    if side_hand_landmarks is not None:
        base_pitch_degrees = compute_side_pitch_degrees(side_hand_landmarks)
    return base_roll_degrees, base_pitch_degrees


def compute_wrist_state(front_hand_landmarks, side_hand_landmarks, base_roll_degrees, base_pitch_degrees):
    """Combine front left/right with side up/down into one wrist state.

    Either axis is None if its camera has no hand in view or isn't calibrated yet.
    """
    roll_direction = roll_delta = None
    if front_hand_landmarks is not None and base_roll_degrees is not None:
        roll_delta = compute_roll_degrees(front_hand_landmarks) - base_roll_degrees
        roll_direction = classify_roll(roll_delta)

    pitch_direction = pitch_delta = None
    if side_hand_landmarks is not None and base_pitch_degrees is not None:
        pitch_delta = compute_side_pitch_degrees(side_hand_landmarks) - base_pitch_degrees
        pitch_direction = classify_pitch(pitch_delta)

    return {
        "roll_direction": roll_direction,
        "pitch_direction": pitch_direction,
        "roll_delta": roll_delta,
        "pitch_delta": pitch_delta,
        "roll_deg": snap_degrees(roll_delta) if roll_delta is not None else None,
        "pitch_deg": snap_degrees(pitch_delta) if pitch_delta is not None else None,
    }


def _create_detector(model_path):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HandLandmarker.create_from_options(options)


def _detect(detector, frame, timestamp_ms):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    now_ms = int(time.monotonic() * 1000)
    timestamp_ms = max(timestamp_ms + 1, now_ms)
    result = detector.detect_for_video(mp_image, timestamp_ms)
    hand_landmarks = result.hand_landmarks[0] if result.hand_landmarks else None
    return hand_landmarks, timestamp_ms


def main():
    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "hand_landmarker.task")
    front_detector = _create_detector(model_path)
    side_detector = _create_detector(model_path)

    front_source = resolve_front_camera_source()
    side_source = resolve_side_camera_source()
    headless = is_headless_environment()

    front_cap = open_camera_capture(front_source) if isinstance(front_source, int) else None
    side_cap = open_camera_capture(side_source) if isinstance(side_source, int) else None

    front_reader = LatestFrameReader(front_source, cap=front_cap)
    side_reader = LatestFrameReader(side_source, cap=side_cap)
    front_reader.start()
    side_reader.start()

    front_timestamp_ms = 0
    side_timestamp_ms = 0
    base_roll_degrees = None
    base_pitch_degrees = None
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)

    print("[KEYS] b = calibrate neutral wrist baseline (works with just one camera connected) | ESC = quit")

    try:
        while True:
            front_frame, _ = front_reader.get_latest()
            side_frame, _ = side_reader.get_latest()

            front_hand = None
            if front_frame is not None:
                if isinstance(front_source, int):
                    front_frame = cv2.flip(front_frame, 1)
                front_hand, front_timestamp_ms = _detect(front_detector, front_frame, front_timestamp_ms)
                if front_hand is not None:
                    draw_hand_landmarks(front_frame, front_hand)

            side_hand = None
            if side_frame is not None:
                if isinstance(side_source, int):
                    side_frame = cv2.flip(side_frame, 1)
                side_hand, side_timestamp_ms = _detect(side_detector, side_frame, side_timestamp_ms)
                if side_hand is not None:
                    draw_hand_landmarks(side_frame, side_hand)

            wrist_state = compute_wrist_state(front_hand, side_hand, base_roll_degrees, base_pitch_degrees)
            if wrist_state["roll_direction"] is not None:
                roll_text = f"{wrist_state['roll_direction']} ({wrist_state['roll_deg']}°)"
            elif front_frame is None:
                roll_text = "No front camera"
            elif front_hand is None:
                roll_text = "No hand"
            else:
                roll_text = "Not calibrated"
            if wrist_state["pitch_direction"] is not None:
                pitch_text = f"{wrist_state['pitch_direction']} ({wrist_state['pitch_deg']}°)"
            elif side_frame is None:
                pitch_text = "No side camera"
            elif side_hand is None:
                pitch_text = "No hand"
            else:
                pitch_text = "Not calibrated"
            label = f"L/R: {roll_text}  U/D: {pitch_text}"
            color = (0, 255, 0) if (wrist_state["roll_direction"] or wrist_state["pitch_direction"]) else (0, 165, 255)

            if not headless:
                if front_frame is not None:
                    cv2.putText(front_frame, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.imshow("Front View", front_frame)
                else:
                    front_status = placeholder.copy()
                    cv2.putText(front_status, "No front camera stream", (20, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imshow("Front View", front_status)

                if side_frame is not None:
                    cv2.putText(side_frame, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.imshow("Side View", side_frame)
                else:
                    side_status = placeholder.copy()
                    cv2.putText(side_status, "No side camera stream", (20, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imshow("Side View", side_status)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key == ord("b") and (front_hand is not None or side_hand is not None):
                    base_roll_degrees, base_pitch_degrees = calibrate_wrist_base(
                        front_hand, side_hand, base_roll_degrees, base_pitch_degrees
                    )
                    print("[CALIBRATED] Neutral wrist baseline updated.")
            else:
                time.sleep(0.01)
    finally:
        front_reader.stop()
        side_reader.stop()
        front_detector.close()
        side_detector.close()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
