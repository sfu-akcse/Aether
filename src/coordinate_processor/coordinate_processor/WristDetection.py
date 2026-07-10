import math
import os
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

#up/down motion is done through creating two vectors, one from wrist to pinky and one from wrist to index. taking the cross product between the two gives 
#us the normal vector which is pratically a arrow coming out of the palm. depending on the angle of this vector compared to the base position, it 
#detects if it is going up or down.

#left/right motion is done through having a vector from middle to wrist. we compare that vector with just a straight vertical line and if the angle between 
#the two goes over the threshold, it detects as turning

# Increase this value if you want the hand to tilt more before detecting
# left/right. Decrease it to make left/right detection trigger sooner.
LEFT_RIGHT_TILT_THRESHOLD_DEGREES = 12.0

# Increase this value if you want the hand to tilt more before detecting
# up/down. Decrease it to make up/down detection trigger sooner.
UP_DOWN_DETECTION_THRESHOLD_DEGREES = 8.0

# Increase this value if you want it to take more tilt to reach 100.
# Lower it if you want the 0-100 score to rise faster.
UP_DOWN_FULL_SCALE_DEGREES = 35.0


def resolve_camera_source():
    raw_source = os.getenv("CAMERA_SOURCE", "http://host.docker.internal:8080/video.mjpg").strip()

    try:
        return int(raw_source)
    except ValueError:
        return raw_source


def open_camera_capture(camera_source):
    cap = cv2.VideoCapture(camera_source, cv2.CAP_ANY)
    if cap.isOpened():
        return cap

    cap.release()
    return cv2.VideoCapture(camera_source)


class LatestFrameReader:
    """Continuously read frames and keep only the most recent one."""

    def __init__(self, camera_source, cap=None):
        self.camera_source = camera_source
        self.cap = cap
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_ts = 0.0
        self._running = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._is_url = isinstance(camera_source, str) and camera_source.startswith(("http://", "https://"))
        self._last_warn_at = 0.0

    def start(self):
        self._running = True
        self._thread.start()

    def _run(self):
        if self._is_url:
            self._run_mjpeg_url()
        else:
            self._run_opencv_capture()

    def _run_opencv_capture(self):
        if self.cap is None:
            self.cap = open_camera_capture(self.camera_source)

        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while self._running:
            success, frame = self.cap.read()
            if not success:
                time.sleep(0.01)
                continue

            with self._lock:
                self._latest_frame = frame
                self._latest_ts = time.monotonic()

    def _warn_throttled(self, message):
        now = time.monotonic()
        if now - self._last_warn_at > 2.0:
            print(message)
            self._last_warn_at = now

    def _run_mjpeg_url(self):
        while self._running:
            try:
                req = urllib.request.Request(self.camera_source, headers={"User-Agent": "AetherCV/1.0"})
                with urllib.request.urlopen(req, timeout=8) as response:
                    buffer = b""

                    while self._running:
                        chunk = response.read(4096)
                        if not chunk:
                            raise RuntimeError("MJPEG stream ended unexpectedly")

                        buffer += chunk
                        start = buffer.find(b"\xff\xd8")
                        end = buffer.find(b"\xff\xd9", start + 2)

                        if start == -1 or end == -1:
                            if len(buffer) > 1_000_000:
                                buffer = buffer[-200_000:]
                            continue

                        jpg = buffer[start:end + 2]
                        buffer = buffer[end + 2:]

                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is None:
                            continue

                        with self._lock:
                            self._latest_frame = frame
                            self._latest_ts = time.monotonic()
            except Exception as exc:
                self._warn_throttled(f"Warning: MJPEG reader reconnecting ({exc})")
                time.sleep(0.2)

    def get_latest(self):
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0
            return self._latest_frame.copy(), self._latest_ts

    def stop(self):
        self._running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self._thread.join(timeout=0.5)


def draw_hand_landmarks(frame, hand_landmarks):
    height, width, _ = frame.shape

    for start_idx, end_idx in HAND_CONNECTIONS:
        start = hand_landmarks[start_idx]
        end = hand_landmarks[end_idx]
        start_point = (int(start.x * width), int(start.y * height))
        end_point = (int(end.x * width), int(end.y * height))
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    for landmark in hand_landmarks:
        point = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(frame, point, 4, (0, 0, 255), -1)


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


def compute_palm_pitch_degrees(hand_landmarks, handedness_label):
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


def get_handedness_label(detection_result):
    if not getattr(detection_result, "handedness", None):
        return "Right"

    handedness = detection_result.handedness[0]
    if not handedness:
        return "Right"

    return handedness[0].category_name or "Right"


def calibrate_wrist_base(hand_landmarks, handedness_label):
    # Save the current wrist/palm angles as the neutral reference pose.
    base_roll = compute_roll_degrees(hand_landmarks)
    base_pitch = compute_palm_pitch_degrees(hand_landmarks, handedness_label)

    return base_roll, base_pitch


def compute_wrist_state(hand_landmarks, handedness_label, base_roll_degrees, base_pitch_degrees):
    # Compute the new pose, compare it against the saved base pose, and turn
    # those angle differences into simple labels the rest of the app can use.
    current_roll = compute_roll_degrees(hand_landmarks)
    current_pitch = compute_palm_pitch_degrees(hand_landmarks, handedness_label)

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


def main():
    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "hand_landmarker.task")
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        running_mode=vision.RunningMode.VIDEO,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    camera_source = resolve_camera_source()
    cap = None

    if isinstance(camera_source, int):
        cap = open_camera_capture(camera_source)

    reader = LatestFrameReader(camera_source, cap=cap)
    reader.start()

    timestamp_ms = 0
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    base_roll_degrees = None
    base_pitch_degrees = None

    try:
        while True:
            frame, latest_ts = reader.get_latest()

            if frame is None:
                status_frame = placeholder.copy()
                cv2.putText(status_frame, "Waiting for camera frames...", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.imshow("Wrist Detection", status_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            if frame.shape[1] > 480:
                scale = 480 / frame.shape[1]
                frame = cv2.resize(frame, (480, max(1, int(frame.shape[0] * scale))))

            if isinstance(camera_source, int):
                frame = cv2.flip(frame, 1)

            frame_age = time.monotonic() - latest_ts
            if frame_age > 1.0:
                cv2.putText(frame, "Stream stale", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                cv2.imshow("Wrist Detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            now_ms = int(time.monotonic() * 1000)
            timestamp_ms = max(timestamp_ms + 1, now_ms)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                handedness_label = get_handedness_label(result)
                draw_hand_landmarks(frame, hand_landmarks)

                if base_roll_degrees is None or base_pitch_degrees is None:
                    cv2.putText(frame, "Press 'b' to set base hand position", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    wrist_state = compute_wrist_state(hand_landmarks, handedness_label, base_roll_degrees, base_pitch_degrees)
                    cv2.putText(frame, f"Left/Right: {wrist_state['roll_direction']}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Up/Down: {wrist_state['pitch_direction']} {wrist_state['pitch_score']}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.putText(frame, "Press 'b' to set base position", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Wrist Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("b") and result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                handedness_label = get_handedness_label(result)
                base_roll_degrees, base_pitch_degrees = calibrate_wrist_base(hand_landmarks, handedness_label)
    finally:
        reader.stop()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
