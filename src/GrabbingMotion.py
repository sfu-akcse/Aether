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

#cd /workspace CAMERA_SOURCE=http://host.docker.internal:8080/video.mjpg python3 src/GrabbingMotion.py
#run outside of devcontainter. cd /Users/admin/Aether source .host-venv/bin/activate ./scripts/run_webcam_pipeline.sh host --port 8080
#im using venv cause my mediapipe installation is messed up

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


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


def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def is_grabbing(hand_landmarks):
    wrist = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]
    hand_size = distance(wrist, middle_mcp)

    if hand_size == 0:
        return False

    fingertips = [8, 12, 16, 20]
    closed_fingers = 0

    for tip in fingertips:
        normalized_distance = distance(hand_landmarks[tip], wrist) / hand_size
        if normalized_distance < 1.3:
            closed_fingers += 1

    return closed_fingers >= 4


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
last_status = "Open"

try:
    while True:
        frame, latest_ts = reader.get_latest()

        if frame is None:
            status_frame = placeholder.copy()
            cv2.putText(status_frame, "Waiting for camera frames...", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow("Grab Detection", status_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        if frame.shape[1] > 640:
            scale = 640 / frame.shape[1]
            frame = cv2.resize(frame, (640, max(1, int(frame.shape[0] * scale))))

        if isinstance(camera_source, int):
            frame = cv2.flip(frame, 1)

        frame_age = time.monotonic() - latest_ts
        status = last_status

        if frame_age <= 1.0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            now_ms = int(time.monotonic() * 1000)
            timestamp_ms = max(timestamp_ms + 1, now_ms)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            status = "Open"
            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    draw_hand_landmarks(frame, hand_landmarks)
                    status = "Grabbing" if is_grabbing(hand_landmarks) else "Open"
        else:
            cv2.putText(frame, "Stream stale", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        last_status = status

        cv2.putText(frame, f"Hand: {status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Grab Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    reader.stop()
    cv2.destroyAllWindows()
    detector.close()
