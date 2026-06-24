import glob
import json
import os
import threading
import time
import urllib.request

os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from GrabbingMotion2 import is_grabbing
from MultiHandTracker import HandSide, MultiHandTracker
from WristDetection import calibrate_wrist_base, compute_wrist_state
from aether_logger import setup_logger
from base_rotation import base_rotation_x, border_box, get_base_rotation_direction

#cd /Users/admin/Aether
#source .host-venv/bin/activate
#./scripts/run_webcam_pipeline.sh host --port 8080

#cd /workspace
#CAMERA_SOURCE=http://host.docker.internal:8080/video.mjpg python3 src/main2.py

logger = setup_logger("aether-system.log", "AETHER.VISION.MAIN2")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def is_headless_environment():
    if os.name == "posix" and hasattr(os, "uname") and os.uname().sysname == "Darwin":
        return False
    return not os.getenv("DISPLAY")


def resolve_camera_source():
    raw_source = os.getenv("CAMERA_SOURCE", "0").strip()
    try:
        return int(raw_source)
    except ValueError:
        return raw_source


def open_camera_capture(camera_source):
    candidates = [("CAP_ANY", cv2.CAP_ANY)]

    for backend_name, backend in candidates:
        cap = cv2.VideoCapture(camera_source, backend)
        if cap.isOpened():
            logger.info("Opened camera source with backend=%s", backend_name)
            return cap
        cap.release()

    cap = cv2.VideoCapture(camera_source)
    if cap.isOpened():
        logger.info("Opened camera source with backend=DEFAULT")
        return cap

    return cap


class LatestFrameReader:
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
            logger.warning(message)
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


def extract_xy_coordinates_for_hand(image, hand_landmarks):
    h, w, _ = image.shape
    box_size = min(h, w)
    x1 = (w - box_size) // 2
    y1 = (h - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size
    hand_indices = []

    for index in [0, 1, 2, 5, 9, 13, 17]:
        if 0.0 <= hand_landmarks[index].x <= 1.0 and 0.0 <= hand_landmarks[index].y <= 1.0:
            hand_indices.append(index)

    if not hand_indices:
        return None

    x_average = sum(hand_landmarks[index].x for index in hand_indices) / len(hand_indices)
    y_average = sum(hand_landmarks[index].y for index in hand_indices) / len(hand_indices)
    hand_x = int(x_average * w)
    hand_y = int(y_average * h)

    clamped_x = max(x1, min(hand_x, x2))
    clamped_y = max(y1, min(hand_y, y2))

    x_value = ((clamped_x - x1) / box_size) * 200 - 100
    y_value = 100 - ((clamped_y - y1) / box_size) * 200

    return {
        "pixel_x": hand_x,
        "pixel_y": hand_y,
        "x": round(x_value, 1),
        "y": round(y_value, 1),
    }


def draw_xy_coordinates_for_hand(image, xy_coordinates, color=(255, 0, 0)):
    if xy_coordinates is None:
        return image

    hand_x = xy_coordinates["pixel_x"]
    hand_y = xy_coordinates["pixel_y"]
    text = f"XYZ: {xy_coordinates['x']}, {xy_coordinates['y']}, {xy_coordinates.get('z', '-')}"
    cv2.circle(image, (hand_x, hand_y), 8, color, -1)
    cv2.putText(image, text, (hand_x + 12, hand_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image


def extract_z_coordinate_for_hand(image, hand_landmarks, z_reset_flag, base_value):
    h, w, _ = image.shape
    palm_indices = [0, 1, 5, 9, 13, 17]
    x_arr = [hand_landmarks[i].x * w for i in palm_indices]
    y_arr = [hand_landmarks[i].y * h for i in palm_indices]

    padding = 20
    min_x = max(0, int(min(x_arr)) - padding)
    max_x = min(w, int(max(x_arr)) + padding)
    min_y = max(0, int(min(y_arr)) - padding)
    max_y = min(h, int(max(y_arr)) + padding)

    box_width = max_x - min_x
    box_height = max_y - min_y
    box_area = box_width * box_height

    if z_reset_flag:
        base_value = box_area

    if base_value is None:
        return None, base_value, (min_x, min_y, max_x, max_y)

    z_offset = int(box_area ** 0.5 - base_value ** 0.5)
    return max(0, z_offset), base_value, (min_x, min_y, max_x, max_y)


def draw_z_overlay(image, z_coordinate, z_box):
    if z_box is None:
        return image

    min_x, min_y, max_x, max_y = z_box
    if z_coordinate is None:
        cv2.putText(image, "Press 'r' to set right-hand Z=0", (min_x, max(20, min_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(image, f"Z: {z_coordinate}", (min_x, max(20, min_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
    return image


def get_hand_label_position(image, hand_landmarks):
    h, w, _ = image.shape
    x_values = [int(landmark.x * w) for landmark in hand_landmarks]
    y_values = [int(landmark.y * h) for landmark in hand_landmarks]
    return int(sum(x_values) / len(x_values)), max(25, min(y_values) - 15)


def draw_hand_name(image, hand_name, hand_landmarks, color):
    label_x, label_y = get_hand_label_position(image, hand_landmarks)
    cv2.putText(image, hand_name, (label_x - 45, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_top_summary(image, summary_lines):
    overlay = image.copy()
    banner_height = 28 + (len(summary_lines) * 24)
    cv2.rectangle(overlay, (0, 0), (image.shape[1], banner_height), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

    for index, line in enumerate(summary_lines):
        y_position = 28 + (index * 24)
        cv2.putText(image, line, (20, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)


def main():
    logger.info("Starting Aether dual-hand pipeline.")

    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "hand_landmarker.task")
    detector = None
    reader = None

    try:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO,
        )
        detector = vision.HandLandmarker.create_from_options(options)
        tracker = MultiHandTracker(is_mirrored=True)
        logger.info("MediaPipe HandLandmarker initialized with model=%s", model_path)

        camera_source = resolve_camera_source()
        cap = None
        headless = is_headless_environment()

        if isinstance(camera_source, int):
            cap = open_camera_capture(camera_source)

        if isinstance(camera_source, int) and not cap.isOpened():
            device_path = f"/dev/video{camera_source}"
            visible_devices = ", ".join(sorted(glob.glob("/dev/video*"))) or "none"

            if not os.path.exists(device_path):
                raise RuntimeError(
                    "Failed to open camera source: "
                    f"{camera_source}. "
                    f"{device_path} is not available in this container. "
                    f"Visible camera devices: {visible_devices}. "
                    "If you are running in a Linux devcontainer, pass through your camera "
                    "with run args like `--device=/dev/video0:/dev/video0` "
                    "(optionally `--group-add=video`) and rebuild the container, "
                    "or use a stream URL for CAMERA_SOURCE."
                )

            raise RuntimeError(
                "Failed to open camera source: "
                f"{camera_source}. "
                "Set CAMERA_SOURCE to an index (e.g. 0) or stream URL "
                "(e.g. http://host.docker.internal:8080/video.mjpg)."
            )

        logger.info("Using CAMERA_SOURCE=%s", camera_source)
        reader = LatestFrameReader(camera_source, cap=cap)
        reader.start()

        timestamp_ms = 0
        black_frame_count = 0
        waiting_warned = False
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)

        right_hand_z_base = None
        left_hand_base_roll = None
        left_hand_base_pitch = None

        while True:
            image, latest_ts = reader.get_latest()

            if image is None:
                if not headless:
                    status = placeholder.copy()
                    cv2.putText(status, "Waiting for camera frames...", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.imshow("Aether Main2", status)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                else:
                    time.sleep(0.01)
                continue

            frame_age = time.monotonic() - latest_ts
            if frame_age > 1.0 and not waiting_warned:
                logger.warning("Frame stream appears stale (>1s).")
                waiting_warned = True
            if frame_age <= 1.0:
                waiting_warned = False

            if image.size > 0:
                if float(image.mean()) < 2.0:
                    black_frame_count += 1
                else:
                    black_frame_count = 0

            if black_frame_count == 45:
                logger.warning(
                    "Received many near-black frames from CAMERA_SOURCE. "
                    "If using host stream, verify host preview at http://localhost:8080/ "
                    "and try another host camera index."
                )

            if isinstance(camera_source, int):
                image = cv2.flip(image, 1)

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            now_ms = int(time.monotonic() * 1000)
            timestamp_ms = max(timestamp_ms + 1, now_ms)
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)
            tracker.update(detection_result)

            image = border_box(image)

            right_state = tracker.get_hand(HandSide.RIGHT)
            left_state = tracker.get_hand(HandSide.LEFT)

            right_xyz = None
            right_z = None
            right_z_box = None
            right_base_rotation = "No hand"

            left_xy = None
            left_grab = "No hand"
            left_wrist = None
            left_base_rotation = "No hand"

            if right_state is not None:
                right_hand = right_state.landmarks
                draw_hand_landmarks(image, right_hand)
                draw_hand_name(image, "Right Hand", right_hand, (255, 120, 120))

                right_xyz = extract_xy_coordinates_for_hand(image, right_hand)
                right_z, right_hand_z_base, right_z_box = extract_z_coordinate_for_hand(
                    image,
                    right_hand,
                    False,
                    right_hand_z_base,
                )

                if right_xyz is not None:
                    right_xyz["z"] = right_z
                    right_base_rotation = get_base_rotation_direction(right_xyz)
                    image = draw_xy_coordinates_for_hand(image, right_xyz, color=(255, 0, 0))
                    base_rotation_x(right_xyz, image)

                image = draw_z_overlay(image, right_z, right_z_box)
            else:
                cv2.putText(image, "Right Hand not detected", (20, image.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if left_state is not None:
                left_hand = left_state.landmarks
                draw_hand_landmarks(image, left_hand)
                draw_hand_name(image, "Left Hand", left_hand, (120, 220, 255))

                left_xy = extract_xy_coordinates_for_hand(image, left_hand)
                left_base_rotation = get_base_rotation_direction(left_xy)
                if left_xy is not None:
                    base_rotation_x(left_xy, image)

                left_grab = "Grabbing" if is_grabbing(left_hand) else "Open"
                if left_hand_base_roll is not None and left_hand_base_pitch is not None:
                    left_wrist = compute_wrist_state(
                        left_hand,
                        "Left",
                        left_hand_base_roll,
                        left_hand_base_pitch,
                    )
            else:
                cv2.putText(image, "Left Hand not detected", (20, image.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            left_wrist_lr = left_wrist["roll_direction"] if left_wrist else "Not calibrated"
            left_wrist_ud = left_wrist["pitch_direction"] if left_wrist else "Not calibrated"
            right_xyz_text = (
                f"{right_xyz['x']}, {right_xyz['y']}, {right_xyz['z']}"
                if right_xyz is not None
                else "No hand"
            )

            summary_lines = [
                f"Left Hand | Grab: {left_grab}",
                f"Left Wrist | LR: {left_wrist_lr} | UD: {left_wrist_ud}",
                f"Right Hand | XYZ: {right_xyz_text}",
                f"Base Rotation | Left: {left_base_rotation} | Right: {right_base_rotation}",
            ]
            draw_top_summary(image, summary_lines)

            if left_hand_base_roll is None or left_hand_base_pitch is None:
                cv2.putText(image, "Press 'b' to set left wrist base", (20, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            else:
                cv2.putText(image, "b = reset left wrist base", (20, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            if right_hand_z_base is None:
                cv2.putText(image, "Press 'r' to set right-hand Z=0", (320, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            else:
                cv2.putText(image, "r = reset right-hand Z=0", (320, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            terminal_state = {
                "left_hand": {
                    "grab": left_grab,
                    "wrist": {
                        "up_down": left_wrist["pitch_direction"] if left_wrist else "Not calibrated",
                        "left_right_rotation": left_wrist["roll_direction"] if left_wrist else "Not calibrated",
                    },
                    "base_rotation": left_base_rotation,
                },
                "right_hand": {
                    "xyz": right_xyz,
                    "base_rotation": right_base_rotation,
                },
            }
            print(json.dumps(terminal_state), flush=True)

            if not headless:
                cv2.imshow("Aether Main2", image)
                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    break
                if key == ord("b") and left_state is not None:
                    left_hand = left_state.landmarks
                    left_hand_base_roll, left_hand_base_pitch = calibrate_wrist_base(left_hand, "Left")
                if key == ord("r") and right_state is not None:
                    right_hand = right_state.landmarks
                    _, right_hand_z_base, _ = extract_z_coordinate_for_hand(
                        image,
                        right_hand,
                        True,
                        right_hand_z_base,
                    )

        return 0
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down main2.")
        return 0
    except Exception:
        logger.exception("main2 terminated due to an unexpected error.")
        return 1
    finally:
        if reader is not None:
            reader.stop()
        if detector is not None:
            detector.close()
        if not is_headless_environment():
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
