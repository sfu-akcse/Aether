import os
import glob
import json
import threading
import time
import urllib.request
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from aether_logger import setup_logger
from xy_coordinate import draw_xy_coordinates, extract_xy_coordinates
from z_coordinate import extract_z_coordinate, label_z_coordinate
from base_rotation import base_rotation_x
from base_rotation import border_box

#cd /Users/admin/Aether
#source .host-venv/bin/activate
#./scripts/run_webcam_pipeline.sh host --port 8080

#cd /workspace
#CAMERA_SOURCE=http://host.docker.internal:8080/video.mjpg python3 src/main.py

logger = setup_logger('aether-system.log', 'AETHER.VISION')

# Hand Connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17),                                # Palm to Pinky Base
]


def is_headless_environment():
    # macOS native OpenCV windows do not require DISPLAY, unlike Linux/X11 setups.
    if os.name == 'posix' and hasattr(os, 'uname') and os.uname().sysname == 'Darwin':
        return False
    return not os.getenv('DISPLAY')


def resolve_camera_source():
    """Resolve CAMERA_SOURCE to either an int index or URL string."""
    raw_source = os.getenv('CAMERA_SOURCE', '0').strip()

    # Allow explicit integer camera indices (e.g. 0, 1, 2, -1)
    try:
        return int(raw_source)
    except ValueError:
        pass

    # Otherwise treat as URL/path supported by OpenCV VideoCapture
    return raw_source


def open_camera_capture(camera_source):
    """Open camera source with backend fallbacks for local camera indices."""
    candidates = []

    candidates.append(('CAP_ANY', cv2.CAP_ANY))

    for backend_name, backend in candidates:
        cap = cv2.VideoCapture(camera_source, backend)
        if cap.isOpened():
            logger.info("Opened camera source with backend=%s", backend_name)
            return cap
        cap.release()

    # Final fallback for environments where backend argument is ignored.
    cap = cv2.VideoCapture(camera_source)
    if cap.isOpened():
        logger.info("Opened camera source with backend=DEFAULT")
        return cap

    return cap


class LatestFrameReader:
    """Read frames on a background thread and expose only the newest frame.

    For URL sources, this parses MJPEG bytes directly to avoid OpenCV URL-decoder
    buffering/black-frame issues in some environments.
    """

    def __init__(self, camera_source, cap=None):
        self.camera_source = camera_source
        self.cap = cap
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_ts = 0.0
        self._running = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._is_url = isinstance(camera_source, str) and camera_source.startswith(('http://', 'https://'))
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

        if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
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
                req = urllib.request.Request(self.camera_source, headers={'User-Agent': 'AetherCV/1.0'})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    buffer = b''

                    while self._running:
                        chunk = resp.read(4096)
                        if not chunk:
                            raise RuntimeError('MJPEG stream ended unexpectedly')

                        buffer += chunk
                        start = buffer.find(b'\xff\xd8')
                        end = buffer.find(b'\xff\xd9', start + 2)

                        if start == -1 or end == -1:
                            # Prevent unbounded growth while waiting for JPEG markers.
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
                self._warn_throttled(f'Warning: MJPEG reader reconnecting ({exc})')
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


def draw_hand_landmarks(image, detection_result):
    """Draw detected hand landmarks on the image"""
    if not detection_result.hand_landmarks:
        return image

    h, w, _ = image.shape
    for hand_landmarks in detection_result.hand_landmarks:
        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)

        # Draw landmarks
        for landmark in hand_landmarks:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

    return image

def main():
    logger.info("Starting Aether vision pipeline.")

    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'hand_landmarker.task')
    detector = None
    reader = None

    try:
        # Reset MediaPipe HandLandmarker (Tasks API)
        # https://ai.google.dev/mediapipe/solutions/vision/hand_landmarker
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
        logger.info("MediaPipe HandLandmarker initialized with model=%s", model_path)

        camera_source = resolve_camera_source()
        cap = None
        headless = is_headless_environment()

        if headless:
            logger.info("No DISPLAY detected. Running in headless mode without OpenCV windows.")

        if isinstance(camera_source, int):
            cap = open_camera_capture(camera_source)

        if isinstance(camera_source, int) and not cap.isOpened():
            device_path = f"/dev/video{camera_source}"
            visible_devices = ", ".join(sorted(glob.glob('/dev/video*'))) or "none"

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
        logger.info("Frame reader started.")

        timestamp_ms = 0
        black_frame_count = 0
        waiting_warned = False
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)

        z_value = None
        base_value = None
        last_xyz = None

        while True:
            image, latest_ts = reader.get_latest()

            if image is None:
                if not headless:
                    status = placeholder.copy()
                    cv2.putText(status, 'Waiting for camera frames...', (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.imshow('MediaPipe Hands', status)
                    if cv2.waitKey(1) & 0xFF == 27:
                        logger.info("ESC pressed while waiting for camera frames.")
                        break
                else:
                    time.sleep(0.01)
                continue

            # Surface stale stream status without blocking UI interaction.
            frame_age = time.monotonic() - latest_ts
            if frame_age > 1.0 and not waiting_warned:
                logger.warning('Frame stream appears stale (>1s).')
                waiting_warned = True
            if frame_age <= 1.0:
                waiting_warned = False

            # Detect persistent blank stream frames and provide actionable guidance.
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

            # For local cameras mirror the image; host-side stream is already mirrored.
            if isinstance(camera_source, int):
                image = cv2.flip(image, 1)

            # BGR to RGB conversion for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            # Hand detection timestamp must be strictly increasing.
            now_ms = int(time.monotonic() * 1000)
            timestamp_ms = max(timestamp_ms + 1, now_ms)
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)

            # Draw landmarks on the original image
            image = draw_hand_landmarks(image, detection_result)
            # Draw XY coordinates on the original image
            image = draw_xy_coordinates(image, detection_result)
            # Create a border box
            image = border_box(image)

            xy_coordinates = extract_xy_coordinates(image, detection_result)
            z_coordinate, base_value = extract_z_coordinate(image, detection_result, z_value, base_value)
            image, base_value = label_z_coordinate(image, detection_result, z_value, base_value)

            base_rotation_x(xy_coordinates, image)

            if xy_coordinates is not None:
                xyz = {
                    'x': xy_coordinates['x'],
                    'y': xy_coordinates['y'],
                    'z': z_coordinate,
                }
                if xyz != last_xyz:
                    print(json.dumps(xyz), flush=True)
                    last_xyz = xyz

            if not headless:
                cv2.imshow('MediaPipe Hands', image)

                key = cv2.waitKey(1) & 0xFF

                if key == 27:  # Exit on 'ESC' key
                    logger.info("ESC pressed. Exiting vision loop.")
                    break
                elif key == ord('r'):
                    z_value = 0
                else:
                    z_value = None

        logger.info("Vision loop exited normally.")
        return 0
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down vision pipeline.")
        return 0
    except Exception:
        logger.exception("Vision pipeline terminated due to an unexpected error.")
        return 1
    finally:
        if reader is not None:
            reader.stop()
            logger.info("Frame reader stopped.")
        if detector is not None:
            detector.close()
            logger.info("HandLandmarker closed.")
        if not is_headless_environment():
            cv2.destroyAllWindows()
            logger.info("OpenCV windows destroyed.")
        logger.info("Aether vision pipeline shutdown complete.")


if __name__ == '__main__':
    raise SystemExit(main())
