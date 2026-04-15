import os
import glob
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
import tkinter as tk
from PIL import Image, ImageTk

# Hand Connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17),                                # Palm to Pinky Base
]


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
            print(f"Opened camera source with backend={backend_name}")
            return cap
        cap.release()

    # Final fallback for environments where backend argument is ignored.
    cap = cv2.VideoCapture(camera_source)
    if cap.isOpened():
        print("Opened camera source with backend=DEFAULT")
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
            print(message)
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


# Model path
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'hand_landmarker.task')

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

# Start video capture
camera_source = resolve_camera_source()
cap = None

if isinstance(camera_source, int):
    cap = open_camera_capture(camera_source)

if isinstance(camera_source, int) and not cap.isOpened():
    if isinstance(camera_source, int):
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

print(f"Using CAMERA_SOURCE={camera_source}")
reader = LatestFrameReader(camera_source, cap=cap)
reader.start()

timestamp_ms = 0
black_frame_count = 0
waiting_warned = False
# placeholder = np.zeros((480, 640, 3), dtype=np.uint8)

display_lock = threading.Lock()
latest_display_frame = None
shutdown_event = threading.Event()

def processing_loop():
    global latest_display_frame
    global timestamp_ms, black_frame_count, waiting_warned
 
    # continuously read the newest frame, run MediaPipe on it, draw landmarks, and save the processed frame for the UI
    while not shutdown_event.is_set():
        image, latest_ts = reader.get_latest()
 
        if image is None:
            time.sleep(0.01)
            continue
        
        # check whether the camera stream is delayed or stale
        frame_age = time.monotonic() - latest_ts
        if frame_age > 1.0 and not waiting_warned:
            print('Warning: frame stream appears stale (>1s).')
            waiting_warned = True
        if frame_age <= 1.0:
            waiting_warned = False
        
        # count repeated black frames to help detect blank camera output
        if image is not None and image.size > 0:
            if float(image.mean()) < 2.0:
                black_frame_count += 1
            else:
                black_frame_count = 0

        # print a warning if many black frames appear in a row
        if black_frame_count == 45:
            print(
                "Warning: received many near-black frames from CAMERA_SOURCE. "
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
 
        image = draw_hand_landmarks(image, detection_result)
 
        # save the processed frame so Tkinter can display it
        with display_lock:
            latest_display_frame = image
 
 
def wait_for_first_processed_frame(timeout_s=15.0):
    # wait until at least one processed frame is ready before opening the UI
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not shutdown_event.is_set():
        with display_lock:
            if latest_display_frame is not None:
                return True
        time.sleep(0.01)
    return False
 
 
def run_tkinter_ui():
    # do not open the window until the first processed frame is ready
    if not wait_for_first_processed_frame(timeout_s=15.0):
        raise RuntimeError("Timed out waiting for first processed frame.")
 
    root = tk.Tk()
    root.title("MediaPipe Hands")
    root.configure(bg="black")
 
    image_label = tk.Label(root, bg="black", borderwidth=0, highlightthickness=0)
    image_label.pack()
 
    # close the app when the window close button or ESC key is used
    def on_close(event=None):
        shutdown_event.set()
        if root.winfo_exists():
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.bind("<Escape>", on_close)
 
    def update_frame():
        if shutdown_event.is_set():
            if root.winfo_exists():
                root.destroy()
            return
 
        with display_lock:
            frame = None if latest_display_frame is None else latest_display_frame.copy()
 
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            tk_image = ImageTk.PhotoImage(image=pil_image)
            image_label.configure(image=tk_image)
            image_label.image = tk_image
 
        root.after(15, update_frame)
 
    update_frame()
    root.mainloop()
 
 
worker = threading.Thread(target=processing_loop, daemon=True)
worker.start()
 
try:
    run_tkinter_ui()
finally:
    shutdown_event.set()
    reader.stop()
    detector.close()