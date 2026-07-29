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

from ControlModeManager import ControlMode
from GrabbingMotion2 import is_grabbing
from WristDetection import calibrate_wrist_base, compute_wrist_state
from aether_logger import setup_logger
from base_rotation import base_rotation_x, border_box, get_base_rotation_direction

logger = setup_logger("aether-system.log", "AETHER.VISION.SINGLE")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

_MODE_COLOR = {
    ControlMode.XYZ:   (180, 60,  0),
    ControlMode.WRIST: (0,  160, 60),
}
_MODE_LABEL = {
    ControlMode.XYZ:   "q: XYZ",
    ControlMode.WRIST: "w: WRIST+GRAB",
}


# camera helpers

def is_headless_environment() -> bool:
    if os.name == "posix" and hasattr(os, "uname") and os.uname().sysname == "Darwin":
        return False
    return not os.getenv("DISPLAY")


def resolve_camera_source():
    raw = os.getenv("CAMERA_SOURCE", "0").strip()
    try:
        return int(raw)
    except ValueError:
        return raw


def open_camera_capture(source):
    cap = cv2.VideoCapture(source, cv2.CAP_ANY)
    if cap.isOpened():
        logger.info("Opened camera with CAP_ANY backend")
        return cap
    cap.release()
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        logger.info("Opened camera with DEFAULT backend")
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
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest_frame = frame
                self._latest_ts = time.monotonic()

    def _warn_throttled(self, msg):
        now = time.monotonic()
        if now - self._last_warn_at > 2.0:
            logger.warning(msg)
            self._last_warn_at = now

    def _run_mjpeg_url(self):
        while self._running:
            try:
                req = urllib.request.Request(self.camera_source, headers={"User-Agent": "AetherCV/1.0"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    buf = b""
                    while self._running:
                        chunk = resp.read(4096)
                        if not chunk:
                            raise RuntimeError("MJPEG stream ended")
                        buf += chunk
                        s = buf.find(b"\xff\xd8")
                        e = buf.find(b"\xff\xd9", s + 2)
                        if s == -1 or e == -1:
                            if len(buf) > 1_000_000:
                                buf = buf[-200_000:]
                            continue
                        jpg = buf[s:e + 2]
                        buf = buf[e + 2:]
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is None:
                            continue
                        with self._lock:
                            self._latest_frame = frame
                            self._latest_ts = time.monotonic()
            except Exception as exc:
                self._warn_throttled(f"MJPEG reconnecting: {exc}")
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


# crawing helpers

def draw_hand_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    for s, e in HAND_CONNECTIONS:
        p1 = (int(hand_landmarks[s].x * w), int(hand_landmarks[s].y * h))
        p2 = (int(hand_landmarks[e].x * w), int(hand_landmarks[e].y * h))
        cv2.line(frame, p1, p2, (0, 255, 0), 2)
    for lm in hand_landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 0, 255), -1)


def draw_mode_badge(image, mode: ControlMode):
    """Colored badge in the top-right and large label at the bottom center."""
    label = _MODE_LABEL[mode]
    color = _MODE_COLOR[mode]
    h, w = image.shape[:2]

    # Small badge top-right
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    pad = 10
    x2, y1 = w - 10, 10
    x1, y2 = x2 - tw - pad * 2, y1 + th + pad * 2
    cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    cv2.putText(image, label, (x1 + pad, y2 - pad), font, scale, (255, 255, 255), thick)

    # Large mode name at bottom center
    big_label = mode.value
    big_scale, big_thick = 1.4, 3
    (bw, bh), _ = cv2.getTextSize(big_label, font, big_scale, big_thick)
    bx = (w - bw) // 2
    by = h - 45
    cv2.rectangle(image, (bx - 8, by - bh - 8), (bx + bw + 8, by + 8), (0, 0, 0), -1)
    cv2.putText(image, big_label, (bx, by), font, big_scale, color, big_thick)


def draw_top_summary(image, lines):
    overlay = image.copy()
    banner_h = 28 + len(lines) * 24
    cv2.rectangle(overlay, (0, 0), (image.shape[1], banner_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
    for i, line in enumerate(lines):
        cv2.putText(image, line, (20, 28 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)


def draw_key_hint(image, text):
    h = image.shape[0]
    cv2.putText(image, text, (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


# gesture detection 

GESTURE_HOLD_FRAMES = 20  # frames to hold gesture before switching mode

def count_extended_fingers(hand_landmarks):
    """Count index-through-pinky fingers extended (fingertip above its MCP knuckle)."""
    tips = [8, 12, 16, 20]
    mcps = [5,  9, 13, 17]
    return sum(1 for tip, mcp in zip(tips, mcps)
               if hand_landmarks[tip].y < hand_landmarks[mcp].y)


def gesture_to_mode(finger_count):
    if finger_count == 1:
        return ControlMode.XYZ
    if finger_count == 2:
        return ControlMode.WRIST
    return None


def draw_gesture_progress(image, pending_mode, frames, total):
    if pending_mode is None or frames == 0:
        return
    h, w = image.shape[:2]
    bar_w = 160
    filled = int(bar_w * frames / total)
    color = _MODE_COLOR[pending_mode]
    x, y = w // 2 - bar_w // 2, h - 80
    cv2.rectangle(image, (x, y), (x + bar_w, y + 10), (60, 60, 60), -1)
    cv2.rectangle(image, (x, y), (x + filled, y + 10), color, -1)
    label = f"Hold to switch: {pending_mode.value}"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(image, label, (w // 2 - tw // 2, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


# degree snapping

SNAP_BAND = 18   # degrees of wrist movement per output step — increase to reduce flickering
SNAP_MAX  = 50   # maximum output value

def snap_degrees(value):
    """Snap to 0, 10, 20, … SNAP_MAX with each output step spanning SNAP_BAND degrees."""
    abs_val = abs(value)
    sign = 1 if value >= 0 else -1
    step_count = round(abs_val / SNAP_BAND)
    return int(sign * min(SNAP_MAX, step_count * 10))


# coordinate extraction

def extract_xy(image, hand_landmarks):
    h, w, _ = image.shape
    box = min(h, w)
    x1 = (w - box) // 2
    y1 = (h - box) // 2

    indices = [i for i in [0, 1, 2, 5, 9, 13, 17]
               if 0.0 <= hand_landmarks[i].x <= 1.0 and 0.0 <= hand_landmarks[i].y <= 1.0]
    if not indices:
        return None

    px = sum(hand_landmarks[i].x for i in indices) / len(indices)
    py = sum(hand_landmarks[i].y for i in indices) / len(indices)
    hx, hy = int(px * w), int(py * h)
    cx = max(x1, min(hx, x1 + box))
    cy = max(y1, min(hy, y1 + box))
    return {
        "pixel_x": hx,
        "pixel_y": hy,
        "x": round(((cx - x1) / box) * 200 - 100, 1),
        "y": round(100 - ((cy - y1) / box) * 200, 1),
    }


def extract_z(image, hand_landmarks, z_reset: bool, base_value):
    h, w, _ = image.shape
    palm_idx = [0, 1, 5, 9, 13, 17]
    xs = [hand_landmarks[i].x * w for i in palm_idx]
    ys = [hand_landmarks[i].y * h for i in palm_idx]
    pad = 20
    min_x = max(0, int(min(xs)) - pad)
    max_x = min(w, int(max(xs)) + pad)
    min_y = max(0, int(min(ys)) - pad)
    max_y = min(h, int(max(ys)) + pad)
    area = (max_x - min_x) * (max_y - min_y)

    if z_reset:
        base_value = area
    if base_value is None:
        return None, base_value, (min_x, min_y, max_x, max_y)

    z = max(0, int(area ** 0.5 - base_value ** 0.5))
    return z, base_value, (min_x, min_y, max_x, max_y)


def draw_z_overlay(image, z, z_box):
    if z_box is None:
        return
    min_x, min_y, max_x, max_y = z_box
    if z is None:
        cv2.putText(image, "Press 'r' to set Z=0", (min_x, max(20, min_y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(image, f"Z: {z}", (min_x, max(20, min_y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

# main

def main():
    logger.info("Starting Aether single-hand pipeline.")

    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "hand_landmarker.task")
    detector = None
    reader = None

    try:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO,
        )
        detector = vision.HandLandmarker.create_from_options(options)
        logger.info("HandLandmarker initialized")

        camera_source = resolve_camera_source()
        headless = is_headless_environment()
        if headless:
            logger.info("Headless mode — no OpenCV windows.")

        cap = None
        if isinstance(camera_source, int):
            cap = open_camera_capture(camera_source)
            if not cap.isOpened():
                visible = ", ".join(sorted(glob.glob("/dev/video*"))) or "none"
                raise RuntimeError(
                    f"Failed to open camera {camera_source}. "
                    f"Visible devices: {visible}. "
                    "Set CAMERA_SOURCE to an index or stream URL."
                )

        reader = LatestFrameReader(camera_source, cap=cap)
        reader.start()
        logger.info("Frame reader started.")

        mode = ControlMode.XYZ
        last_mode = None

        gesture_pending = None
        gesture_frames = 0

        z_base = None
        wrist_base_roll = None
        wrist_base_pitch = None

        timestamp_ms = 0
        black_count = 0
        stale_warned = False
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        last_output = None

        while True:
            image, latest_ts = reader.get_latest()

            if image is None:
                if not headless:
                    s = placeholder.copy()
                    cv2.putText(s, "Waiting for camera frames...", (30, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.imshow("Aether Single-Hand", s)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                else:
                    time.sleep(0.01)
                continue

            frame_age = time.monotonic() - latest_ts
            if frame_age > 1.0 and not stale_warned:
                logger.warning("Frame stream stale (>1 s).")
                stale_warned = True
            if frame_age <= 1.0:
                stale_warned = False

            if image.size > 0:
                black_count = (black_count + 1) if float(image.mean()) < 2.0 else 0
            if black_count == 45:
                logger.warning("Receiving near-black frames — check CAMERA_SOURCE.")

            if isinstance(camera_source, int):
                image = cv2.flip(image, 1)

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            now_ms = int(time.monotonic() * 1000)
            timestamp_ms = max(timestamp_ms + 1, now_ms)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            image = border_box(image)
            hand = result.hand_landmarks[0] if result.hand_landmarks else None
            output = None

            if hand is not None:
                draw_hand_landmarks(image, hand)

                # Gesture-based mode switching
                target = gesture_to_mode(count_extended_fingers(hand))
                if target == gesture_pending:
                    if target is not None and target != mode:
                        gesture_frames += 1
                        if gesture_frames >= GESTURE_HOLD_FRAMES:
                            mode = target
                            gesture_pending = None
                            gesture_frames = 0
                            print(f"[GESTURE] switched to {mode.value}", flush=True)
                    else:
                        # Already in this mode or no gesture — clear progress bar
                        gesture_pending = None
                        gesture_frames = 0
                else:
                    gesture_pending = target
                    gesture_frames = 0

                draw_gesture_progress(image, gesture_pending, gesture_frames, GESTURE_HOLD_FRAMES)

                if mode == ControlMode.XYZ:
                    xy = extract_xy(image, hand)
                    z, z_base, z_box = extract_z(image, hand, False, z_base)
                    draw_z_overlay(image, z, z_box)

                    if xy is not None:
                        xy["z"] = z
                        rot = get_base_rotation_direction(xy)
                        base_rotation_x(xy, image)
                        cv2.circle(image, (xy["pixel_x"], xy["pixel_y"]), 8, (255, 60, 0), -1)
                        cv2.putText(image,
                                    f"XYZ: {xy['x']}, {xy['y']}, {z if z is not None else '-'}",
                                    (xy["pixel_x"] + 12, xy["pixel_y"]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 60, 0), 1)
                        output = {"mode": "XYZ",
                                  "xyz": {"x": xy["x"], "y": xy["y"], "z": z},
                                  "base_rotation": rot}
                    else:
                        rot = "No hand"

                    draw_top_summary(image, [
                        "Mode: XYZ",
                        f"XYZ: {xy['x']}, {xy['y']}, {z if z is not None else 'not set'}" if xy else "XYZ: no data",
                        f"Base rotation: {rot}",
                    ])
                    hint = "r = reset Z=0 | q/w change mode"
                    draw_key_hint(image, hint)

                elif mode == ControlMode.WRIST:
                    wrist = None
                    if wrist_base_roll is not None and wrist_base_pitch is not None:
                        wrist = compute_wrist_state(hand, wrist_base_roll, wrist_base_pitch)

                    grab = "Grabbing" if is_grabbing(hand) else "Open"

                    if wrist is not None:
                        wrist_lr = wrist["roll_direction"]
                        wrist_ud = wrist["pitch_direction"]
                        roll_deg = 0 if wrist_lr == "Center" else snap_degrees(wrist["roll_delta"])
                        pitch_deg = 0 if wrist_ud == "Neutral" else snap_degrees(wrist["pitch_delta"])
                        output = {"mode": "WRIST",
                                  "wrist": {
                                      "up_down": wrist_ud,
                                      "left_right_rotation": wrist_lr,
                                      "roll_deg": roll_deg,
                                      "pitch_deg": pitch_deg,
                                  },
                                  "grab": grab}
                        print(json.dumps(output), flush=True)
                    else:
                        wrist_lr = "Not calibrated"
                        wrist_ud = "Not calibrated"
                        roll_deg = 0
                        pitch_deg = 0

                    roll_deg_str = f"{roll_deg:+d}°" if wrist else "-"
                    pitch_deg_str = f"{pitch_deg:+d}°" if wrist else "-"
                    draw_top_summary(image, [
                        "Mode: WRIST+GRAB",
                        f"Left/Right: {wrist_lr}  ({roll_deg_str})",
                        f"Up/Down:    {wrist_ud}  ({pitch_deg_str})",
                        f"Grab: {grab}",
                    ])
                    hint = "b = reset wrist base | q/w change mode" if wrist_base_roll is not None else "b to calibrate wrist | q/w change mode"
                    draw_key_hint(image, hint)

            else:
                gesture_pending = None
                gesture_frames = 0
                label = "WRIST+GRAB" if mode == ControlMode.WRIST else mode.value
                draw_top_summary(image, [f"Mode: {label}", "No hand detected"])

            draw_mode_badge(image, mode)

            if mode != last_mode:
                print(f"\n=== MODE: {mode.value} ===", flush=True)
                last_mode = mode
                last_output = None  # force first output of new mode to always print

            if output is not None and output != last_output:
                print(json.dumps(output), flush=True)
                last_output = output

            if not headless:
                cv2.imshow("Aether Single-Hand", image)
                key = cv2.waitKey(1) & 0xFF

                if key != 255:
                    print(f"[KEY] {key} ({chr(key) if 32 <= key < 127 else '?'})", flush=True)

                if key == 27:
                    logger.info("ESC pressed — exiting.")
                    break
                elif key == ord("q"):
                    mode = ControlMode.XYZ
                    print("[MODE] XYZ", flush=True)
                elif key == ord("w"):
                    mode = ControlMode.WRIST
                    print("[MODE] WRIST", flush=True)
                elif key == ord("r") and mode == ControlMode.XYZ and hand is not None:
                    _, z_base, _ = extract_z(image, hand, True, z_base)
                    logger.info("Z=0 baseline set.")
                elif key == ord("b") and mode == ControlMode.WRIST and hand is not None:
                    wrist_base_roll, wrist_base_pitch = calibrate_wrist_base(hand)
                    print("[WRIST] Calibrated. Move your wrist to see output.", flush=True)

        logger.info("Vision loop exited normally.")
        return 0

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — shutting down.")
        return 0
    except Exception:
        logger.exception("Single-hand pipeline terminated unexpectedly.")
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
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    raise SystemExit(main())
