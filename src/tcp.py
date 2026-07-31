import json
import os
import socket
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from GrabbingMotion2 import is_grabbing
from MultiHandTracker import HandSide, MultiHandTracker
from WristDetection import calibrate_wrist_base, compute_wrist_state
from aether_logger import setup_logger
from base_rotation import (
    border_box,
    draw_base_rotation_indicators,
    get_active_base_rotation_zones,
    get_base_rotation_direction,
    get_combined_base_rotation_output,
    resolve_base_rotation_zone_owners,
)
from main2 import (
    LatestFrameReader,
    adjust_terminal_stream_interval,
    draw_hand_landmarks,
    draw_hand_name,
    draw_xy_coordinates_for_hand,
    draw_z_overlay,
    extract_xy_coordinates_for_hand,
    extract_z_coordinate_for_hand,
    format_display_number,
    format_xyz_payload,
    format_terminal_stream_rate,
    is_headless_environment,
    is_stream_rate_decrease_key,
    is_stream_rate_increase_key,
    open_camera_capture,
    resolve_terminal_stream_interval,
    resolve_camera_source,
)
from smoothing import GrabDebouncer, WristSmoother, XYZSmoother

#CAMERA_SOURCE=http://host.docker.internal:8080/video.mjpg \                 
#TCP_BIND_HOST=0.0.0.0 \
#TCP_PORT=8765 \
#python3 src/tcp.py

logger = setup_logger("aether-system.log", "AETHER.VISION.TCP")


def resolve_tcp_stream_interval(default_hz="30"):
    # Match main2's interval handling, but allow a TCP-specific default rate.
    raw_interval_ms = os.getenv("TCP_STREAM_INTERVAL_MS", "").strip()
    if raw_interval_ms:
        try:
            interval_ms = max(0.0, float(raw_interval_ms))
            return interval_ms / 1000.0
        except ValueError:
            logger.warning(
                "Invalid TCP_STREAM_INTERVAL_MS=%r. Falling back to TCP_STREAM_HZ/default.",
                raw_interval_ms,
            )

    raw_hz = os.getenv("TCP_STREAM_HZ", default_hz).strip()
    try:
        hz = float(raw_hz)
    except ValueError:
        logger.warning("Invalid TCP_STREAM_HZ=%r. Falling back to 30 Hz.", raw_hz)
        hz = 30.0

    if hz <= 0:
        return 0.0

    return 1.0 / hz


def resolve_tcp_bind_host():
    return os.getenv("TCP_BIND_HOST", "0.0.0.0").strip() or "0.0.0.0"


def resolve_tcp_port():
    raw_port = os.getenv("TCP_PORT", "8765").strip()
    try:
        return int(raw_port)
    except ValueError:
        logger.warning("Invalid TCP_PORT=%r. Falling back to 8765.", raw_port)
        return 8765


class JsonTcpServer:
    """Tiny newline-delimited JSON TCP server with a single active client."""

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.server_socket.setblocking(False)
        self.client_socket = None
        self.client_address = None

    def poll_for_client(self):
        if self.client_socket is not None:
            return

        try:
            client_socket, client_address = self.server_socket.accept()
        except BlockingIOError:
            return

        # Keep the accepted client socket in blocking mode so newline-delimited
        # JSON writes behave predictably through forwarded ports and proxies.
        client_socket.setblocking(True)
        client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.client_socket = client_socket
        self.client_address = client_address
        logger.info("TCP client connected from %s:%s", client_address[0], client_address[1])

    def send_json(self, payload):
        if self.client_socket is None:
            return False

        message = (json.dumps(payload) + "\n").encode("utf-8")
        try:
            self.client_socket.sendall(message)
            return True
        except BlockingIOError:
            logger.warning("TCP send would block; dropping this frame for %s:%s", self.client_address[0], self.client_address[1])
            return False
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.close_client()
            return False

    def client_status(self):
        if self.client_socket is None:
            return "waiting for client"
        return f"connected {self.client_address[0]}:{self.client_address[1]}"

    def close_client(self):
        if self.client_socket is not None:
            try:
                self.client_socket.close()
            except OSError:
                pass
        self.client_socket = None
        self.client_address = None

    def close(self):
        self.close_client()
        try:
            self.server_socket.close()
        except OSError:
            pass


def main():
    logger.info("Starting Aether TCP hand-state pipeline.")

    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "hand_landmarker.task")
    detector = None
    reader = None
    tcp_server = None

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

        camera_source = resolve_camera_source()
        cap = None
        headless = is_headless_environment()

        if isinstance(camera_source, int):
            cap = open_camera_capture(camera_source)

        logger.info("Using CAMERA_SOURCE=%s", camera_source)
        reader = LatestFrameReader(camera_source, cap=cap)
        reader.start()

        tcp_host = resolve_tcp_bind_host()
        tcp_port = resolve_tcp_port()
        stream_interval = resolve_terminal_stream_interval()
        last_stream_at = 0.0

        tcp_server = JsonTcpServer(tcp_host, tcp_port)
        logger.info("TCP server listening on %s:%s", tcp_host, tcp_port)

        timestamp_ms = 0
        black_frame_count = 0
        waiting_warned = False
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)

        right_hand_z_base = None
        left_hand_base_roll = None
        left_hand_base_pitch = None
        base_rotation_zone_owners = {"Left": None, "Right": None}

        xyz_smoother = XYZSmoother()
        wrist_smoother = WristSmoother()
        grab_debouncer = GrabDebouncer()

        while True:
            tcp_server.poll_for_client()
            image, latest_ts = reader.get_latest()

            if image is None:
                if not headless:
                    status = placeholder.copy()
                    cv2.putText(status, "Waiting for camera frames...", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.imshow("Aether TCP", status)
                    key = cv2.waitKeyEx(1)
                    if key == 27:
                        break
                    if is_stream_rate_increase_key(key):
                        stream_interval = adjust_terminal_stream_interval(stream_interval, 1)
                    if is_stream_rate_decrease_key(key):
                        stream_interval = adjust_terminal_stream_interval(stream_interval, -1)
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
            raw_right_xyz = None

            left_grab = "No hand"
            left_wrist = None
            left_base_rotation = "No hand"
            raw_left_grab = "No hand"
            raw_left_wrist = None

            if right_state is not None:
                right_hand = right_state.landmarks
                draw_hand_landmarks(image, right_hand)
                draw_hand_name(image, "Right Hand", right_hand, (255, 50, 50))

                right_xyz = extract_xy_coordinates_for_hand(image, right_hand)
                right_z, right_hand_z_base, right_z_box = extract_z_coordinate_for_hand(
                    image,
                    right_hand,
                    False,
                    right_hand_z_base,
                )

                if right_xyz is not None:
                    right_xyz["z"] = right_z
                    raw_right_xyz = {k: right_xyz[k] for k in ("x", "y", "z")}
                    right_xyz = xyz_smoother.update(right_xyz)
                    right_base_rotation = get_base_rotation_direction(right_xyz)
                    image = draw_xy_coordinates_for_hand(image, right_xyz, color=(255, 50, 50))

                displayed_right_z = right_xyz["z"] if right_xyz is not None else right_z
                image = draw_z_overlay(
                    image,
                    displayed_right_z,
                    right_z_box,
                    right_hand_z_base is not None,
                )
            else:
                xyz_smoother.reset()

            if left_state is not None:
                left_hand = left_state.landmarks
                draw_hand_landmarks(image, left_hand)
                draw_hand_name(image, "Left Hand", left_hand, (120, 220, 255))

                left_xy = extract_xy_coordinates_for_hand(image, left_hand)
                if left_xy is not None:
                    left_base_rotation = get_base_rotation_direction(left_xy)

                    raw_left_grab = "Grabbing" if is_grabbing(left_hand) else "Open"
                    left_grab = grab_debouncer.update(raw_left_grab == "Grabbing")
                    middle_base = left_hand[9]
                    text_x = int(middle_base.x * image.shape[1]) - 175
                    text_y = int(middle_base.y * image.shape[0]) - 25

                if left_hand_base_roll is None or left_hand_base_pitch is None:
                    cv2.putText(
                        image,
                        "Press 'B' to set left wrist base",
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 255, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        image,
                        "B = reset left wrist base",
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 255, 255),
                        2,
                    )

                if left_hand_base_roll is not None and left_hand_base_pitch is not None:
                    raw_left_wrist = compute_wrist_state(
                        left_hand,
                        "Left",
                        left_hand_base_roll,
                        left_hand_base_pitch,
                    )
                    left_wrist = wrist_smoother.update(raw_left_wrist)
            else:
                wrist_smoother.reset()
                grab_debouncer.reset()

            left_wrist_lr = left_wrist["roll_direction"] if left_wrist else "Not calibrated"
            left_wrist_ud = left_wrist["pitch_direction"] if left_wrist else "Not calibrated"
            right_xyz_text = (
                f"{format_display_number(right_xyz['x'])}, "
                f"{format_display_number(right_xyz['y'])}, "
                f"{format_display_number(right_xyz['z'])}"
                if right_xyz is not None
                else "No hand"
            )
            base_rotation_zone_owners = resolve_base_rotation_zone_owners(
                base_rotation_zone_owners,
                {
                    "right_hand": right_base_rotation,
                    "left_hand": left_base_rotation,
                },
            )
            active_base_rotation_zones = get_active_base_rotation_zones(base_rotation_zone_owners)
            combined_base_rotation = get_combined_base_rotation_output(
                base_rotation_zone_owners,
                {
                    "right_hand": right_base_rotation,
                    "left_hand": left_base_rotation,
                },
            )
            draw_base_rotation_indicators(active_base_rotation_zones, image)

            summary_lines = [
                f"Grab: {left_grab}",
                f"LR: {left_wrist_lr}",
                f"UD: {left_wrist_ud}",
            ]

            if left_state is not None:
                for i, line in enumerate(summary_lines):
                    cv2.putText(
                        image,
                        line,
                        (20, 40 + i * 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )

            terminal_state = {
                "left_hand": {
                    "grab": left_grab,
                    "wrist": {
                        "up_down": left_wrist["pitch_direction"] if left_wrist else "Not calibrated",
                        "left_right_rotation": left_wrist["roll_direction"] if left_wrist else "Not calibrated",
                    },
                },
                "right_hand": {
                    "xyz": format_xyz_payload(right_xyz),
                },
                "raw": {
                    "right_hand": {"xyz": raw_right_xyz},
                    "left_hand": {
                        "grab": raw_left_grab,
                        "wrist": {
                            "up_down": raw_left_wrist["pitch_direction"] if raw_left_wrist else "Not calibrated",
                            "left_right_rotation": raw_left_wrist["roll_direction"] if raw_left_wrist else "Not calibrated",
                            "roll_delta": round(raw_left_wrist["roll_delta"], 2) if raw_left_wrist else None,
                            "pitch_delta": round(raw_left_wrist["pitch_delta"], 2) if raw_left_wrist else None,
                        },
                    },
                } if os.getenv("LOG_RAW", "true").lower() != "false" else None,
                "base_rotation": combined_base_rotation,
            }

            stream_now = time.monotonic()
            if (
                stream_interval == 0.0 or
                stream_now - last_stream_at >= stream_interval
            ):
                print(json.dumps(terminal_state), flush=True)
                tcp_server.send_json(terminal_state)
                last_stream_at = stream_now

            if not headless:
                cv2.putText(
                    image,
                    (
                        f"{format_terminal_stream_rate(stream_interval)} | "
                        f"TCP: {tcp_server.client_status()} | [: slower  ]: faster"
                    ),
                    (20, image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Aether TCP", image)
                key = cv2.waitKeyEx(1)

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
                if is_stream_rate_increase_key(key):
                    stream_interval = adjust_terminal_stream_interval(stream_interval, 1)
                if is_stream_rate_decrease_key(key):
                    stream_interval = adjust_terminal_stream_interval(stream_interval, -1)

        return 0
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down TCP pipeline.")
        return 0
    except Exception:
        logger.exception("TCP pipeline terminated due to an unexpected error.")
        return 1
    finally:
        if tcp_server is not None:
            tcp_server.close()
        if reader is not None:
            reader.stop()
        if detector is not None:
            detector.close()
        if not is_headless_environment():
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
