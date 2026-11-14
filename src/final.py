"""Two-camera final control pipeline.

Front camera: XY, wrist left/right, and base rotation.
Side camera: Z, grabbing, and wrist up/down.
"""

"""
cd /workspace
XYZ_CAMERA_SOURCE=http://host.docker.internal:8080/video.mjpg \
GESTURE_CAMERA_SOURCE=http://host.docker.internal:8081/video.mjpg \
TCP_BIND_HOST=0.0.0.0 \
TCP_PORT=8765 \
python3 src/final.py
"""

import glob
import json
import os
import time

os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from GrabbingMotion_sideview import sideview_is_grabbing
from WristDetection import (
    compute_roll_degrees,
)
from aether_logger import setup_logger
from base_rotation import (
    border_box,
    draw_base_rotation_indicators,
    get_active_base_rotation_zones,
    get_combined_base_rotation_output,
    resolve_base_rotation_zone_owners,
)
from main2 import (
    LatestFrameReader,
    adjust_terminal_stream_interval,
    draw_hand_landmarks,
    draw_hand_name,
    draw_top_summary,
    draw_xy_coordinates_for_hand,
    extract_xy_coordinates_for_hand,
    format_display_number,
    format_terminal_stream_rate,
    format_xyz_payload,
    is_headless_environment,
    is_stream_rate_decrease_key,
    is_stream_rate_increase_key,
    open_camera_capture,
    resolve_terminal_stream_interval,
)
from smoothing import EMASmoother, GrabDebouncer, XYZSmoother
from tcp import JsonTcpServer, resolve_tcp_bind_host, resolve_tcp_port


logger = setup_logger("aether-system.log", "AETHER.VISION.FINAL")


def draw_footer_panel(image, line_one, line_two):
    """Draw keyboard controls over a translucent footer."""
    overlay = image.copy()
    panel_height = 78
    top_y = max(0, image.shape[0] - panel_height)
    cv2.rectangle(overlay, (0, top_y), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)
    cv2.putText(image, line_one, (20, top_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(image, line_two, (20, top_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def resolve_camera_source_from_env(primary_name, legacy_name, default_value):
    """Read a device index or MJPEG URL, accepting old two_camera variable names."""
    raw_source = os.getenv(primary_name, os.getenv(legacy_name, default_value)).strip()
    try:
        return int(raw_source)
    except ValueError:
        return raw_source


def resolve_front_camera_source():
    return resolve_camera_source_from_env("FRONT_CAMERA_SOURCE", "XYZ_CAMERA_SOURCE", "0")


def resolve_side_camera_source():
    return resolve_camera_source_from_env("SIDE_CAMERA_SOURCE", "GESTURE_CAMERA_SOURCE", "1")


def create_detector(model_path):
    """Create one MediaPipe detector per camera; each feed tracks one intended hand."""
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HandLandmarker.create_from_options(options)


def open_reader(camera_source, source_name):
    """Open a local camera or begin reading an MJPEG stream in the background."""
    cap = None
    if isinstance(camera_source, int):
        cap = open_camera_capture(camera_source)
        if not cap.isOpened():
            device_path = f"/dev/video{camera_source}"
            visible_devices = ", ".join(sorted(glob.glob("/dev/video*"))) or "none"
            if not os.path.exists(device_path):
                raise RuntimeError(
                    f"Failed to open {source_name}: {camera_source}. {device_path} is not available. "
                    f"Visible camera devices: {visible_devices}. Use an MJPEG URL in a devcontainer."
                )
            raise RuntimeError(f"Failed to open {source_name}: {camera_source}.")

    reader = LatestFrameReader(camera_source, cap=cap)
    reader.start()
    logger.info("Using %s=%s", source_name, camera_source)
    return reader


def detect_primary_hand(image, detector, timestamp_ms, mirror_local):
    """Detect the first hand in one camera frame and return its handedness label."""
    if image is None:
        return None, timestamp_ms, None, "Unknown"
    if mirror_local:
        image = cv2.flip(image, 1)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    timestamp_ms = max(timestamp_ms + 1, int(time.monotonic() * 1000))
    result = detector.detect_for_video(mp_image, timestamp_ms)

    if not result.hand_landmarks:
        return image, timestamp_ms, None, "Unknown"

    handedness = "Unknown"
    if getattr(result, "handedness", None) and result.handedness[0]:
        handedness = result.handedness[0][0].category_name or "Unknown"
    return image, timestamp_ms, result.hand_landmarks[0], handedness


def compose_dual_view(front_frame, side_frame):
    """Render the front and side previews next to each other at equal height."""
    if front_frame is None and side_frame is None:
        return np.zeros((480, 1280, 3), dtype=np.uint8)
    if front_frame is None:
        front_frame = np.zeros_like(side_frame)
    if side_frame is None:
        side_frame = np.zeros_like(front_frame)

    target_height = min(front_frame.shape[0], side_frame.shape[0])

    def resize_to_height(frame):
        if frame.shape[0] == target_height:
            return frame
        scale = target_height / frame.shape[0]
        return cv2.resize(frame, (max(1, int(frame.shape[1] * scale)), target_height))

    return np.hstack([resize_to_height(front_frame), resize_to_height(side_frame)])


def classify_front_roll(roll_delta, previous_direction, threshold=20.0, release_threshold=12.0):
    """Classify front left/right tilt with a larger activation threshold and hysteresis."""
    if previous_direction == "Right" and roll_delta > release_threshold:
        return "Right"
    if previous_direction == "Left" and roll_delta < -release_threshold:
        return "Left"
    if roll_delta >= threshold:
        return "Right"
    if roll_delta <= -threshold:
        return "Left"
    return "Center"


def make_roll_state(hand_landmarks, base_roll, smoother, previous_direction):
    """Return only front-camera left/right wrist motion relative to its neutral pose."""
    if base_roll is None:
        return None
    roll_delta = smoother.update(compute_roll_degrees(hand_landmarks) - base_roll)
    direction = classify_front_roll(roll_delta, previous_direction)
    return {"direction": direction, "delta": roll_delta}


def compute_side_pitch_degrees(hand_landmarks):
    """Measure side-view wrist pitch from a stable wrist-to-palm axis in image space."""
    wrist = hand_landmarks[0]
    # The MCP row remains stable while fingers curl, unlike fingertip-based angles.
    palm_mcp_indices = (5, 9, 13, 17)
    palm_x = sum(hand_landmarks[index].x for index in palm_mcp_indices) / len(palm_mcp_indices)
    palm_y = sum(hand_landmarks[index].y for index in palm_mcp_indices) / len(palm_mcp_indices)
    return float(np.degrees(np.arctan2(palm_x - wrist.x, -(palm_y - wrist.y))))


def classify_side_pitch(pitch_delta, previous_direction, threshold=20.0, release_threshold=12.0):
    """Classify side pitch with hysteresis so small movements cannot flicker the result."""
    # The side camera is oriented opposite the original pitch convention.
    if previous_direction == "Down" and pitch_delta > release_threshold:
        return "Down"
    if previous_direction == "Up" and pitch_delta < -release_threshold:
        return "Up"
    if pitch_delta >= threshold:
        return "Down"
    if pitch_delta <= -threshold:
        return "Up"
    return "Neutral"


def make_pitch_state(hand_landmarks, base_pitch, smoother, previous_direction):
    """Return side-camera up/down wrist state using only stable 2D palm landmarks."""
    if base_pitch is None:
        return None
    pitch_delta = smoother.update(compute_side_pitch_degrees(hand_landmarks) - base_pitch)
    direction = classify_side_pitch(pitch_delta, previous_direction)
    return {"direction": direction, "delta": pitch_delta}


def extract_side_z_coordinate(image, hand_landmarks, reset, base_x):
    """Use side-camera horizontal hand motion as Z, relative to a saved start point."""
    side_xy = extract_xy_coordinates_for_hand(image, hand_landmarks)
    if side_xy is None:
        return None, base_x, None

    current_x = side_xy["x"]
    if reset:
        base_x = current_x
    if base_x is None:
        return None, base_x, side_xy

    # A side view turns physical forward/backward movement into horizontal motion.
    return round(current_x - base_x, 1), base_x, side_xy


def draw_side_z_overlay(image, z_coordinate, side_xy, is_calibrated):
    """Display the horizontal side-view Z measurement at the tracked palm center."""
    if side_xy is None:
        return image

    text_x = side_xy["pixel_x"] + 12
    text_y = max(25, side_xy["pixel_y"] - 12)
    if not is_calibrated:
        text = "Press 'R' to set side Z=0"
        color = (255, 255, 255)
    else:
        text = f"Z: {format_display_number(z_coordinate)}"
        color = (0, 255, 255)

    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return image


def get_front_base_rotation_direction(image, front_xy):
    """Activate rotation only when the palm center is inside a front-camera red zone."""
    if image is None or front_xy is None:
        return "No hand"

    height, width = image.shape[:2]
    square_size = min(height, width)
    left_boundary = (width - square_size) // 2
    right_boundary = left_boundary + square_size
    palm_x = front_xy["pixel_x"]

    if palm_x < left_boundary:
        return "Left"
    if palm_x > right_boundary:
        return "Right"
    return "Center"


def main():
    logger.info("Starting Aether final two-camera pipeline.")
    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "hand_landmarker.task")
    front_detector = side_detector = front_reader = side_reader = tcp_server = None

    try:
        front_detector = create_detector(model_path)
        side_detector = create_detector(model_path)
        front_source = resolve_front_camera_source()
        side_source = resolve_side_camera_source()
        headless = is_headless_environment()
        front_reader = open_reader(front_source, "FRONT_CAMERA_SOURCE")
        side_reader = open_reader(side_source, "SIDE_CAMERA_SOURCE")

        front_timestamp_ms = side_timestamp_ms = 0
        stream_interval = resolve_terminal_stream_interval()
        last_stream_at = 0.0
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)

        # Each motion has a separate neutral calibration because it comes from a different view.
        front_wrist_base_roll = None
        front_roll_direction = "Center"
        side_wrist_base_pitch = None
        side_pitch_direction = "Neutral"
        side_z_base = None
        front_zone_owners = {"Left": None, "Right": None}
        side_grab_was_active = False

        tcp_server = JsonTcpServer(resolve_tcp_bind_host(), resolve_tcp_port())
        logger.info("TCP server listening on %s:%s", tcp_server.host, tcp_server.port)

        xyz_smoother = XYZSmoother()
        side_z_smoother = EMASmoother(alpha=0.4)
        roll_smoother = EMASmoother(alpha=0.4)
        pitch_smoother = EMASmoother(alpha=0.4)
        grab_debouncer = GrabDebouncer()

        while True:
            tcp_server.poll_for_client()
            front_image, _ = front_reader.get_latest()
            side_image, _ = side_reader.get_latest()

            if front_image is None and side_image is None:
                if headless:
                    time.sleep(0.01)
                    continue
                status = placeholder.copy()
                cv2.putText(status, "Waiting for camera feeds...", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.imshow("Aether Final Two Camera", status)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            front_image, front_timestamp_ms, front_hand, front_handedness = detect_primary_hand(
                front_image, front_detector, front_timestamp_ms, isinstance(front_source, int)
            )
            side_image, side_timestamp_ms, side_hand, _ = detect_primary_hand(
                side_image, side_detector, side_timestamp_ms, isinstance(side_source, int)
            )

            # The base-rotation zones belong only to the front view.
            if front_image is not None:
                front_image = border_box(front_image)

            front_xyz = None
            raw_front_xyz = None
            front_rotation_direction = "No hand"
            front_roll = None
            side_z = None
            side_z_xy = None
            side_pitch = None
            raw_side_grab = "No hand"
            side_grab = "No hand"

            if front_hand is not None and front_image is not None:
                draw_hand_landmarks(front_image, front_hand)
                draw_hand_name(front_image, "Front: XY + Wrist LR", front_hand, (255, 120, 120))
                front_xyz = extract_xy_coordinates_for_hand(front_image, front_hand)
                if front_xyz is not None:
                    raw_front_xyz = {"x": front_xyz["x"], "y": front_xyz["y"], "z": None}
                    front_xyz = xyz_smoother.update({**front_xyz, "z": 0})
                    front_rotation_direction = get_front_base_rotation_direction(front_image, front_xyz)
                    front_image = draw_xy_coordinates_for_hand(front_image, front_xyz, color=(255, 0, 0))
                front_roll = make_roll_state(
                    front_hand,
                    front_wrist_base_roll,
                    roll_smoother,
                    front_roll_direction,
                )
                if front_roll is not None:
                    front_roll_direction = front_roll["direction"]
            else:
                xyz_smoother.reset()
                roll_smoother.reset()
                front_roll_direction = "Center"

            if side_hand is not None and side_image is not None:
                draw_hand_landmarks(side_image, side_hand)
                draw_hand_name(side_image, "Side: Z + Grab + Wrist UD", side_hand, (120, 220, 255))
                side_z, side_z_base, side_z_xy = extract_side_z_coordinate(side_image, side_hand, False, side_z_base)
                if side_z is not None:
                    side_z = side_z_smoother.update(side_z)
                raw_grab_bool = sideview_is_grabbing(side_hand, side_grab_was_active)
                side_grab_was_active = raw_grab_bool
                raw_side_grab = "Grabbing" if raw_grab_bool else "Open"
                side_grab = grab_debouncer.update(raw_grab_bool)
                side_pitch = make_pitch_state(
                    side_hand,
                    side_wrist_base_pitch,
                    pitch_smoother,
                    side_pitch_direction,
                )
                if side_pitch is not None:
                    side_pitch_direction = side_pitch["direction"]
                side_image = draw_side_z_overlay(side_image, side_z, side_z_xy, side_z_base is not None)
            else:
                side_grab_was_active = False
                grab_debouncer.reset()
                pitch_smoother.reset()
                side_pitch_direction = "Neutral"
                side_z_smoother.reset()

            front_rotation_owners = resolve_base_rotation_zone_owners(
                front_zone_owners, {"front_hand": front_rotation_direction}
            )
            front_zone_owners = front_rotation_owners
            active_base_rotation_zones = get_active_base_rotation_zones(front_rotation_owners)
            base_rotation = get_combined_base_rotation_output(
                front_rotation_owners, {"front_hand": front_rotation_direction}
            )
            if front_image is not None:
                draw_base_rotation_indicators(active_base_rotation_zones, front_image)

            front_lr = front_roll["direction"] if front_roll else "Not calibrated"
            side_ud = side_pitch["direction"] if side_pitch else "Not calibrated"
            xyz_text = (
                f"{format_display_number(front_xyz['x'])}, {format_display_number(front_xyz['y'])}, {format_display_number(side_z)}"
                if front_xyz is not None else "No hand"
            )
            combined_state = {
                # Keep the established TCP schema while each field now comes from its assigned camera.
                "left_hand": {
                    "grab": side_grab,
                    "wrist": {"up_down": side_ud, "left_right_rotation": front_lr},
                },
                "right_hand": {
                    "xyz": (
                        {**format_xyz_payload(front_xyz), "z": None if side_z is None else round(float(side_z), 1)}
                        if front_xyz is not None else None
                    ),
                },
                "raw": {
                    "front_camera": {
                        "xyz": raw_front_xyz,
                        "wrist_left_right": None if front_roll is None else round(front_roll["delta"], 2),
                    },
                    "side_camera": {
                        "z": side_z,
                        "grab": raw_side_grab,
                        "wrist_up_down": None if side_pitch is None else round(side_pitch["delta"], 2),
                    },
                } if os.getenv("LOG_RAW", "true").lower() != "false" else None,
                "base_rotation": base_rotation,
            }

            now = time.monotonic()
            if stream_interval == 0.0 or now - last_stream_at >= stream_interval:
                print(json.dumps(combined_state), flush=True)
                tcp_server.send_json(combined_state)
                last_stream_at = now

            if not headless:
                combined_image = compose_dual_view(front_image, side_image)
                draw_top_summary(combined_image, [
                    f"Front Camera | XY: {xyz_text} | Wrist LR: {front_lr}",
                    f"Side Camera | Z: {format_display_number(side_z)} | Grab: {side_grab} | Wrist UD: {side_ud}",
                    f"Base Rotation (Front Only) | {base_rotation}",
                ])
                front_text = "f: reset front wrist LR neutral" if front_wrist_base_roll is not None else "f: set front wrist LR neutral"
                side_text = "s: reset side wrist UD neutral" if side_wrist_base_pitch is not None else "s: set side wrist UD neutral"
                z_text = "r: reset side Z=0" if side_z_base is not None else "r: set side Z=0"
                draw_footer_panel(
                    combined_image,
                    f"{format_terminal_stream_rate(stream_interval)} | TCP: {tcp_server.client_status()} | [: slower  ]: faster",
                    f"{front_text} | {side_text} | {z_text} | Esc: quit",
                )
                cv2.imshow("Aether Final Two Camera", combined_image)
                key = cv2.waitKeyEx(1)

                if key == 27:
                    break
                if key == ord("f") and front_hand is not None:
                    front_wrist_base_roll = compute_roll_degrees(front_hand)
                    roll_smoother.reset()
                    front_roll_direction = "Center"
                if key == ord("s") and side_hand is not None:
                    side_wrist_base_pitch = compute_side_pitch_degrees(side_hand)
                    pitch_smoother.reset()
                    side_pitch_direction = "Neutral"
                if key == ord("r") and side_hand is not None and side_image is not None:
                    _, side_z_base, _ = extract_side_z_coordinate(side_image, side_hand, True, side_z_base)
                    side_z_smoother.reset()
                if is_stream_rate_increase_key(key):
                    stream_interval = adjust_terminal_stream_interval(stream_interval, 1)
                if is_stream_rate_decrease_key(key):
                    stream_interval = adjust_terminal_stream_interval(stream_interval, -1)

        return 0
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down final two-camera pipeline.")
        return 0
    except Exception:
        logger.exception("final pipeline terminated due to an unexpected error.")
        return 1
    finally:
        if front_reader is not None:
            front_reader.stop()
        if side_reader is not None:
            side_reader.stop()
        if tcp_server is not None:
            tcp_server.close()
        if front_detector is not None:
            front_detector.close()
        if side_detector is not None:
            side_detector.close()
        if not is_headless_environment():
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
