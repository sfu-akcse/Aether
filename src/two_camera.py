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

from GrabbingMotion2 import is_grabbing
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
    draw_top_summary,
    draw_xy_coordinates_for_hand,
    draw_z_overlay,
    extract_xy_coordinates_for_hand,
    extract_z_coordinate_for_hand,
    format_display_number,
    format_terminal_stream_rate,
    format_xyz_payload,
    is_headless_environment,
    is_stream_rate_decrease_key,
    is_stream_rate_increase_key,
    open_camera_capture,
    resolve_terminal_stream_interval,
)
from smoothing import GrabDebouncer, WristSmoother, XYZSmoother
from tcp import JsonTcpServer, resolve_tcp_bind_host, resolve_tcp_port, resolve_tcp_stream_interval


#cd /workspace
#XYZ_CAMERA_SOURCE=http://host.docker.internal:8080/video.mjpg \
#GESTURE_CAMERA_SOURCE=http://host.docker.internal:8081/video.mjpg \
#TCP_BIND_HOST=0.0.0.0 \
#TCP_PORT=8765 \
#python3 src/two_camera.py

#cd /Users/admin/Aether
#source .host-venv/bin/activate
#python3 scripts/host_webcam_stream.py --port 8081 --camera-index 1

#cd /Users/admin/Aether
#source .host-venv/bin/activate
#python3 scripts/host_webcam_stream.py --port 8080 --camera-index 0


logger = setup_logger("aether-system.log", "AETHER.VISION.TWO_CAMERA")


def draw_footer_panel(image, line_one, line_two):
    # Draw a simple translucent footer so control hints stay readable over video.
    overlay = image.copy()
    panel_height = 78
    top_y = max(0, image.shape[0] - panel_height)
    cv2.rectangle(overlay, (0, top_y), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)
    cv2.putText(
        image,
        line_one,
        (20, top_y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        image,
        line_two,
        (20, top_y + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )


def resolve_camera_source_from_env(env_name, default_value):
    # Accept either a numeric device index or a stream URL from the environment.
    raw_source = os.getenv(env_name, default_value).strip()
    try:
        return int(raw_source)
    except ValueError:
        return raw_source


def resolve_xyz_camera_source():
    # Camera dedicated to XY/Z tracking.
    return resolve_camera_source_from_env("XYZ_CAMERA_SOURCE", "0")


def resolve_gesture_camera_source():
    # Camera dedicated to grabbing and wrist gestures.
    return resolve_camera_source_from_env("GESTURE_CAMERA_SOURCE", "1")


def create_detector(model_path):
    # Each camera gets its own single-hand detector to keep the pipelines separate.
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HandLandmarker.create_from_options(options)


def open_reader(camera_source, source_name):
    # Start a background latest-frame reader for either a device index or MJPEG URL.
    cap = None
    if isinstance(camera_source, int):
        cap = open_camera_capture(camera_source)

        if not cap.isOpened():
            device_path = f"/dev/video{camera_source}"
            visible_devices = ", ".join(sorted(glob.glob("/dev/video*"))) or "none"

            if not os.path.exists(device_path):
                raise RuntimeError(
                    f"Failed to open {source_name}: {camera_source}. "
                    f"{device_path} is not available in this container. "
                    f"Visible camera devices: {visible_devices}. "
                    "If you are in a devcontainer, either pass the camera through "
                    "or use a stream URL like http://host.docker.internal:8080/video.mjpg."
                )

            raise RuntimeError(
                f"Failed to open {source_name}: {camera_source}. "
                "Set it to a valid camera index or stream URL."
            )

    reader = LatestFrameReader(camera_source, cap=cap)
    reader.start()
    logger.info("Using %s=%s", source_name, camera_source)
    return reader


def detect_primary_hand(image, latest_ts, detector, timestamp_ms, mirror_local):
    # Run MediaPipe on one frame and return only the primary hand for that camera.
    if image is None:
        return None, timestamp_ms, None, "Unknown"

    if mirror_local:
        image = cv2.flip(image, 1)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    now_ms = int(time.monotonic() * 1000)
    timestamp_ms = max(timestamp_ms + 1, now_ms)
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)

    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]
        handedness_label = "Unknown"
        if getattr(detection_result, "handedness", None) and detection_result.handedness[0]:
            handedness_label = detection_result.handedness[0][0].category_name or "Unknown"
        return image, timestamp_ms, hand_landmarks, handedness_label

    return image, timestamp_ms, None, "Unknown"


def compose_dual_view(xyz_frame, gesture_frame):
    # Resize both feeds to the same height and show them side-by-side in one window.
    if xyz_frame is None and gesture_frame is None:
        return np.zeros((480, 1280, 3), dtype=np.uint8)

    if xyz_frame is None:
        xyz_frame = np.zeros_like(gesture_frame)
    if gesture_frame is None:
        gesture_frame = np.zeros_like(xyz_frame)

    target_height = min(xyz_frame.shape[0], gesture_frame.shape[0])

    def resize_to_height(frame, height):
        if frame.shape[0] == height:
            return frame
        scale = height / frame.shape[0]
        return cv2.resize(frame, (max(1, int(frame.shape[1] * scale)), height))

    xyz_frame = resize_to_height(xyz_frame, target_height)
    gesture_frame = resize_to_height(gesture_frame, target_height)

    return np.hstack([xyz_frame, gesture_frame])


def main():
    # Main loop: read both cameras, process each for its assigned task, then merge output.
    logger.info("Starting Aether two-camera pipeline.")

    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "hand_landmarker.task")
    xyz_detector = None
    gesture_detector = None
    xyz_reader = None
    gesture_reader = None

    try:
        xyz_detector = create_detector(model_path)
        gesture_detector = create_detector(model_path)

        # Resolve the two independent camera sources from environment variables.
        xyz_camera_source = resolve_xyz_camera_source()
        gesture_camera_source = resolve_gesture_camera_source()
        headless = is_headless_environment()

        # Each camera runs through its own latest-frame reader.
        xyz_reader = open_reader(xyz_camera_source, "XYZ_CAMERA_SOURCE")
        gesture_reader = open_reader(gesture_camera_source, "GESTURE_CAMERA_SOURCE")

        xyz_timestamp_ms = 0
        gesture_timestamp_ms = 0
        stream_interval = resolve_terminal_stream_interval()
        last_stream_at = 0.0
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)

        right_hand_z_base = None
        gesture_hand_base_roll = None
        gesture_hand_base_pitch = None
        base_rotation_zone_owners = {"Left": None, "Right": None}
        tcp_server = JsonTcpServer(resolve_tcp_bind_host(), resolve_tcp_port())
        logger.info("TCP server listening on %s:%s", tcp_server.host, tcp_server.port)

        xyz_smoother = XYZSmoother()
        wrist_smoother = WristSmoother()
        grab_debouncer = GrabDebouncer()

        while True:
            tcp_server.poll_for_client()
            # Pull the newest frame from each camera without blocking on old buffered frames.
            xyz_image, xyz_latest_ts = xyz_reader.get_latest()
            gesture_image, gesture_latest_ts = gesture_reader.get_latest()

            if xyz_image is None and gesture_image is None:
                if not headless:
                    status = placeholder.copy()
                    cv2.putText(status, "Waiting for both camera feeds...", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.imshow("Aether Two Camera", status)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                else:
                    time.sleep(0.01)
                continue

            xyz_image, xyz_timestamp_ms, xyz_hand, _ = detect_primary_hand(
                xyz_image,
                xyz_latest_ts,
                xyz_detector,
                xyz_timestamp_ms,
                isinstance(xyz_camera_source, int),
            )
            gesture_image, gesture_timestamp_ms, gesture_hand, gesture_handedness = detect_primary_hand(
                gesture_image,
                gesture_latest_ts,
                gesture_detector,
                gesture_timestamp_ms,
                isinstance(gesture_camera_source, int),
            )

            if xyz_image is not None:
                xyz_image = border_box(xyz_image)
            if gesture_image is not None:
                gesture_image = border_box(gesture_image)

            right_xyz = None
            right_z = None
            right_base_rotation = "No hand"
            raw_right_xyz = None

            left_grab = "No hand"
            left_wrist = None
            left_base_rotation = "No hand"
            raw_left_grab = "No hand"
            raw_left_wrist = None

            if xyz_hand is not None and xyz_image is not None:
                # XYZ camera only contributes position/depth and its share of base rotation.
                draw_hand_landmarks(xyz_image, xyz_hand)
                draw_hand_name(xyz_image, "XYZ Camera", xyz_hand, (255, 120, 120))

                right_xyz = extract_xy_coordinates_for_hand(xyz_image, xyz_hand)
                right_z, right_hand_z_base, right_z_box = extract_z_coordinate_for_hand(
                    xyz_image,
                    xyz_hand,
                    False,
                    right_hand_z_base,
                )

                if right_xyz is not None:
                    right_xyz["z"] = right_z
                    raw_right_xyz = {k: right_xyz[k] for k in ("x", "y", "z")}
                    right_xyz = xyz_smoother.update(right_xyz)
                    right_base_rotation = get_base_rotation_direction(right_xyz)
                    xyz_image = draw_xy_coordinates_for_hand(xyz_image, right_xyz, color=(255, 0, 0))

                displayed_right_z = right_xyz["z"] if right_xyz is not None else right_z
                xyz_image = draw_z_overlay(
                    xyz_image,
                    displayed_right_z,
                    right_z_box,
                    right_hand_z_base is not None,
                )
            else:
                xyz_smoother.reset()

            if gesture_hand is not None and gesture_image is not None:
                # Gesture camera only contributes grab/wrist state and its share of base rotation.
                draw_hand_landmarks(gesture_image, gesture_hand)
                draw_hand_name(gesture_image, "Gesture Camera", gesture_hand, (120, 220, 255))

                gesture_xy = extract_xy_coordinates_for_hand(gesture_image, gesture_hand)
                left_base_rotation = get_base_rotation_direction(gesture_xy)

                raw_left_grab = "Grabbing" if is_grabbing(gesture_hand) else "Open"
                left_grab = grab_debouncer.update(raw_left_grab == "Grabbing")
                if gesture_hand_base_roll is not None and gesture_hand_base_pitch is not None:
                    raw_left_wrist = compute_wrist_state(
                        gesture_hand,
                        gesture_handedness,
                        gesture_hand_base_roll,
                        gesture_hand_base_pitch,
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
                    "xyz_hand": right_base_rotation,
                    "gesture_hand": left_base_rotation,
                },
            )
            # Merge both camera contributions into one shared base-rotation state.
            active_base_rotation_zones = get_active_base_rotation_zones(base_rotation_zone_owners)
            combined_base_rotation = get_combined_base_rotation_output(
                base_rotation_zone_owners,
                {
                    "xyz_hand": right_base_rotation,
                    "gesture_hand": left_base_rotation,
                },
            )
            if xyz_image is not None:
                draw_base_rotation_indicators(active_base_rotation_zones, xyz_image)
            if gesture_image is not None:
                draw_base_rotation_indicators(active_base_rotation_zones, gesture_image)

            combined_state = {
                "left_hand": {
                    "grab": left_grab,
                    "wrist": {
                        "up_down": left_wrist_ud,
                        "left_right_rotation": left_wrist_lr,
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

            now = time.monotonic()
            if (
                stream_interval == 0.0 or
                now - last_stream_at >= stream_interval
            ):
                print(json.dumps(combined_state), flush=True)
                tcp_server.send_json(combined_state)
                last_stream_at = now

            if not headless:
                # Build the combined preview window with summaries and keyboard hints.
                combined_image = compose_dual_view(xyz_image, gesture_image)
                summary_lines = [
                    f"Gesture Camera | Grab: {left_grab}",
                    f"Gesture Wrist | LR: {left_wrist_lr} | UD: {left_wrist_ud}",
                    f"XYZ Camera | XYZ: {right_xyz_text}",
                    f"Base Rotation | {combined_base_rotation}",
                ]
                draw_top_summary(combined_image, summary_lines)

                base_text = (
                    "b: set gesture wrist base"
                    if gesture_hand_base_roll is None or gesture_hand_base_pitch is None
                    else "b: reset gesture wrist base"
                )
                z_text = (
                    "r: set XYZ camera Z=0"
                    if right_hand_z_base is None
                    else "r: reset XYZ camera Z=0"
                )
                footer_line_one = f"{format_terminal_stream_rate(stream_interval)} | TCP: {tcp_server.client_status()} | [: slower  ]: faster"
                footer_line_two = f"{base_text} | {z_text} | TCP: {tcp_server.client_status()} | Esc: quit"
                draw_footer_panel(combined_image, footer_line_one, footer_line_two)

                cv2.imshow("Aether Two Camera", combined_image)
                key = cv2.waitKeyEx(1)

                if key == 27:
                    break
                if key == ord("b") and gesture_hand is not None:
                    gesture_hand_base_roll, gesture_hand_base_pitch = calibrate_wrist_base(gesture_hand, gesture_handedness)
                if key == ord("r") and xyz_hand is not None and xyz_image is not None:
                    _, right_hand_z_base, _ = extract_z_coordinate_for_hand(
                        xyz_image,
                        xyz_hand,
                        True,
                        right_hand_z_base,
                    )
                if is_stream_rate_increase_key(key):
                    stream_interval = adjust_terminal_stream_interval(stream_interval, 1)
                if is_stream_rate_decrease_key(key):
                    stream_interval = adjust_terminal_stream_interval(stream_interval, -1)

        return 0
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down two-camera pipeline.")
        return 0
    except Exception:
        logger.exception("two_camera terminated due to an unexpected error.")
        return 1
    finally:
        if xyz_reader is not None:
            xyz_reader.stop()
        if gesture_reader is not None:
            gesture_reader.stop()
        if 'tcp_server' in locals() and tcp_server is not None:
            tcp_server.close()
        if xyz_detector is not None:
            xyz_detector.close()
        if gesture_detector is not None:
            gesture_detector.close()
        if not is_headless_environment():
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
