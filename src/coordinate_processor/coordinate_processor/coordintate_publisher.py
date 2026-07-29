import os
import glob
from ament_index_python.packages import get_package_share_directory
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import time
import mediapipe as mp

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from .GrabbingMotion2 import is_grabbing
from .MultiHandTracker import HandSide, MultiHandTracker
from .WristDetection import calibrate_wrist_base, compute_wrist_state
from .aether_logger import setup_logger
from .base_rotation import base_rotation_x, border_box, get_base_rotation_direction

from aether_interfaces.msg import HandState
from .hand_tracker import (
    is_headless_environment, 
    resolve_camera_source,
    open_camera_capture,
    LatestFrameReader,
    draw_hand_landmarks,
    extract_xy_coordinates_for_hand,
    draw_xy_coordinates_for_hand,
    extract_z_coordinate_for_hand,
    draw_z_overlay,
    get_hand_label_position,
    draw_hand_name,
    draw_top_summary,
    HAND_CONNECTIONS,
    calibrate_z_side,       
    extract_z_side,         
    draw_z_side_overlay,
)

class CoordinatePublisher(Node):

    def __init__(self):
        super().__init__('coordinate_publisher')

        self.logger = setup_logger("aether-system.log", "AETHER.VISION.MAIN2")

        package_share_directory = get_package_share_directory('coordinate_processor')
        model_path = os.path.join(package_share_directory, 'model', 'hand_landmarker.task')
        self.detector = None
        self.reader = None
        self.cap = None

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.tracker = MultiHandTracker(is_mirrored=True)
        self.logger.info("MediaPipe HandLandmarker initialized with model=%s", model_path)
        self.headless = is_headless_environment()
        # self.headless = True
        self.camera_source = resolve_camera_source()

        if isinstance(self.camera_source, int):
            self.cap = open_camera_capture(self.camera_source)

        if isinstance(self.camera_source, int) and not self.cap.isOpened():
            device_path = f"/dev/video{self.camera_source}"
            visible_devices = ", ".join(sorted(glob.glob("/dev/video*"))) or "none"

            if not os.path.exists(device_path):
                raise RuntimeError(
                    "Failed to open camera source: "
                    f"{self.camera_source}. "
                    f"{device_path} is not available in this container. "
                    f"Visible camera devices: {visible_devices}. "
                    "If you are running in a Linux devcontainer, pass through your camera "
                    "with run args like `--device=/dev/video0:/dev/video0` "
                    "(optionally `--group-add=video`) and rebuild the container, "
                    "or use a stream URL for CAMERA_SOURCE."
                )

            raise RuntimeError(
                "Failed to open camera source: "
                f"{self.camera_source}. "
                "Set CAMERA_SOURCE to an index (e.g. 0) or stream URL "
                "(e.g. http://host.docker.internal:8080/video.mjpg)."
            )

        self.logger.info("Using CAMERA_SOURCE=%s", self.camera_source)
        self.reader = LatestFrameReader(self.camera_source, cap=self.cap)
        self.reader.start()

        self.timestamp_ms = 0
        self.black_frame_count = 0
        self.waiting_warned = False
        self.placeholder = np.zeros((480, 640, 3), dtype=np.uint8)

        self.right_hand_z_base = None
        self.right_hand_z_side_base = None 
        self.left_hand_base_roll = None
        self.left_hand_base_pitch = None

        

        # Created custom .msg to wrap all metrics into a single message type
        # Queue size of 10 messages
        self.publisher_ = self.create_publisher(HandState, 'HandStates', 10)
        timer_period = 0.5 # seconds
        self.timer = self.create_timer(timer_period, self.camera_callback)


    # Need to run mediapipe through here to get metrics and publish them
    def camera_callback(self):
        msg = HandState()
        msg.x = 0.0
        msg.y = 0.0
        msg.z = 0.0
        msg.wrist_x = 0.0
        msg.wrist_y = 0.0
        msg.hand_closed = False

        image, latest_ts = self.reader.get_latest()

        if image is None:
            if not self.headless:
                status = self.placeholder.copy()
                cv2.putText(status, "Waiting for camera frames...", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.imshow("Aether Main2", status)
                if cv2.waitKey(1) & 0xFF == 27:
                    raise KeyboardInterrupt
            else:
                time.sleep(0.01)
            return
        
        frame_age = time.monotonic() - latest_ts
        if frame_age > 1.0 and not self.waiting_warned:
            self.logger.warning("Frame stream appears stale (>1s).")
            self.waiting_warned = True
        if frame_age <= 1.0:
            self.waiting_warned = False

        if image.size > 0:
            if float(image.mean()) < 2.0:
                self.black_frame_count += 1
            else:
                self.black_frame_count = 0

        if self.black_frame_count == 45:
            self.logger.warning(
                "Received many near-black frames from CAMERA_SOURCE. "
                "If using host stream, verify host preview at http://localhost:8080/ "
                "and try another host camera index."
            )

        if isinstance(self.camera_source, int):
            image = cv2.flip(image, 1)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        now_ms = int(time.monotonic() * 1000)
        self.timestamp_ms = max(self.timestamp_ms + 1, now_ms)
        detection_result = self.detector.detect_for_video(mp_image, self.timestamp_ms)
        self.tracker.update(detection_result)

        image = border_box(image)

        right_state = self.tracker.get_hand(HandSide.RIGHT)
        left_state = self.tracker.get_hand(HandSide.LEFT)

        self.right_xyz = None
        right_z = None
        right_z_box = None
        right_base_rotation = "No hand"

        self.left_xy = None
        self.left_grab = False
        self.left_wrist = None
        left_base_rotation = "No hand"

        if right_state is not None:
            right_hand = right_state.landmarks
            draw_hand_landmarks(image, right_hand)
            draw_hand_name(image, "Right Hand", right_hand, (255, 120, 120))

            self.right_xyz = extract_xy_coordinates_for_hand(image, right_hand)
            right_z, right_hand_z_base, right_z_box = extract_z_coordinate_for_hand(
                image,
                right_hand,
                False,
                self.right_hand_z_base,
            )
            self.right_hand_z_base = right_hand_z_base

            if self.right_xyz is not None:
                self.right_xyz["z"] = right_z
                right_base_rotation = get_base_rotation_direction(self.right_xyz)
                image = draw_xy_coordinates_for_hand(image, self.right_xyz, color=(255, 0, 0))
                base_rotation_x(self.right_xyz, image)

            image = draw_z_overlay(image, right_z, right_z_box)

            right_z_side = extract_z_side(right_hand, self.right_hand_z_side_base)
            image = draw_z_side_overlay(image, right_z_side, right_hand)
            if self.right_xyz is not None:
                self.right_xyz["z_side"] = right_z_side
        else:
            cv2.putText(image, "Right Hand not detected", (20, image.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if left_state is not None:
            left_hand = left_state.landmarks
            draw_hand_landmarks(image, left_hand)
            draw_hand_name(image, "Left Hand", left_hand, (120, 220, 255))

            self.left_xy = extract_xy_coordinates_for_hand(image, left_hand)
            left_base_rotation = get_base_rotation_direction(self.left_xy)
            if self.left_xy is not None:
                base_rotation_x(self.left_xy, image)

            self.left_grab = True if is_grabbing(left_hand) else False
            if self.left_hand_base_roll is not None and self.left_hand_base_pitch is not None:
                self.left_wrist = compute_wrist_state(
                    left_hand,
                    "Left",
                    self.left_hand_base_roll,
                    self.left_hand_base_pitch,
                )
        else:
            cv2.putText(image, "Left Hand not detected", (20, image.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        self.left_wrist_lr = self.left_wrist["roll_direction"] if self.left_wrist else "Not calibrated"
        self.left_wrist_ud = self.left_wrist["pitch_direction"] if self.left_wrist else "Not calibrated"
        right_xyz_text = (
            f"{self.right_xyz['x']}, {self.right_xyz['y']}, {self.right_xyz['z']}"
            if self.right_xyz is not None
            else "No hand"
        )    

        summary_lines = [
            f"Left Hand | Grab: {self.left_grab}",
            f"Left Wrist | LR: {self.left_wrist_lr} | UD: {self.left_wrist_ud}",
            f"Right Hand | XYZ: {right_xyz_text}", 
            f"Base Rotation | Left: {left_base_rotation} | Right: {right_base_rotation}",
        ]
        draw_top_summary(image, summary_lines)

        if self.left_hand_base_roll is None or self.left_hand_base_pitch is None:
            cv2.putText(image, "Press 'b' to set left wrist base", (20, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        else:
            cv2.putText(image, "b = reset left wrist base", (20, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        if self.right_hand_z_base is None:
            cv2.putText(image, "Press 'r' to set right-hand Z=0", (320, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        else:
            cv2.putText(image, "r = reset right-hand Z=0", (320, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Terminal State replacement
        if self.right_xyz is not None:
            msg.x = float(self.right_xyz['x'])
            msg.y = float(self.right_xyz['y'])
            msg.z = float(self.right_xyz['z']) if self.right_xyz['z'] is not None else 0.0
        if self.left_grab is not None:
            msg.hand_closed = self.left_grab
        if self.left_wrist is not None:
            msg.wrist_x = self.left_wrist_lr if self.left_wrist_lr != "Not calibrated" else 0.0
            msg.wrist_y = self.left_wrist_ud if self.left_wrist_ud != "Not calibrated" else 0.0
        self.publisher_.publish(msg)

        
        if not self.headless:
            cv2.imshow("Aether Main2", image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                raise KeyboardInterrupt
            if key == ord("b") and left_state is not None:
                left_hand = left_state.landmarks
                self.left_hand_base_roll, self.left_hand_base_pitch = calibrate_wrist_base(left_hand, "Left")
            if key == ord("z") and right_state is not None:  
                right_hand = right_state.landmarks
                self.right_hand_z_side_base = calibrate_z_side(right_hand)
            if key == ord("r") and right_state is not None:
                right_hand = right_state.landmarks
                _, self.right_hand_z_base, _ = extract_z_coordinate_for_hand(
                    image,
                    right_hand,
                    True,
                    self.right_hand_z_base,
                )

    
    # Override destroy node function to safely close camera, thread, etc.
    def destroy_node(self):
        self.get_logger().info('Shutting down vision pipeline cleanly...')
        if self.reader is not None:
            self.reader.stop()
        if self.detector is not None:
            self.detector.close()
        if not is_headless_environment():
            cv2.destroyAllWindows()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        # Instantiate your specific class
        coord_publisher = CoordinatePublisher()
        rclpy.spin(coord_publisher)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Clean shutdown
        if 'coord_publisher' in locals():
            coord_publisher.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()