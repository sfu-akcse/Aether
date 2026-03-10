import os
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Hand Connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17),                                # Palm to Pinky Base
]


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
cap = cv2.VideoCapture(0)
timestamp_ms = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)

    # BGR to RGB conversion for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Hand detection
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33  # Approx. 30 FPS

    # Draw landmarks on the original image
    image = draw_hand_landmarks(image, detection_result)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:  # Exit on 'ESC' key
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
