import numpy as np
from unittest.mock import MagicMock
from main import draw_xy_coordinates

# sample
def add(a,b):
  return a+b

def test_pytest_add_success():
    assert add(1, 1) == 2   

def test_pytest_add_edge_case():
    assert add(0, 0) == 0   

def test_pytest_add_negative():
    assert add(-1, -1) == -2  

# Pytests for XY Coordinate Detection
# environment
def make_landmark(x,y):
    landmark = MagicMock()
    landmark.x = x
    landmark.y = y
    return landmark

def make_mediapipe_result(landmarks):
    landmark_objects = []
    for x,y in landmarks:
        landmark_objects.append(make_landmark(x,y))

    result = MagicMock()
    result.hand_landmarks = [landmark_objects]

    return result

# pytest - When some of the hand landmarks are not detected
def test_handlandmarks_disappeared():
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    landmarks = [(0.5, 0.5)] * 21
    landmarks[9]  = (1.5, 1.3)   # outside screen
    landmarks[13] = (-0.2, 0.5)  # outside screen
    landmarks[17] = (0.5, 1.8)   # outside screen

    detection_result = make_mediapipe_result(landmarks)

    try:
        result = draw_xy_coordinates(image, detection_result)
        assert result is not None
    except Exception as e:
        pytest.fail(f"Error: {e}")
