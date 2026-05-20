import numpy as np
from unittest.mock import MagicMock

# sample
def add(a,b):
  return a+b

def test_pytest_add_success():
    assert add(1, 1) == 2   

def test_pytest_add_edge_case():
    assert add(0, 0) == 0   

def test_pytest_add_negative():
    assert add(-1, -1) == -2  

# XY Coordinate Detection
# environment
def make_landmark(x,y):
    landmark = MagicMock()
    landmark.x = x
    landmark.y = y
    return landmark

# get coordinate data of "hands"
# def make_mediapipe_result(hands):



# pytests
# 1.
# 2.
# 3.
# 4.