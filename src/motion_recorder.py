import time
from pathlib import Path

FIELDNAMES = ["timestamp", "x", "y", "z", "wrist_roll_deg", "wrist_pitch_deg", "grab"]

RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "recordings"

# right hand = XYZ, left hand = wrist + grab (matches main_recorder.py)
SNAP_BAND = 18  # degrees per output step
SNAP_MAX = 50   # max output value


def snap_degrees(value):
    """Round a delta to the nearest 10°, capped at SNAP_MAX."""
    abs_val = abs(value)
    sign = 1 if value >= 0 else -1
    step_count = round(abs_val / SNAP_BAND)
    return int(sign * min(SNAP_MAX, step_count * 10))


def default_output_path() -> Path:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return RECORDINGS_DIR / time.strftime("motion_%Y%m%d_%H%M%S.csv")


def build_row(elapsed, xy, z, wrist, grab):
    return {
        "timestamp": round(elapsed, 3),
        "x": xy["x"] if xy else "",
        "y": xy["y"] if xy else "",
        "z": z if z is not None else "",
        "wrist_roll_deg": wrist["roll_deg"] if wrist else "",
        "wrist_pitch_deg": wrist["pitch_deg"] if wrist else "",
        "grab": grab if grab is not None else "",
    }
