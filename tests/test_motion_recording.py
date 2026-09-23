import csv

import pytest

from motion_recorder import FIELDNAMES, build_row
from motion_replay import read_motion_file, replay_motion, to_live_format


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# build_row() should fill in blanks when a frame is missing data
def test_build_row_with_full_data():
    row = build_row(
        elapsed=1.234,
        xy={"x": 10.5, "y": -20.0, "pixel_x": 1, "pixel_y": 2},
        z=42,
        wrist={"roll_deg": 10, "pitch_deg": -10},
        grab="Grabbing",
    )
    assert row == {
        "timestamp": 1.234,
        "x": 10.5,
        "y": -20.0,
        "z": 42,
        "wrist_roll_deg": 10,
        "wrist_pitch_deg": -10,
        "grab": "Grabbing",
    }


def test_build_row_with_no_hand():
    row = build_row(elapsed=0.5, xy=None, z=None, wrist=None, grab=None)
    assert row == {
        "timestamp": 0.5,
        "x": "",
        "y": "",
        "z": "",
        "wrist_roll_deg": "",
        "wrist_pitch_deg": "",
        "grab": "",
    }


# reading a recorded file back should restore types and round-trip cleanly
def test_read_motion_file_round_trip(tmp_path):
    csv_path = tmp_path / "motion.csv"
    write_csv(csv_path, [
        build_row(0.0, {"x": 1.0, "y": 2.0}, 3, {"roll_deg": 4, "pitch_deg": -5}, "Open"),
        build_row(0.033, None, None, None, None),
    ])

    rows = read_motion_file(csv_path)
    assert rows[0] == {
        "timestamp": 0.0, "x": 1.0, "y": 2.0, "z": 3,
        "wrist_roll_deg": 4, "wrist_pitch_deg": -5, "grab": "Open",
    }
    assert rows[1] == {
        "timestamp": 0.033, "x": None, "y": None, "z": None,
        "wrist_roll_deg": None, "wrist_pitch_deg": None, "grab": None,
    }


def test_to_live_format_matches_live_output_shape():
    row = {"timestamp": 1.0, "x": 5.0, "y": 6.0, "z": 7,
           "wrist_roll_deg": None, "wrist_pitch_deg": None, "grab": "Open"}
    output = to_live_format(row)
    assert output["right_hand"]["xyz"] == {"x": 5.0, "y": 6.0, "z": 7}
    assert output["left_hand"]["wrist"] == {"up_down": "Not calibrated", "left_right_rotation": "Not calibrated"}
    assert output["left_hand"]["grab"] == "Open"


def test_to_live_format_classifies_wrist_direction():
    row = {"timestamp": 1.0, "x": None, "y": None, "z": None,
           "wrist_roll_deg": 20, "wrist_pitch_deg": -20, "grab": "Grabbing"}
    output = to_live_format(row)
    assert output["right_hand"]["xyz"] is None
    assert output["left_hand"]["wrist"] == {"up_down": "Down", "left_right_rotation": "Right"}
    assert output["left_hand"]["grab"] == "Grabbing"


def test_to_live_format_defaults_missing_left_hand_data():
    row = {"timestamp": 1.0, "x": None, "y": None, "z": None,
           "wrist_roll_deg": None, "wrist_pitch_deg": None, "grab": None}
    output = to_live_format(row)
    assert output["left_hand"]["grab"] == "No hand"
    assert output["left_hand"]["wrist"] == {"up_down": "Not calibrated", "left_right_rotation": "Not calibrated"}


def test_replay_motion_fixed_speed_sleeps_at_fixed_rate(tmp_path, monkeypatch):
    csv_path = tmp_path / "motion.csv"
    write_csv(csv_path, [
        build_row(0.0, {"x": 0.0, "y": 0.0}, 0, None, "Open"),
        build_row(5.0, {"x": 1.0, "y": 1.0}, 0, None, "Open"),
        build_row(5.1, {"x": 2.0, "y": 2.0}, 0, None, "Open"),
    ])

    sleeps = []
    monkeypatch.setattr("motion_replay.time.sleep", lambda s: sleeps.append(s))

    outputs = list(replay_motion(csv_path, speed="fixed", rate=10.0))

    assert len(outputs) == 3
    assert sleeps == pytest.approx([0.1, 0.1])


def test_replay_motion_original_speed_uses_recorded_gaps(tmp_path, monkeypatch):
    csv_path = tmp_path / "motion.csv"
    write_csv(csv_path, [
        build_row(0.0, {"x": 0.0, "y": 0.0}, 0, None, "Open"),
        build_row(0.5, {"x": 1.0, "y": 1.0}, 0, None, "Open"),
        build_row(0.8, {"x": 2.0, "y": 2.0}, 0, None, "Open"),
    ])

    sleeps = []
    monkeypatch.setattr("motion_replay.time.sleep", lambda s: sleeps.append(s))

    list(replay_motion(csv_path, speed="original"))

    assert sleeps == pytest.approx([0.5, 0.3])
