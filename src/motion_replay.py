import argparse
import csv
import json
import socket
import time

from WristDetection import classify_pitch, classify_roll

def _parse_float(value):
    return float(value) if value not in (None, "") else None

def _parse_int(value):
    return int(float(value)) if value not in (None, "") else None

def read_motion_file(path):
    """Read a recorded CSV file into a list of typed row dicts."""
    rows = []
    with open(path, newline="") as f:
        for raw in csv.DictReader(f):
            rows.append({
                "timestamp": _parse_float(raw["timestamp"]),
                "x": _parse_float(raw["x"]),
                "y": _parse_float(raw["y"]),
                "z": _parse_int(raw["z"]),
                "wrist_roll_deg": _parse_int(raw["wrist_roll_deg"]),
                "wrist_pitch_deg": _parse_int(raw["wrist_pitch_deg"]),
                "grab": raw["grab"] or None,
            })
    return rows

def to_live_format(row):
    """Convert a recorded row into the same JSON shape main2.py prints live.

    Blank fields fall back to "Not calibrated" / "No hand", same as main2.py.
    """
    has_xyz = row["x"] is not None and row["y"] is not None
    has_wrist = row["wrist_roll_deg"] is not None and row["wrist_pitch_deg"] is not None
    return {
        "timestamp": row["timestamp"],
        "left_hand": {
            "grab": row["grab"] or "No hand",
            "wrist": {
                "up_down": classify_pitch(row["wrist_pitch_deg"]) if has_wrist else "Not calibrated",
                "left_right_rotation": classify_roll(row["wrist_roll_deg"]) if has_wrist else "Not calibrated",
            },
        },
        "right_hand": {
            "xyz": {"x": row["x"], "y": row["y"], "z": row["z"]} if has_xyz else None,
        },
    }

def replay_motion(path, speed="original", rate=30.0):
    """Yield live-format dicts, paced like the original recording.

    "original" replays at the recorded timestamps; "fixed" waits 1/rate
    seconds between every frame instead.
    """
    rows = read_motion_file(path)
    prev_timestamp = None

    for row in rows:
        if prev_timestamp is not None:
            if speed == "original":
                delay = max(0.0, row["timestamp"] - prev_timestamp)
            elif speed == "fixed":
                delay = 1.0 / rate if rate > 0 else 0.0
            else:
                raise ValueError(f"Unknown speed mode: {speed}")
            time.sleep(delay)
        prev_timestamp = row["timestamp"]
        yield to_live_format(row)

def main():
    parser = argparse.ArgumentParser(description="Replay recorded hand-tracking motion data.")
    parser.add_argument("file", help="Path to a CSV file recorded by motion_recorder.py")
    parser.add_argument("--speed", choices=["original", "fixed"], default="original",
                         help="Replay at recorded timing (original) or a fixed rate (fixed).")
    parser.add_argument("--rate", type=float, default=30.0,
                         help="Frames per second to use when --speed=fixed (default: 30).")
    parser.add_argument("--socket", metavar="HOST:PORT", default=None,
                         help="Also send each frame as a newline-delimited JSON line over a "
                              "TCP socket to HOST:PORT (e.g. the robot control pipeline).")
    args = parser.parse_args()

    sock = None
    if args.socket:
        host, _, port = args.socket.rpartition(":")
        if not host or not port:
            parser.error("--socket must be in the form HOST:PORT")
        sock = socket.create_connection((host, int(port)))

    try:
        for output in replay_motion(args.file, speed=args.speed, rate=args.rate):
            line = json.dumps(output)
            print(line, flush=True)
            if sock is not None:
                sock.sendall((line + "\n").encode("utf-8"))
    finally:
        if sock is not None:
            sock.close()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
