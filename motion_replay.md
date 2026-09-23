# Motion Recording & Replay

Record live hand-tracking output to a CSV file and replay it later without a
camera. Useful for testing coordinate mapping, smoothing, socket
communication with the robot arm, and robot movement against a fixed,
repeatable input instead of live camera data.

- `src/main_recorder.py` — same dual-hand tracking as `src/main2.py` (right
  hand = XYZ, left hand = wrist tilt + grab), plus a key to record it to a
  CSV. Separate file so `main.py`/`main2.py` stay untouched.
- `src/motion_recorder.py` — the CSV row-building helpers `main_recorder.py`
  uses (no camera code).
- `src/motion_replay.py` — reads a recorded CSV back and replays it in the
  same JSON shape `main2.py`/`main_recorder.py` print live.

## Quick start

```bash
# Terminal 1 — host
./scripts/run_main_recorder.sh host --port 8080

# Terminal 2 — devcontainer
./scripts/run_main_recorder.sh cv --port 8080
```

## Recording

Press `s` in the window to start recording, `s` again to stop. (`b`
calibrates the left wrist baseline, `r` zeroes the right hand's depth — set
these first if you want wrist/grab or Z data captured.)

- **Prints to the terminal** running `main_recorder.py`:
  - `[REC] Recording started -> recordings/motion_<timestamp>.csv`
  - `[REC] Recording stopped. N frames written to recordings/motion_<timestamp>.csv`
- **File name:** auto-generated as `motion_<YYYYMMDD_HHMMSS>.csv`, saved in
  `recordings/` at the project root.

## Replaying

```bash
# 1) original recorded pacing
PYTHONPATH=src python3 src/motion_replay.py recordings/motion_20260916_053103.csv

# 2) fixed-rate playback, e.g. 30 fps regardless of how it was recorded
PYTHONPATH=src python3 src/motion_replay.py recordings/motion_20260916_053103.csv --speed fixed --rate 30

# 3) also stream each frame over a TCP socket to the control pipeline
PYTHONPATH=src python3 src/motion_replay.py recordings/motion_20260916_053103.csv --socket 127.0.0.1:9000
```

Each replayed frame prints as one JSON line, in the same shape
`main_recorder.py`/`main2.py` print live:

```json
{"timestamp": 0.1, "left_hand": {"grab": "Open", "wrist": {"up_down": "Neutral", "left_right_rotation": "Center"}}, "right_hand": {"xyz": {"x": 12.0, "y": 22.0, "z": 6}}}
```
