#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

print_usage() {
  cat <<'EOF'
Usage:
  scripts/run_webcam_pipeline.sh host [--port PORT] [--camera-index IDX|auto] [--width W] [--height H] [--fps FPS]
  scripts/run_webcam_pipeline.sh cv [--port PORT] [--source CAMERA_SOURCE]

Subcommands:
  host    Run host-side webcam MJPEG streamer.
  cv      Run CV app (inside devcontainer recommended) using CAMERA_SOURCE.

Examples:
  # On macOS host
  scripts/run_webcam_pipeline.sh host --port 8080

  # Inside devcontainer
  scripts/run_webcam_pipeline.sh cv --port 8080

  # Inside devcontainer with explicit source URL
  scripts/run_webcam_pipeline.sh cv --source http://host.docker.internal:8080/video.mjpg
EOF
}

require_python3() {
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required but not found in PATH." >&2
    exit 1
  fi
}

require_host_cv2() {
  if ! python3 -c "import cv2" >/dev/null 2>&1; then
    cat >&2 <<'EOF'
Host dependency missing: cv2 (OpenCV) for python3.

Install once on host with one of:
  python3 -m pip install opencv-python
  conda install -c conda-forge opencv
EOF
    exit 1
  fi
}

run_host_streamer() {
  local port=8080
  local camera_index="auto"
  local width=640
  local height=480
  local fps=20

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --port)
        port="$2"
        shift 2
        ;;
      --camera-index)
        camera_index="$2"
        shift 2
        ;;
      --width)
        width="$2"
        shift 2
        ;;
      --height)
        height="$2"
        shift 2
        ;;
      --fps)
        fps="$2"
        shift 2
        ;;
      -h|--help)
        print_usage
        exit 0
        ;;
      *)
        echo "Unknown option for 'host': $1" >&2
        print_usage
        exit 1
        ;;
    esac
  done

  require_python3
  require_host_cv2

  echo "Starting host webcam stream on port ${port}..."
  exec python3 "${SCRIPT_DIR}/host_webcam_stream.py" \
    --port "${port}" \
    --camera-index "${camera_index}" \
    --width "${width}" \
    --height "${height}" \
    --fps "${fps}"
}

run_cv() {
  local port=8080
  local source=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --port)
        port="$2"
        shift 2
        ;;
      --source)
        source="$2"
        shift 2
        ;;
      -h|--help)
        print_usage
        exit 0
        ;;
      *)
        echo "Unknown option for 'cv': $1" >&2
        print_usage
        exit 1
        ;;
    esac
  done

  require_python3

  if [[ -z "${source}" ]]; then
    source="http://host.docker.internal:${port}/video.mjpg"
  fi

  export CAMERA_SOURCE="${source}"
  echo "Running CV with CAMERA_SOURCE=${CAMERA_SOURCE}"
  exec python3 "${PROJECT_ROOT}/src/main.py"
}

main() {
  if [[ $# -lt 1 ]]; then
    print_usage
    exit 1
  fi

  local subcommand="$1"
  shift

  case "${subcommand}" in
    host)
      run_host_streamer "$@"
      ;;
    cv)
      run_cv "$@"
      ;;
    -h|--help)
      print_usage
      ;;
    *)
      echo "Unknown subcommand: ${subcommand}" >&2
      print_usage
      exit 1
      ;;
  esac
}

main "$@"
