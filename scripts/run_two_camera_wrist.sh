#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

print_usage() {
  cat <<'EOF'
Usage:
  scripts/run_two_camera_wrist.sh [--front-port PORT] [--side-port PORT]
                                   [--front-source SOURCE] [--side-source SOURCE]

Run the two-camera wrist detector (inside devcontainer recommended). Expects
two host webcam streamers already running, one per camera, e.g.:

  # host, terminal 1 (front camera)
  python3 scripts/host_webcam_stream.py --port 8080 --camera-index 0

  # host, terminal 2 (side camera)
  python3 scripts/host_webcam_stream.py --port 8081 --camera-index 1

  # devcontainer
  scripts/run_two_camera_wrist.sh --front-port 8080 --side-port 8081

Pass --front-source/--side-source instead of --front-port/--side-port to
target a source directly (camera index or stream URL), same convention as
scripts/run_webcam_pipeline.sh.
EOF
}

require_python3() {
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required but not found in PATH." >&2
    exit 1
  fi
}

front_port=8080
side_port=8081
front_source=""
side_source=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --front-port)
      front_port="$2"
      shift 2
      ;;
    --side-port)
      side_port="$2"
      shift 2
      ;;
    --front-source)
      front_source="$2"
      shift 2
      ;;
    --side-source)
      side_source="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      print_usage
      exit 1
      ;;
  esac
done

require_python3

if [[ -z "${front_source}" ]]; then
  front_source="http://host.docker.internal:${front_port}/video.mjpg"
fi
if [[ -z "${side_source}" ]]; then
  side_source="http://host.docker.internal:${side_port}/video.mjpg"
fi

export CAMERA_SOURCE_FRONT="${front_source}"
export CAMERA_SOURCE_SIDE="${side_source}"
echo "Running two-camera wrist detector with:"
echo "  CAMERA_SOURCE_FRONT=${CAMERA_SOURCE_FRONT}"
echo "  CAMERA_SOURCE_SIDE=${CAMERA_SOURCE_SIDE}"

exec python3 "${PROJECT_ROOT}/src/TwoCameraWristDetection.py"