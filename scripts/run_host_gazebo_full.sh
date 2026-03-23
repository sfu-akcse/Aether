#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_host_gazebo_full.sh [--partition NAME] [--model PATH]

Options:
  --partition NAME   Gazebo transport partition (default: aether)
  --model PATH       SDF model path (default: model/robot_arm.sdf)
  -h, --help         Show this help

Description:
  Runs BOTH Gazebo server and GUI on the host using the same CLI/version.
  This avoids host<->container discovery and rendering issues.
EOF
}

if [[ -f /.dockerenv ]]; then
  echo "Error: This script must be run on the host, not inside devcontainer." >&2
  echo "Open a host terminal and run: ./scripts/run_host_gazebo_full.sh" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_PATH="${PROJECT_ROOT}/model/robot_arm.sdf"
SIM_PARTITION="${SIM_PARTITION:-aether}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition)
      SIM_PARTITION="$2"
      shift 2
      ;;
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Error: model file not found: $MODEL_PATH" >&2
  exit 1
fi

export GZ_PARTITION="$SIM_PARTITION"
export IGN_PARTITION="$SIM_PARTITION"

SERVER_CMD=()
GUI_CMD=()

if command -v gz >/dev/null 2>&1; then
  versions="$(gz sim --versions 2>/dev/null || true)"
  if printf '%s\n' "$versions" | grep -qE '^6(\.|$)'; then
    echo "Using gz sim version 6 on host (partition=$SIM_PARTITION)"
    SERVER_CMD=(gz sim --force-version 6 -s -v 4 "$MODEL_PATH")
    GUI_CMD=(gz sim --force-version 6 -g -v 4)
  else
    echo "Using default gz sim on host (partition=$SIM_PARTITION)"
    SERVER_CMD=(gz sim -s -v 4 "$MODEL_PATH")
    GUI_CMD=(gz sim -g -v 4)
  fi
elif command -v ign >/dev/null 2>&1; then
  echo "Using ign gazebo on host (partition=$SIM_PARTITION)"
  SERVER_CMD=(ign gazebo -s -v 4 "$MODEL_PATH")
  GUI_CMD=(ign gazebo -g -v 4)
else
  echo "Error: neither 'gz' nor 'ign' found on host PATH." >&2
  exit 127
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting host Gazebo server..."
"${SERVER_CMD[@]}" &
SERVER_PID=$!

# Give server time to initialize before opening GUI.
sleep 2

echo "Starting host Gazebo GUI..."
"${GUI_CMD[@]}"
