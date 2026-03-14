#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_container_gazebo_novnc.sh [--partition NAME] [--model PATH] [--display :N] [--vnc-port PORT] [--web-port PORT]

Options:
  --partition NAME   Gazebo transport partition (default: aether)
  --model PATH       SDF model path (default: /workspace/model/robot_arm.sdf)
  --display :N       Virtual X display (default: :1)
  --vnc-port PORT    x11vnc port (default: 5901)
  --web-port PORT    noVNC/websockify port (default: 6080)
  -h, --help         Show this help

Description:
  Runs Gazebo server and Gazebo GUI fully inside the devcontainer and exposes
  the GUI over noVNC (browser).
EOF
}

SIM_PARTITION="${SIM_PARTITION:-aether}"
MODEL_PATH="/workspace/model/robot_arm.sdf"
DISPLAY_NUM=":1"
VNC_PORT=5901
WEB_PORT=6080

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
    --display)
      DISPLAY_NUM="$2"
      shift 2
      ;;
    --vnc-port)
      VNC_PORT="$2"
      shift 2
      ;;
    --web-port)
      WEB_PORT="$2"
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

for cmd in Xvfb x11vnc websockify openbox; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: required command '$cmd' not found in container." >&2
    echo "Rebuild devcontainer after pulling latest Dockerfile changes." >&2
    exit 127
  fi
done

if command -v ign >/dev/null 2>&1; then
  SERVER_CMD=(ign gazebo -s -v 4 "$MODEL_PATH")
  GUI_CMD=(dbus-run-session ign gazebo -g -v 4)
elif command -v gz >/dev/null 2>&1 && gz sim -h >/dev/null 2>&1; then
  SERVER_CMD=(gz sim -s -v 4 "$MODEL_PATH")
  GUI_CMD=(dbus-run-session gz sim -g -v 4)
else
  echo "Error: no Gazebo Sim CLI found (need 'ign' or 'gz sim')." >&2
  exit 127
fi

export SIM_PARTITION
export GZ_PARTITION="$SIM_PARTITION"
export IGN_PARTITION="$SIM_PARTITION"
export DISPLAY="$DISPLAY_NUM"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-root}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export QT_X11_NO_MITSHM="${QT_X11_NO_MITSHM:-1}"

mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

cleanup() {
  for pid_var in NOVNC_PID VNC_PID GUI_PID SERVER_PID WM_PID XVFB_PID; do
    pid="${!pid_var:-}"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

# Stale GUI processes can keep Qt selection ownership and cause an apparent
# black noVNC screen. Clean them before launching a fresh GUI session.
if pgrep -f '^ruby /usr/bin/ign gazebo -g' >/dev/null 2>&1; then
  echo "Stopping stale Gazebo GUI processes..."
  pgrep -f '^ruby /usr/bin/ign gazebo -g' | xargs -r kill -9 || true
fi

has_mapped_window() {
  if ! command -v xwininfo >/dev/null 2>&1; then
    # If xwininfo is unavailable, skip this validation.
    return 0
  fi

  DISPLAY="$DISPLAY" xwininfo -root -tree 2>/dev/null | awk '
    /^[[:space:]]+0x/ {
      line = $0
      if (line ~ /Qt Selection Owner/) next
      if (line ~ /Openbox/) next
      if (line ~ / 1x1\+/) next
      found = 1
    }
    END { exit(found ? 0 : 1) }
  '
}

echo "Starting Xvfb on display $DISPLAY..."
Xvfb "$DISPLAY" -screen 0 1920x1080x24 -nolisten tcp +extension GLX +render -noreset >/tmp/xvfb.log 2>&1 &
XVFB_PID=$!

sleep 1

echo "Starting openbox window manager..."
openbox >/tmp/openbox.log 2>&1 &
WM_PID=$!

echo "Starting Gazebo server in container (partition=$SIM_PARTITION)..."
"${SERVER_CMD[@]}" >/tmp/gazebo_server.log 2>&1 &
SERVER_PID=$!

sleep 2

echo "Starting Gazebo GUI in container..."
"${GUI_CMD[@]}" >/tmp/gazebo_gui.log 2>&1 &
GUI_PID=$!

echo "Starting VNC server on port $VNC_PORT..."
x11vnc -display "$DISPLAY" -forever -shared -nopw -noxdamage -rfbport "$VNC_PORT" >/tmp/x11vnc.log 2>&1 &
VNC_PID=$!

echo "Starting noVNC on port $WEB_PORT..."
websockify --web /usr/share/novnc/ "$WEB_PORT" "localhost:$VNC_PORT" >/tmp/novnc.log 2>&1 &
NOVNC_PID=$!

# Give Qt a moment to map its top-level window, then validate.
sleep 4

if ! kill -0 "$GUI_PID" 2>/dev/null; then
  echo "Error: Gazebo GUI process exited early." >&2
  echo "See: /tmp/gazebo_gui.log" >&2
  exit 2
fi

if ! has_mapped_window; then
  echo "Error: Gazebo GUI did not map a visible window on the virtual display." >&2
  echo "This environment can show a black noVNC screen for Gazebo Sim GUI." >&2
  echo "Fallback options:" >&2
  echo "  1) Host-only reliable mode: ./scripts/run_host_gazebo_full.sh" >&2
  echo "  2) Container backend only: ./launch_gazebo.sh" >&2
  echo "Logs: /tmp/gazebo_gui.log and ~/.ignition/auto_default.log" >&2
  exit 2
fi

echo ""
echo "Gazebo noVNC is ready. Open:"
echo "  http://localhost:$WEB_PORT/vnc.html?autoconnect=1&resize=scale"
echo ""
echo "Logs:"
echo "  /tmp/gazebo_server.log"
echo "  /tmp/gazebo_gui.log"
echo "  /tmp/x11vnc.log"
echo "  /tmp/novnc.log"
echo ""
echo "Press Ctrl+C to stop all processes."

wait "$GUI_PID"
