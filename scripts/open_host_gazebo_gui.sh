#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/open_host_gazebo_gui.sh [--partition NAME]

Options:
  --partition NAME   Gazebo transport partition (default: aether)
  -h, --help         Show this help
EOF
}

# Use same default partition as container server launcher.
SIM_PARTITION="${SIM_PARTITION:-aether}"

if [[ -f /.dockerenv ]]; then
  echo "Error: This script must be run on the host, not inside devcontainer." >&2
  echo "Open a host terminal and run: ./scripts/open_host_gazebo_gui.sh" >&2
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition)
      SIM_PARTITION="$2"
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

export GZ_PARTITION="$SIM_PARTITION"
export IGN_PARTITION="$SIM_PARTITION"

if command -v gz >/dev/null 2>&1; then
  versions="$(gz sim --versions 2>/dev/null || true)"

  if printf '%s\n' "$versions" | grep -qE '^6(\.|$)'; then
    echo "Starting host Gazebo GUI with gz-sim6 (partition=$SIM_PARTITION)"
    exec gz sim --force-version 6 -g -v 4
  fi

  echo "Warning: Host does not have gz-sim6 installed."
  echo "Container server uses Gazebo Sim 6 (ign), so host gz version mismatch may prevent attach."
  echo "Install compatible host GUI with:"
  echo "  brew install osrf/simulation/gz-sim6"
  if [[ -n "$versions" ]]; then
    echo "Detected host gz sim versions:"
    printf '%s\n' "$versions"
  fi
  echo ""
  echo "Starting default host GUI anyway (may not connect to container server)..."
  exec gz sim -g -v 4
fi

if command -v ign >/dev/null 2>&1; then
  echo "Starting host Ignition Gazebo GUI (partition=$SIM_PARTITION)"
  exec ign gazebo -g -v 4
fi

echo "Error: No Gazebo GUI CLI found on host (need 'gz' or 'ign')." >&2
exit 127
