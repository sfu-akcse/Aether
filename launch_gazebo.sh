#!/bin/bash
# launch_gazebo.sh - Launch Ignition Gazebo on macOS
# Usage: ./launch_gazebo.sh
#
# On macOS, ign gazebo GUI has known issues with OGRE rendering (NSWindow
# threading + protobuf conflicts). This script runs the simulation server
# headlessly. Use RViz2 or `ign topic` CLI tools to inspect the simulation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="$SCRIPT_DIR/model/robot_arm.sdf"
SIM_PARTITION="${SIM_PARTITION:-aether}"

# Export both naming conventions for cross-version Gazebo tools.
export GZ_PARTITION="$SIM_PARTITION"
export IGN_PARTITION="$SIM_PARTITION"

SIM_CLI=""
SIM_CMD=()

# Prefer ign gazebo (Ubuntu apt / Fortress), then fall back to gz sim.
if command -v ign >/dev/null 2>&1; then
    SIM_CLI="ign"
    SIM_CMD=(ign gazebo -s -v 4 "$MODEL_PATH")
elif command -v gz >/dev/null 2>&1 && gz sim -h >/dev/null 2>&1; then
    SIM_CLI="gz"
    SIM_CMD=(gz sim -s -v 4 "$MODEL_PATH")
else
    echo "Error: Gazebo Sim CLI not found."
    echo "Install one of:"
    echo "  - apt: ignition-fortress (provides: ign gazebo)"
    echo "  - conda: ign-gazebo / gz-sim (provides: gz sim)"
    exit 127
fi

# If conda exists, try to activate the project env, but keep going on failure.
if command -v conda >/dev/null 2>&1; then
    if [[ "${CONDA_DEFAULT_ENV:-}" != "ros2-robot-arm" ]]; then
        echo "Conda detected. Attempting to activate ros2-robot-arm..."
        if eval "$(conda shell.bash hook 2>/dev/null)" && conda activate ros2-robot-arm; then
            echo "Activated conda env: ros2-robot-arm"
        else
            echo "Warning: Could not activate ros2-robot-arm. Using system environment."
        fi
    fi
else
    echo "Conda not found. Using system ROS/Gazebo packages."
fi

# Use conda plugin paths when running from a conda environment.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export IGN_GAZEBO_SYSTEM_PLUGIN_PATH="$CONDA_PREFIX/lib/ign-gazebo-6/plugins${IGN_GAZEBO_SYSTEM_PLUGIN_PATH:+:$IGN_GAZEBO_SYSTEM_PLUGIN_PATH}"
    export IGN_GUI_PLUGIN_PATH="$CONDA_PREFIX/lib/ign-gui-6/plugins:$CONDA_PREFIX/lib/ign-gazebo-6/plugins/gui${IGN_GUI_PLUGIN_PATH:+:$IGN_GUI_PLUGIN_PATH}"
    export IGN_RENDERING_PLUGIN_PATH="$CONDA_PREFIX/lib/ign-rendering-6/engine-plugins${IGN_RENDERING_PLUGIN_PATH:+:$IGN_RENDERING_PLUGIN_PATH}"
    export OGRE2_RESOURCE_PATH="$CONDA_PREFIX/lib/OGRE-Next${OGRE2_RESOURCE_PATH:+:$OGRE2_RESOURCE_PATH}"
    export OGRE_RESOURCE_PATH="$CONDA_PREFIX/lib/OGRE${OGRE_RESOURCE_PATH:+:$OGRE_RESOURCE_PATH}"
fi

echo "============================================"
echo " Ignition Gazebo - Robot Arm Simulation"
echo "============================================"
echo ""
echo "Starting server (headless)..."
echo "Model: $MODEL_PATH"
echo "CLI: $SIM_CLI"
echo "Partition: $SIM_PARTITION"
echo ""
if [[ "$SIM_CLI" == "ign" ]]; then
    echo "Useful commands (in another terminal):"
    echo "  ign topic -l                    # List topics"
    echo "  ign topic -e -t /world/arm_world/pose/info  # Echo pose info"
    echo "  ign service -l                  # List services"
else
    echo "Useful commands (in another terminal):"
    echo "  gz topic -l                     # List topics"
    echo "  gz topic -e -t /world/arm_world/pose/info  # Echo pose info"
    echo "  gz service -l                   # List services"
fi
echo ""
echo "To visualize with RViz2:"
echo "  ros2 launch ros_gz_bridge parameter_bridge.launch.py"
echo "  rviz2"
echo ""
echo "Host GUI attach (experimental on macOS Docker Desktop):"
echo "  ./scripts/open_host_gazebo_gui.sh"
echo ""
echo "Press Ctrl+C to stop the server."
echo "============================================"
echo ""

"${SIM_CMD[@]}"
