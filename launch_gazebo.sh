#!/bin/bash
# launch_gazebo.sh - Launch Ignition Gazebo on macOS
# Usage: ./launch_gazebo.sh
#
# On macOS, ign gazebo GUI has known issues with OGRE rendering (NSWindow
# threading + protobuf conflicts). This script runs the simulation server
# headlessly. Use RViz2 or `ign topic` CLI tools to inspect the simulation.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="$SCRIPT_DIR/model/robot_arm.sdf"

# Ensure we're in the ros2-robot-arm conda environment
if [[ "$CONDA_DEFAULT_ENV" != "ros2-robot-arm" ]]; then
    echo "Activating ros2-robot-arm conda environment..."
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate ros2-robot-arm
fi

# Set Ignition environment variables
export IGN_GAZEBO_SYSTEM_PLUGIN_PATH="$CONDA_PREFIX/lib/ign-gazebo-6/plugins"
export IGN_GUI_PLUGIN_PATH="$CONDA_PREFIX/lib/ign-gui-6/plugins:$CONDA_PREFIX/lib/ign-gazebo-6/plugins/gui"
export IGN_RENDERING_PLUGIN_PATH="$CONDA_PREFIX/lib/ign-rendering-6/engine-plugins"
export OGRE2_RESOURCE_PATH="$CONDA_PREFIX/lib/OGRE-Next"
export OGRE_RESOURCE_PATH="$CONDA_PREFIX/lib/OGRE"

echo "============================================"
echo " Ignition Gazebo - Robot Arm Simulation"
echo "============================================"
echo ""
echo "Starting server (headless)..."
echo "Model: $MODEL_PATH"
echo ""
echo "Useful commands (in another terminal with ros2-robot-arm activated):"
echo "  ign topic -l                    # List topics"
echo "  ign topic -e -t /world/arm_world/pose/info  # Echo pose info"
echo "  ign model --list                # List models"
echo "  ign service -l                  # List services"
echo ""
echo "To visualize with RViz2:"
echo "  ros2 launch ros_gz_bridge parameter_bridge.launch.py"
echo "  rviz2"
echo ""
echo "Press Ctrl+C to stop the server."
echo "============================================"
echo ""

gz sim -s -v 4 "$MODEL_PATH"
