"""
Launch file for the robot arm simulation.

Starts:
  1. Ignition Gazebo server (headless, macOS compatible)
  2. robot_state_publisher (publishes URDF + TF transforms)
  3. joint_state_publisher (publishes default joint states for RViz display)
  4. RViz2 for visualization

Usage:
  conda activate ros2-robot-arm
  ros2 launch launch/robot_arm.launch.py
"""

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    # Paths
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sdf_file = os.path.join(pkg_dir, 'model', 'robot_arm.sdf')
    urdf_file = os.path.join(pkg_dir, 'model', 'robot_arm.urdf')
    rviz_config = os.path.join(pkg_dir, 'config', 'robot_arm.rviz')

    # Read URDF for robot_state_publisher
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    conda_prefix = os.environ.get('CONDA_PREFIX', '')

    # Environment variables for Ignition Gazebo
    ign_env = {
        'IGN_GAZEBO_SYSTEM_PLUGIN_PATH': os.path.join(conda_prefix, 'lib', 'ign-gazebo-6', 'plugins'),
    }

    return LaunchDescription([
        # 1. Start Ignition Gazebo server (headless - required on macOS)
        ExecuteProcess(
            cmd=['ign', 'gazebo', '-s', '-v', '4', sdf_file],
            additional_env=ign_env,
            output='screen',
        ),

        # 2. Robot state publisher - publishes /robot_description and TF
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
        ),

        # 3. Joint state publisher - publishes /joint_states so TF and
        #    RobotModel display work. Uses GUI sliders to move joints.
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen',
        ),

        # 4. RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
        ),
    ])
