# ROS 2 Workspace

This workspace contains ROS 2 packages for Aether experiments.

## Random XYZ stream

Build:

```bash
cd /workspace/ros2_ws
colcon build
source install/setup.zsh
```

Run publisher:

```bash
cd /workspace/ros2_ws
source install/setup.zsh
ros2 run random_xyz_stream random_xyz_publisher
```

Run subscriber in another terminal:

```bash
cd /workspace/ros2_ws
source install/setup.zsh
ros2 run random_xyz_stream random_xyz_subscriber
```
