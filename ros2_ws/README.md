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

## Stream ROS 2 points to the Mac host over TCP

Start the TCP receiver on the Mac host:

Build:

```bash
cd /workspace/ros2_ws
colcon build --packages-select random_xyz_stream
source install/setup.zsh
ros2 run random_xyz_stream tcp_xyz_bridge
```

```bash
cd /Users/admin/Aether
python3 scripts/tcp_xyz_receiver.py --port 5001
```

Start the ROS 2 publisher in the devcontainer:

```bash
cd /workspace/ros2_ws
source install/setup.zsh
ros2 run random_xyz_stream random_xyz_publisher
```

Start the TCP bridge in another devcontainer terminal:

```bash
cd /workspace/ros2_ws
source install/setup.zsh
TCP_XYZ_HOST=host.docker.internal TCP_XYZ_PORT=5001 ros2 run random_xyz_stream tcp_xyz_bridge
```

The bridge subscribes to `/hand_xyz` inside the container and forwards each
message as a JSON line like `{"x":1.23,"y":4.56,"z":7.89}` to the host.
