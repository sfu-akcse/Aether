# Aether

Aether is the project that makes users control the physical robot arm with laptop webcam and no additional devices. 

This project contains two sections which are software and mechanical section. 

Software section handles computer vision using OpenCV and Mediapipe, robot arm simulation using ROS2, and the socket communication with the physical robot arm.

Mechanical section handles the design and the control of the robot arm. 

All the process will be documented and published through the Docusaurus documentation system.

## How to setup

This project will use ROS2 Dev Container for the development environment but this project will require to run programs through host Operating Systems since the project requires to use Computer Vision through laptop webcam.

## How to run the project

### 1) On host OS, start webcam streamer

In order to run the camera through the host operating system, install this dependency on this host operating system. 
```bash
python3 -m pip install opencv-python
```

Next, run either scripts on the host operating system.
```bash
./scripts/run_webcam_pipeline.sh host --port 8080
```
or 
```bash
python3 scripts/host_webcam_stream.py --port 8080
```

Wrapper defaults are tuned for lower latency: `640x480 @ 20fps`.
Override if needed, for example:

```bash
./scripts/run_webcam_pipeline.sh host --port 8080 --width 960 --height 540 --fps 24
```

The host script defaults to `--camera-index auto` and probes common indices.
If needed, set it explicitly, for example:

```bash
./scripts/run_webcam_pipeline.sh host --port 8080 --camera-index 1
```

### 2) Inside devcontainer, point CV node to host stream

Open up the Dev Container and run this script
```bash
./scripts/run_webcam_pipeline.sh cv --port 8080
```
or
```bash
export CAMERA_SOURCE=http://host.docker.internal:8080/video.mjpg
python3 src/main.py
```

## Webcam on Linux with devcontainer

On Linux, `CAMERA_SOURCE=0` only works if the camera device is passed into the
container. If no `/dev/video*` device is mounted, OpenCV reports:
`can't open camera by index`.

### Option 1) Direct camera passthrough (`CAMERA_SOURCE=0`)

1. Confirm your host camera exists:

```bash
ls /dev/video*
```

2. Add Linux camera run args in `.devcontainer/devcontainer.json`:

```jsonc
"runArgs": [
	"--ipc=host",
	"--device=/dev/video0:/dev/video0",
	"--group-add=video"
]
```

3. Rebuild/reopen the devcontainer.
4. Verify camera visibility inside container:

```bash
ls /dev/video*
```

5. Run CV app:

```bash
export CAMERA_SOURCE=0
python3 src/main.py
```

### Option 2) Host stream to container (no `/dev/video` passthrough)

1. Add host gateway mapping in `.devcontainer/devcontainer.json`:

```jsonc
"runArgs": [
	"--ipc=host",
	"--add-host=host.docker.internal:host-gateway"
]
```

2. Rebuild/reopen the devcontainer.
3. On host, start streamer:

```bash
python3 scripts/host_webcam_stream.py --port 8080
```

4. In container, run:

```bash
export CAMERA_SOURCE=http://host.docker.internal:8080/video.mjpg
python3 src/main.py
```

### How to run the Robot Arm Simulation

Run this script inside the Dev Container.
```bash
# Run the Gazebo backend server using this shell script
./launch_gazebo.sh

# Run the Gazebo GUI through browser using noVNC
./scripts/run_gazebo_gui.sh
```