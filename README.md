<h4 align="center">
    <br> <img src="public/icon.png" width="150">
</h4>

<h4 align="center">
    Aether: Robot Arm Controller Project
</h4>

<p align="center">
    <a href="#description">Description</a> •
    <a href="#team-members">Team</a> •
	<a href="#additional-links">Resources</a> •
    <a href="#how-to-run-the-project">Set up</a> •
	<a href="#Acknowledgement">Acknowledgement</a>
</p>

## Description

Aether is the project that makes users control the physical robot arm with a laptop webcam and no additional devices. 

This project contains three engineering parts: 
- **Software**
- **Firmware**
- **Mechatronics**

This repository is primarily developed and maintained by the Software Team of the Aether Project.

The project has three main components:
- **Hand detection** — hand location and grabbing gesture detection using OpenCV and MediaPipe
- **Robot arm simulation** — ROS2-based simulation using Gazebo, rendered in the browser via noVNC
- **Socket communication** — real-time communication between the laptop and the Raspberry Pi robot arm

Development runs inside a ROS2 Dev Container. However, some programs (such as the computer vision pipeline) must run on the host OS to access the laptop webcam.

## Team Members

- Software Team:
	- Software Team Co-lead: Tommy Oh [SFU] [@TommyOh0428](https://github.com/TommyOh0428)
	- Software Team Co-lead: Kwanghyuk Ryu [SFU] [@kwanghyukryu](https://github.com/kwanghyukryu)
	- Sofware Team Member: Jooyoung Lee [SFU] [@jylee2033](https://github.com/jylee2033)
	- Software Team Member: Yujun Song [SFU] [@Pinkbear20056](https://github.com/Pinkbear20056)
	- Software Team Member: Yoobeen Hong [UBC] [@rubyoobeen](https://github.com/rubyoobeen)
    - Software Team Member: Soyoung Lee [SFU] [@sla602](https://github.com/sla602) 
- Mechatronics Team:
	- Mechatronics Team Co-lead: Vincent Hong [SFU] [@Vincent-Elec](https://github.com/Vincent-Elec)
	- Mechatronics Team Co-lead: Sungmin Lee [SFU] [@S-ngminL-e](https://github.com/S-ngminL-e)
    - Mechatronics Team Member: Peter Yoon [SFU] [@pitta1209](https://github.com/pitta1209)
    - Mechatronics Team Member: Marco Yeung [UBC] [@yeungshmarco](https://github.com/yeungshmarco)
    - Mechatronics Team Member: Alex Sung [UBC] [@alxsno7](https://github.com/alxsno7)
    - Mechatronics Team Member: Minseok Oh [SFU] [@theise136](https://github.com/theise136)
  - Firmware Team:
    - Firmware Team Lead: Hugo Kwon [SFU] [@hugokwon5](https://github.com/hugokwon5)
    - Firmware Team Member: Joshua Kim [SFU] [@joshuakim17](https://github.com/joshuakim17)
    - Firmware Team Member: Anson Wong [SFU] [@shripsr](https://github.com/shripsr)
    - Firmware Team Member: Auston Ng [SFU] [@AcxhN](https://github.com/AcxhN)

## Additional Links:
If you would like to see other related sources to this project, check out these links:
- [Aether Documentation](https://sfu-akcse.github.io/Aether-docs/)
- [Aether Documentation Repository](https://github.com/sfu-akcse/Aether-docs)
- [Aether Firmware Repository](https://github.com/sfu-akcse/Aether-firmware)

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

Runtime logs are written to `logs/aether-system.log` and also mirrored to the console.

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

## Acknowledgement

This project initiated by [Tommy Oh](https://github.com/TommyOh0428) and [Vincent Hong](https://github.com/Vincent-Elec) of [@SFU AKCSE](https://github.com/sfu-akcse)

Joint Collaboration Project with SFU and UBC AKCSE.

This project is under the MIT License, and we welcome contributions. Check out CONTRIBUTE.md for the details. 
