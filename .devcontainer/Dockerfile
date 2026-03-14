# Use the official ROS 2 Humble Desktop image (Ubuntu 22.04)
# Desktop includes Gazebo, RViz, and other useful GUI tools
FROM osrf/ros:humble-desktop

# Avoid user interaction prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Update apt and install essential build tools and system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    git \
    wget \
    curl \
    nano \
    zsh \
    # Dependencies for OpenCV and GUI forwarding
    libgl1-mesa-glx \
    libglib2.0-0 \
    x11-apps \
    # Fully containerized GUI stack (Xvfb + noVNC)
    xvfb \
    x11vnc \
    novnc \
    websockify \
    x11-utils \
    openbox \
    # Gazebo Sim (Ignition Fortress) + ROS 2 bridge
    ignition-fortress \
    ros-humble-ros-gz \
    # ROS 2 hardware packages
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for Computer Vision and ML
# (MediaPipe is great for out-of-the-box hand tracking)
RUN pip3 install --no-cache-dir \
    opencv-python \
    mediapipe \
    "numpy<2" \
    scipy

# Install Oh My Zsh unattended
RUN sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Source the ROS 2 setup script automatically for bash and zsh terminals
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc \
    && echo "source /opt/ros/humble/setup.zsh" >> ~/.zshrc

# Set zsh as default shell for root
RUN chsh -s /bin/zsh

# Set the default working directory
WORKDIR /workspace