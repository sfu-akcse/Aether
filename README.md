# Aether

Aether is the project

## Requirements

This project requires the miniconda and Python version 3.11

## How to setup

```bash
conda install mamba -c conda-forge

# Setup conda environment
conda create -n ros2-robot-arm python=3.11 numpy=1.26.4

conda activate ros2-robot-arm

conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging
conda config --env --remove channels defaults
conda install ros-humble-desktop-full
conda install compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep

# Restart the conda environment
conda deactivate
conda activate ros2-robot-arm

# Testing (Rviz2 will be deployed if the installation is complete)
rviz2
```

## How to run the project

```bash
# Activate the conda environment
conda activate ros2-robot-arm

# Run the script that runs Gazebo backend
./launch_gazebo.sh

# Open another terminal but not in conda environment
# Run the frontend
gz sim -g
```
