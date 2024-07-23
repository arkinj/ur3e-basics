# UR3e Arm + Robotiq Gripper Basics
This project provides a Docker container for running and developing code related to the basics of using the UR3e robot arm with a Robotiq 2F85 gripper. The Docker container provides the system environment required to run the project's source code. The project source code is mounted when running the container via the provided scripts. The project can be further developed via Docker by modifying the source on the host machine executing changes in the container.

Two different Docker container options are provided that handle hosts with and without nvidia GPUs.

## Dockerfiles
The dockerfiles can be found in `catkin_ws/.docker/dockerfiles/`. The provided dockerfiles can be used to build an image that provides a development workspace for interacting with the UR3e robot arm & the Robotiq rs85 gripper.

### dockerfile-ur3e-basics
This dockerfile can be used to build an image that starts with the provided ROS Noetic docker image and builds the required robot libraries on top of it. It also adds dependencies for the Realsense cameras, python AprilTags library, and various useful packages for development & debugging.

### dockerfile-nvidia-ros
This dockerfile provides a base image that starts with the provided nvidia docker image (cuda11.1.1 & ubuntu 20.04) and builds ROS Noetic on top of it. This can be used as the base image if it is useful to access an Nvidia GPU on the host machine while inside the docker container.

## Build Docker Image & Run Container (no Nvidia)
These instructions are for building & running a docker image that does NOT provide functionality to use a host machine's nvidia GPU. These steps assume you are at the root of the repo.

### Instructions
1. Build the image. Note that this can take a few minutes when building from scratch.
```bash
cd .docker/dockerfiles
docker build -f dockerfile-ur3e-basics -t ur3e-basics .
```
The `-f` flag specifies the Dockerfile to build and `-t` specifies the name of the resulting built image.

2. Enter a container running the build image.
```bash
# Allow graphics to display from the docker container (e.g. rviz).
# This only needs to be done once per login session.
xhost +

# Start the container by running the provided script
cd /catkin_ws/ # Move to the root of the workspace
./.docker/scripts/docker-mount-run

# To enter the same running container in another terminal window, run the following.
# Note that `<container ID>` is the ID of the running container and should be tab-completable.
docker exec -it <container ID> bash
```
