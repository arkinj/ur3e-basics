# UR3e Arm + Robotiq Gripper Basics
This project provides a Docker container for running and developing code related to the basics of using the UR3e robot arm with a Robotiq 2F85 gripper. The Docker container provides the system environment required to run the project's source code. The project source code is mounted when running the container via the provided scripts. The project can be further developed via Docker by modifying the source on the host machine executing changes in the container.

Two different Docker container options are provided that handle hosts with and without nvidia GPUs.

## Dockerfiles
The dockerfiles can be found in `catkin_ws/.docker/dockerfiles/`. The provided dockerfiles can be used to build an image that provides a development workspace for interacting with the UR3e robot arm & the Robotiq rs85 gripper.

### dockerfile-ur3e-basics
This dockerfile can be used to build an image that starts with the provided ROS Noetic docker image and builds the required robot libraries on top of it. It also adds dependencies for the Realsense cameras, python AprilTags library, and various useful packages for development & debugging.

## Build Docker Image & Run Container
These instructions are for building & running the docker image. These steps assume you are at the root of the repo.

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

# Start the container by running the provided script. This assumes you are at the root of the repo.
./.docker/scripts/docker-mount-run

# To enter the same running container in another terminal window, run the following.
# Note that `<container ID>` is the ID of the running container and should be tab-completable.
docker exec -it <container ID> bash
```

## Commit Docker Image Changes
It can be useful to commit changes you have made to the docker image while running in a container. For example, you can build the packages that get mounted into `/src/` and commit the built state as an image.

```bash
# Get the Container ID of the running container
docker container ls

# Commit the changes (you can overwrite the existing image or create a new one with a new name)
docker commit <container ID> <new docker image name>
```

## Bring up UR3E & Robotiq
These are the steps to go from a powered off arm + gripper to a powered on & ready-to-receive state. Remember that the emergency stop (e-stop) is located at the top of the Polyscope.

### Power on the UR3e
1. Power on the UR3e computer via the UR Polyscope (light blue touchscreen). This may take a few minutes.

2. Check that the `Installation -> Tool I/O` is controlled by the User. Select "Installation" from the top menu bar, then "Tool I/O" from the left pane.

3. Check that the RS485 URcaps is installed. Select "URCaps" from the left pane. There should be an "RS485" pane that appears below in the left pane. If it is there, it is installed.

4. Power on the UR3e via the Polyscope (button in the bottom left on the touchscreen).

  - Press "Power Off" in the bottom left of the touchscreen. This should open a new menu.
  - Confirm that the Active Payload is set and the Payload is 0.850 kg.
  - Press "On", which should begin booting the arm. It will pause at the "Robot Active" step.
  - Press "Start" to make the arm operational. You should hear some clicking as the breaks release.
  - Press "Exit" at the bottom left to return to the main menu.

### Start the docker container
These next steps assume you have followed the directions above to build the docker image.
5. In a new terminal, start the docker container using the script.
```bash
# Assumes you are in the root of the repo.
./.docker/scripts/docker-mount-run
```

### Setup the network between the host machine & the UR3e
Connect the host machine & the UR3e computer via a wired network (UR3e -> switch <- Host Machine).

6. Check the UR3e's IP address on the Polyscope

  - Go to "Settings" via the hamburger menu in the top right of the touchscreen.
  - Go to "System -> Network" via the left pane.
  - The IP address should be shown in the window.

7. Check that the host machine is on the same network using the host machine's network settings manager.

8. Confirm network connection via by pinging the UR3e computer from the host machine.

### Bringup the UR3e & the Robotiq gripper

### Power down the UR3e

