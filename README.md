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

6. If needed, build the mounted packages & source the workspace (from inside the docker container).
```bash
catkin build
source devel/setup.bash
```

### Setup the network between the host machine & the UR3e
Connect the host machine & the UR3e computer via a wired network (UR3e -> switch <- Host Machine).

7. Check the UR3e's IP address on the Polyscope:

  - Go to "Settings" via the hamburger menu in the top right of the touchscreen.
  - Go to "System -> Network" via the left pane.
  - The IP address should be shown in the window.

8. Check that the host machine is on the same network using the host machine's network settings manager.

9. Confirm network connection via by pinging the UR3e computer from the host machine.

### Bringup the UR3e & the Robotiq gripper
Launch the UR3e and Robotiq gripper & allow external control to the robot.

10. Prepare to run the `external_control` program on the UR3e polyscope. Press "Run" in the top left of the touchscreen. You should see "external_control" in the "Program" box and the status should be "Stopped".

11. Bringup the UR3e & Robotiq gripper.
   - The `launch_ur_driver_tool_use.sh` script runs the `bringup_ur3e_robotiq.launch` file provided by the `mit_robot` package.
  - The Robotiq gripper should have a blue light and open-close its gripper. The terminal output should say "Gripper on port /tmp/ttyUR Activated" then "Robotiq server started".
  - A bit further up, the terminal should also read "Robot mode is now RUNNING", then "Robot's safety mode is now NORMAL".
```bash
cd /catkin_ws/
./scripts/launch_ur_driver_tool_use.sh
```

12. Enable external control.
  - On the polyscope, press the large play button to enable external control. The terminal should read the following:
```
[ INFO] [1721836055.876387186]: Robot requested program
[ INFO] [1721836055.876551560]: Sent program to robot
[ INFO] [1721836056.224020343]: Robot connected to reverse interface. Ready to receive control commands.
```

  - Note that sometimes the Robotiq server get interrupted by this step. The light may switch from blue to red, but it should shortly switch back to blue. If this happens, the following terminal output is expected:
```
Modbus Error: [Input/Output] Modbus Error: [Invalid Message] Incomplete message received, expected at least 2 bytes (0 received)
[FATAL] [1721836061.219493]: Failed to contact gripper on port: /tmp/ttyUR

```

### Power down the UR3e
Terminate any processes that are running.

13. Power off the UR3e

  - Press "Normal" (formerly "Power Off") in the bottom left of the touchscreen.
  - Press "OFF" to power off the UR3e.
  - Press "Exit" to leave the menu.

14. Shut down the UR3e computer

  - Press "Shutdown Robot" in the hamburger menu in the top right of the touchscreen.



## Run a Simple Demo
These are the steps to go from a powered off arm + gripper to running a simple demo that moves the arm and opens the gripper. Many of the steps are the same as the above and refer to them for concision.

1. Power on the UR3e (see above section).

2. Start the docker container (see above section).

3. Setup the network between the host machine & the UR3e (see above section).

4. Bringup the UR3e & the Robotiq gripper (see above section).

5. Launch the MoveIt! planning interface.
    - open a new terminal and enter the docker container: ```docker exec -it <container ID> bash```
    - source the workspace: ```source devel/setup.bash```
    - start the process: ```roslaunch ur3e_robotiq_moveit_config move_group.launch```

6. Run the demo.
    - open a new terminal and enter the docker container: ```docker exec -it <container ID> bash```
    - source the workspace: ```source devel/setup.bash```
    - Start the demo: ```./scripts/ur3e_robotiq_demo.py```
