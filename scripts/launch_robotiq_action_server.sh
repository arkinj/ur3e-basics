#!/bin/bash
source /catkin_ws/devel/setup.bash
roslaunch robotiq_2f_gripper_control robotiq_action_server.launch \
  respawn:=true \
  comport:=/tmp/ttyUR
