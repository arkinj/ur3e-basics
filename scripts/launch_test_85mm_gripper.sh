#!/bin/bash
source /catkin_ws/devel/setup.bash
roslaunch robotiq_2f_gripper_control test_85mm_gripper.launch \
  respawn:=true \
  comport:=/tmp/ttyUR
