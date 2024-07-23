#!/bin/bash
source /catkin_ws/devel/setup.bash
roslaunch mit_robot bringup_ur3e_robotiq.launch \
  robot_ip:=172.16.0.1 \
  kinematics_config:=/catkin_ws/src/mit_robot/config/ur3e_calibration.yaml \
  use_tool_communication:=true \
  tool_voltage:=24 \
  tool_parity:=0 \
  tool_baud_rate:=115200 \
  tool_stop_bits:=1 \
  tool_rx_idle_chars:=1.5 \
  tool_tx_idle_chars:=3.5 \
  tool_device_name:=/tmp/ttyUR
