#!/usr/bin/env python
import time
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import shape_msgs.msg
import sensor_msgs.msg
import visualization_msgs.msg
import std_msgs.msg
from geometry_msgs.msg import PoseStamped

import actionlib
from robotiq_2f_gripper_msgs.msg import CommandRobotiqGripperFeedback, CommandRobotiqGripperResult, CommandRobotiqGripperAction, CommandRobotiqGripperGoal
from robotiq_2f_gripper_control.robotiq_2f_gripper_driver import Robotiq2FingerGripperDriver as Robotiq

from math import pi, tau, dist, fabs, cos

from tf.transformations import quaternion_from_euler

from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list


def display_box(marker_publisher, marker_id_counter, ref_link, pose, dimensions, color):
  assert len(dimensions) == 3
  # setup cube / box marker type
  marker = visualization_msgs.msg.Marker()
  marker.header.stamp = rospy.Time.now()
  marker.ns = "/"
  marker.id = marker_id_counter
  marker.type = visualization_msgs.msg.Marker.CUBE
  marker.action = visualization_msgs.msg.Marker.ADD
  marker.header.frame_id = ref_link
  marker.color = color
  marker.pose = pose
  marker.scale.x = dimensions[0]
  marker.scale.y = dimensions[1]
  marker.scale.z = dimensions[2]

  # Publish
  marker_publisher.publish(marker)
  marker_id_counter += 1
  return marker_id_counter

def display_sphere(marker_publisher, marker_id_counter, ref_link, pose, radius, color):
  """ Utility function to visualize the goal pose"""
  # setup sphere marker type
  marker = visualization_msgs.msg.Marker()
  marker.header.stamp = rospy.Time.now()
  marker.ns = "/"
  marker.id = marker_id_counter
  marker.type = visualization_msgs.msg.Marker.SPHERE
  marker.action = visualization_msgs.msg.Marker.ADD
  marker.header.frame_id = ref_link
  marker.color = color
  marker.pose = pose
  marker.scale.x = radius
  marker.scale.y = radius
  marker.scale.z = radius

  # publish it!
  marker_publisher.publish(marker)
  marker_id_counter += 1
  return marker_id_counter

def remove_all_markers(marker_publisher):
  marker = visualization_msgs.msg.Marker()
  marker.header.stamp = rospy.Time.now()
  marker.ns = "/"
  marker.action = visualization_msgs.msg.Marker.DELETEALL
  marker_publisher.publish(marker)

def main():
  # Initialize a commander and node
  moveit_commander.roscpp_initialize(sys.argv)
  rospy.init_node("move_group_python_interface_tutorial", anonymous=True)

  # Instantiate the RobotCommander object
  robot = moveit_commander.RobotCommander()
  group_names = robot.get_group_names()
  print("========== Available Planning Groups: %s" % group_names)

  # Instantiate a MoveGroupCommander
  move_group_arm = moveit_commander.MoveGroupCommander("manipulator")

  # Create a trajectory display publisher
  display_trajectory_publisher = rospy.Publisher(
      "/move_group/display_planned_path",
      moveit_msgs.msg.DisplayTrajectory,
      queue_size=20
    )

  # Create a marker publisher to visualize the constraints
  marker_publisher = rospy.Publisher(
    "/visualization_marker", visualization_msgs.msg.Marker, queue_size=20
  )
  marker_id_counter = 1

  print("Pausing to allow publishers to connect to their topics")
  time.sleep(1)

  # Define some colors
  COLOR_RED = std_msgs.msg.ColorRGBA(1.0, 0.0, 0.0, 1.0)
  COLOR_GREEN = std_msgs.msg.ColorRGBA(0.0, 1.0, 0.0, 1.0)
  COLOR_TRANSLUCENT = std_msgs.msg.ColorRGBA(0.0, 0.0, 0.0, 0.5)

  eef_link_arm = move_group_arm.get_end_effector_link()
  ref_link = move_group_arm.get_pose_reference_frame()

  # Define the constraints
  pcm = moveit_msgs.msg.PositionConstraint()
  pcm.header.frame_id = ref_link
  pcm.link_name = eef_link_arm

  cbox = shape_msgs.msg.SolidPrimitive()
  cbox.type = shape_msgs.msg.SolidPrimitive.BOX
  cbox.dimensions = [1.0, 0.005, 1.0]
  pcm.constraint_region.primitives.append(cbox)

  current_pose = move_group_arm.get_current_pose()

  cbox_pose = geometry_msgs.msg.Pose()
  cbox_pose.position.x = current_pose.pose.position.x
  cbox_pose.position.y = current_pose.pose.position.y
  cbox_pose.position.z = current_pose.pose.position.z

  quat = quaternion_from_euler(pi / 2, 0, 0)
  cbox_pose.orientation.x = quat[0]
  cbox_pose.orientation.y = quat[1]
  cbox_pose.orientation.z = quat[2]
  cbox_pose.orientation.w = quat[3]
  pcm.constraint_region.primitive_poses.append(cbox_pose)

  # Display the constraints
  marker_id_counter = display_box(
    marker_publisher=marker_publisher,
    marker_id_counter=marker_id_counter,
    ref_link=ref_link,
    pose=cbox_pose,
    dimensions=cbox.dimensions,
    color=COLOR_TRANSLUCENT
  )
  input("Displaying the constraint. Press enter to continue...")

  # Wrap the constraint in a generic constraints message
  path_constraints = moveit_msgs.msg.Constraints()
  path_constraints.position_constraints.append(pcm)
  move_group_arm.clear_pose_targets()

  # Construct the planning problem
  start_state = moveit_msgs.msg.RobotState()
  start_state.joint_state = robot.get_current_state().joint_state
  
  # Display the current pose
  marker_id_counter = display_sphere(
    marker_publisher=marker_publisher,
    marker_id_counter=marker_id_counter,
    ref_link=ref_link,
    pose=move_group_arm.get_current_pose().pose,
    radius=0.05,
    color=COLOR_RED
  )
  input("Displaying the current pose. Press enter to continue...")

  # Create a sequence of waypoints for Cartesian path planning
  waypoints = []
  wpose = move_group_arm.get_current_pose().pose
  wpose.position.x -= 0.2
  wpose.position.y -= 0.4
  waypoints.append(copy.deepcopy(wpose))

  wpose.position.x += 0.2
  waypoints.append(copy.deepcopy(wpose))

  wpose.position.y += 0.4
  waypoints.append(copy.deepcopy(wpose))

  wpose.position.x -= 0.2
  waypoints.append(copy.deepcopy(wpose))
  

  # Display the waypoints
  for waypoint in waypoints:
    marker_id_counter = display_sphere(
      marker_publisher=marker_publisher,
      marker_id_counter=marker_id_counter,
      ref_link=ref_link,
      pose=waypoint,
      radius=0.05,
      color=COLOR_GREEN
    )
  input("Displaying the waypoints. Press enter to plan & execute.")

  # Solve
  move_group_arm.set_start_state(start_state)
  (plan, fraction) = move_group_arm.compute_cartesian_path(
    waypoints=waypoints,
    eef_step=0.01,
    jump_threshold=0.0,
    path_constraints=path_constraints
  )
  #move_group_arm.set_pose_target(goal_pose)
  #move_group_arm.set_path_constraints(path_constraints)

  # The move_group node should autmatically visualize the solution in Rviz if a path is found.
  #err, plan, _, _ = move_group_arm.plan()
  # Execute the plan
  move_group_arm.execute(plan)

  # Clear the path constraints for our next experiment
  #move_group_arm.clear_path_constraints()
  return

if __name__ == "__main__":
  main()
