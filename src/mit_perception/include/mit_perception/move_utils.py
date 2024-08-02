#!/usr/bin/env python
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped, Pose, Point
import numpy as np


import time
import actionlib
from robotiq_2f_gripper_msgs.msg import CommandRobotiqGripperFeedback, CommandRobotiqGripperResult, CommandRobotiqGripperAction, CommandRobotiqGripperGoal
from robotiq_2f_gripper_control.robotiq_2f_gripper_driver import Robotiq2FingerGripperDriver as Robotiq

from math import pi, tau, dist, fabs, cos, sin, ceil

from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

from mit_perception.transform_utils import env2arm, arm2env

def close_hand():
  input("Press enter to close the hand...")
  action_name = rospy.get_param('~action_name', 'command_robotiq_action')
  robotiq_client = actionlib.SimpleActionClient(action_name, CommandRobotiqGripperAction)
  # Wait until grippers are ready to take command
  robotiq_client.wait_for_server()

  rospy.logwarn("Client test: Starting sending goals")
  ## Manually set all the parameters of the gripper goal state.
  ######################################################################################

  goal = CommandRobotiqGripperGoal()
  goal.emergency_release = False
  goal.stop = False
  goal.position = 0.00
  goal.speed = 0.1
  goal.force = 5.0

  # Sends the goal to the gripper.
  robotiq_client.send_goal(goal)
  # Block processing thread until gripper movement is finished, comment if waiting is not necesary.
  robotiq_client.wait_for_result()
  # Prints out the result of executing the action
  return robotiq_client.get_result()  # A FibonacciResult


def open_hand():
  input("Press enter to open the hand...")
  action_name = rospy.get_param('~action_name', 'command_robotiq_action')
  robotiq_client = actionlib.SimpleActionClient(action_name, CommandRobotiqGripperAction)
  # Wait until grippers are ready to take command
  robotiq_client.wait_for_server()

  rospy.logwarn("Client test: Starting sending goals")
  ## Manually set all the parameters of the gripper goal state.
  ######################################################################################

  goal = CommandRobotiqGripperGoal()
  goal.emergency_release = False
  goal.stop = False
  goal.position = 1.00
  goal.speed = 0.1
  goal.force = 5.0

  # Sends the goal to the gripper.
  robotiq_client.send_goal(goal)
  # Block processing thread until gripper movement is finished, comment if waiting is not necesary.
  robotiq_client.wait_for_result()
  # Prints out the result of executing the action
  return robotiq_client.get_result()  # A FibonacciResult



def go_grasp():
  input("Press enter to plan to move to grasp...")
  joint_goal = move_group.get_current_joint_values()
  joint_goal[0] = 1.80348
  joint_goal[1] = -1.106003
  joint_goal[2] = 1.55695
  joint_goal[3] = -2.08087
  joint_goal[4] = -1.608721
  joint_goal[5] = 0.306845

  input("Press enter to execute...")
  move_group.go(joint_goal, wait=True)
  move_group.stop()

def go_home():
  input("Press enter to plan to move home...")
  joint_goal = move_group.get_current_joint_values()
  joint_goal[0] = 1.665353775024414
  joint_goal[1] = -1.952281137506002
  joint_goal[2] = 1.808349911366598
  joint_goal[3] = -2.932059427300924
  joint_goal[4] = -1.599253002797262
  joint_goal[5] = 0.0709202

  input("Press enter to execute...")
  move_group.go(joint_goal, wait=True)
  move_group.stop()


def generate_circle_poses(pose_center, r, n, frame_id=None):
  x = pose_center.pose.position.x
  y = pose_center.pose.position.y
  z = pose_center.pose.position.z
  theta = pi / (3*n)
  poses = []
  
  for i in range(n):
    p = copy.deepcopy(pose_center)
    if frame_id is not None:
      p.header.frame_id = frame_id
    p.pose.position.x = x + r * cos(theta * i)
    p.pose.position.y = y + r * sin(theta * i)
    p.pose.position.z = z
    # p.pose.orientation = pose_center.pose.orientation
    print(f"{i :2d}: ({p.pose.position.x :.4f}, {p.pose.position.y :.4f}, {p.pose.position.z :.4f})")
    poses.append(p)
  # print(poses)
  return poses



# def move_to_position(move_group_arm, x=None, y=None, z=None):
#   pose = move_group_arm.get_current_pose()
#   if 


############################################
###   BEGIN RELEVANT CODE TO STEP_REAL   ###
############################################

def move_to_pose(
  move_group_arm, 
  pose_goal, 
  manual=True, 
  dry_run=False, 
  confirm_twice=False, 
  idx=None, 
  quiet=False):
  """
  move arm to goal pose(s)

  Args:
    move_group_arm: arm
    pose_goal: pose or list of poses to move to
    manual: if True, asks for confirmation before plan/move
    dry_run: if True, disables actual movement
    confirm_twice: if True, only asks confirmation for move
    idx: index for debugging purposes if repeatedly calling
    quiet: suppress all non-essential logging
  
  Returns:
    ok: whether or not plan and move was successful
    move_time: how long it took to move (or None if failure)
  """

  ok = False
  # pose goal can be list of poses, but if not, make it one first
  if not isinstance(pose_goal, list):
    # pose_goal = [p.pose for p in pose_goal]
    pose_goal = [pose_goal]
  
  if not quiet:
    # print some useful data
    positions = [pose.position for pose in pose_goal]
    if idx is not None:
      for i, p in enumerate(positions):
        print(f"\ngoal pose {idx :2d} step {i}: ({p.x :.4f}, {p.y :.4f}, {p.z :.4f})")
    else:
      for p in positions:
        print(f"\ngoal pose: ({p.x :.4f}, {p.y :.4f}, {p.z :.4f})")

  if manual and confirm_twice:
    input("Press enter to plan to goal pose...")
  elif not quiet:
    print("planning goal pose...")

  if not dry_run:
    # try to plan poses, may be unsuccessful
    try:
      # print(pose_goal)
      move_group_arm.set_pose_targets(pose_goal)
    except:
      print("failed to plan :(")
      # failed to plan, notify caller
      return ok, None
  elif not quiet:
    print("dry run, we won't actually move arm :)")

  # Move to the pose goal
  if manual:
    input("Press enter to move to the goal pose...")
  elif not quiet:
    print("moving to the goal pose...")

  if not dry_run:
    s = time.time()
    # try to move the arm
    ok = move_group_arm.go(wait=True)
    # Stop and Clear after execution
    move_group_arm.stop()
    move_group_arm.clear_pose_targets()
    return ok, time.time() - s
  elif not quiet:
    print("dry run, we won't actually move arm :)")
    return ok, None


def move_to_poses(
  move_group_arm, 
  pose_goals,
  manual=True, 
  dry_run=False, 
  confirm_twice=False,
  quiet=False, 
  stride=1):
  """
  move arm through sequence of poses in steps, 
  by partitioning the sequence, 
  and calling move_to_pose on each part

  Args:
    move_group_arm: arm
    pose_goals: list of poses to move through
    manual: if True, asks for confirmation before plan/move
    dry_run: if True, disables actual movement
    confirm_twice: if True, only asks confirmation for move
    quiet: suppress all non-essential logging
    stride: how many poses to plan at a time
  
  Returns:
    ok: whether or not all moves were successful
    move_time: how long each move took (or None if failure)
  """

  # record move times in listposition =
  move_times = []
  for i in range(0, len(pose_goals), stride):
    # choose j so that i:j is gives us stride poses 
    j = min(i+stride, len(pose_goals))
    # try to move to pose and record move time
    ok, move_time = move_to_pose(
      move_group_arm, pose_goals[i:j], 
      manual, dry_run, confirm_twice, i, quiet)
    if ok:
      move_times.append(move_time)
    else:
      # failed to plan or move, notify caller
      return ok, None
  return True, move_times


def pose_to_xy(pose):
  return np.array(
      [pose.position.x, pose.position.y],
      dtype=float)
  

def get_current_pose_as_xy(move_group_arm):
  pose = move_group_arm.get_current_pose()
  return pose_to_xy(pose.pose)


def perform_action_env(move_group_arm, action_env, manual=False):
  """
  performs action on real arm based on env_T positions
  (see usage in env_T.py, step_real function)

  Args:
    move_group_arm: arm
    action_env: numpy array of shape (n,2) of xy positions in env frame

  Returns:
    position: xy position of the arm after move, in env frame
  """
  # hopefully this works with not just 1x2, should be able to broadcast
  action_arm = env2arm(action_env)
  # print(action_arm.shape)

  # get current pose, also record it naive velocity estimate
  old_pose = move_group_arm.get_current_pose().pose
  old_position = pose_to_xy(old_pose)

  # move to goal pose based on action, do any of these need flip?
  pose_goals = [None] * len(action_arm)
  # gemove_to_posesnerate poses by modifying x and y of current pose copy
  for i in range(len(action_arm)):
    # copy current pose but change x and y for action
    pose = copy.deepcopy(old_pose)
    pose.position.x = -action_arm[i,0]
    pose.position.y = action_arm[i,1]
    pose_goals[i] = pose

  # # attempt to plan and move to all of these together
  # ok, move_times = move_to_poses(
  #   move_group_arm, pose_goals, stride=len(pose_goals))

  # below is more direct, assuming single movement, but same
  #  thing
  # print(pose_goals)
  ok, move_time = move_to_pose(
    move_group_arm, pose_goals, manual=manual)
  # move_times = [move_time]

  # TODO: what to do when not ok?
  # currently doesn't handle failure to plan/move
  # next part will have errors in that case...

  # update position, does velocity actually matter?
  new_position = get_current_pose_as_xy(move_group_arm)

  # old_pos_env = arm2env(old_position)
  new_pos_env = arm2env(new_position)

  # velocity = (new_pos_env - old_pos_env) / sum(move_times)
  return new_pos_env

############################################
###    END RELEVANT CODE TO STEP_REAL    ###
############################################





def setup():
  # Initialize a commander and node
  moveit_commander.roscpp_initialize(sys.argv)
  rospy.init_node("move_group_python_interface_tutorial", anonymous=True)

  # Instantiate the RobotCommander object
  robot = moveit_commander.RobotCommander()
  group_names = robot.get_group_names()
  print("========== Available Planning Groups: %s" % group_names)

  # Instantiate a MoveGroupCommander
  #group_name_arm = "manipulator"
  move_group_arm = moveit_commander.MoveGroupCommander("ur3e_arm")
  move_group_hand = moveit_commander.MoveGroupCommander("robotiq_2f85")

  # Create a trajectory display publisher
  display_trajectory_publisher = rospy.Publisher(
      "/move_group/display_planned_path",
      moveit_msgs.msg.DisplayTrajectory,
      queue_size=20
    )

  # Print various info
  print("========== Printing robot.get_current_state()")
  print(robot.get_current_state())
  print("")

  planning_frame_arm = move_group_arm.get_planning_frame()
  print("========== Arm Planning Frame: %s" % planning_frame_arm)

  eef_link_arm = move_group_arm.get_end_effector_link()
  print("========== Arm End effector link: %s" % eef_link_arm)


  print("========== Printing move_group_arm.get_current_pose()")
  print(move_group_arm.get_current_pose())

  # Instantiate a PlanningSceneInterface object
  input("Press enter to add planning scene object")
  scene = moveit_commander.PlanningSceneInterface()
  # Add the table
  table_pose = PoseStamped()
  table_pose.header.frame_id = robot.get_planning_frame()
  table_pose.pose.position.x = 0
  table_pose.pose.position.y = 0
  table_pose.pose.position.z = -0.01
  scene.add_box("table", table_pose, (10,10,0.01))
  # Add a wall behind
  backwall_pose = PoseStamped()
  backwall_pose.header.frame_id = robot.get_planning_frame()
  backwall_pose.pose.position.x = 0
  backwall_pose.pose.position.y = -0.5
  backwall_pose.pose.position.z = 0
  scene.add_box("backwall", backwall_pose, (1,0.01,1))

  return move_group_arm, move_group_hand

def move_to_home_pose(move_group_arm, manual=True):
  if manual:
    input("Press enter to move to \"home\"...")
  else:
    print("moving to home...")
  s = time.time()
  move_group_arm.set_named_target("home")
  success = move_group_arm.go(wait=True)
  move_group_arm.stop()
  move_group_arm.clear_pose_targets()
  print('Time taken to go home: ', time.time() - s)
 
  pose_goal = move_group_arm.get_current_pose()
  pose_goal.pose.position.x = -0.1688111302792703
  pose_goal.pose.position.y = 0.35195621031298655
  pose_goal.pose.position.z = 0.3196369615044568

def close_gripper(move_group_hand, manual=True):
  # Close the gripper
  if manual:
    input("Press enter to close the gripper")
  else:
    print("closing the gripper...")
  move_group_hand.set_named_target("close")
  success = move_group_hand.go(wait=True)
  move_group_hand.stop()
  move_group_hand.clear_pose_targets()

def open_gripper(move_group_hand, manual=True):
  # Open the gripper
  if manual:
    input("Press enter to open the gripper")
  else:
    print("opening the gripper...")
  move_group_hand.set_named_target("open")
  success = move_group_hand.go(wait=True)
  move_group_hand.stop()
  move_group_hand.clear_pose_targets()
  

def main():
#   # Initialize a commander and node
#   moveit_commander.roscpp_initialize(sys.argv)
#   rospy.init_node("move_group_python_interface_tutorial", anonymous=True)

#   # Instantiate the RobotCommander object
#   robot = moveit_commander.RobotCommander()
#   group_names = robot.get_group_names()
#   print("========== Available Planning Groups: %s" % group_names)

#   # Instantiate a MoveGroupCommander
#   #group_name_arm = "manipulator"
#   move_group_arm = moveit_commander.MoveGroupCommander("ur3e_arm")
#   move_group_hand = moveit_commander.MoveGroupCommander("robotiq_2f85")

#   # Create a trajectory display publisher
#   display_trajectory_publisher = rospy.Publisher(
#       "/move_group/display_planned_path",
#       moveit_msgs.msg.DisplayTrajectory,
#       queue_size=20
#     )

#   # Print various info
#   print("========== Printing robot.get_current_state()")
#   print(robot.get_current_state())
#   print("")

#   planning_frame_arm = move_group_arm.get_planning_frame()
#   print("========== Arm Planning Frame: %s" % planning_frame_arm)

#   eef_link_arm = move_group_arm.get_end_effector_link()
#   print("========== Arm End effector link: %s" % eef_link_arm)


#   print("========== Printing move_group_arm.get_current_pose()")
#   print(move_group_arm.get_current_pose())

#   # Instantiate a PlanningSceneInterface object
#   input("Press enter to add planning scene object")
#   scene = moveit_commander.PlanningSceneInterface()
#   # Add the table
#   table_pose = PoseStamped()
#   table_pose.header.frame_id = robot.get_planning_frame()
#   table_pose.pose.position.x = 0
#   table_pose.pose.position.y = 0
#   table_pose.pose.position.z = -0.01
#   scene.add_box("table", table_pose, (10,10,0.01))
#   # Add a wall behind
#   backwall_pose = PoseStamped()
#   backwall_pose.header.frame_id = robot.get_planning_frame()
#   backwall_pose.pose.position.x = 0
#   backwall_pose.pose.position.y = -0.5
#   backwall_pose.pose.position.z = 0
#   scene.add_box("backwall", backwall_pose, (1,0.01,1))
  move_group_arm, move_group_hand = setup()
  """
  # Close the gripper
  input("Press enter to close the gripper")
  move_group_hand.set_named_target("close")
  success = move_group_hand.go(wait=True)
  move_group_hand.stop()
  move_group_hand.clear_pose_targets()
  """
  # Go to the home pose
  input("Press enter to move to \"home\"...")
  s = time.time()
  move_group_arm.set_named_target("home")
  success = move_group_arm.go(wait=True)
  move_group_arm.stop()
  move_group_arm.clear_pose_targets()
  print('Time taken to go home: ', time.time() - s)
 
  pose_goal = move_group_arm.get_current_pose()
  pose_goal.pose.position.x = -0.1688111302792703
  pose_goal.pose.position.y = 0.35195621031298655
  pose_goal.pose.position.z = 0.3196369615044568

  #Goal-2
  pose_goal = move_group_arm.get_current_pose()
  pose_goal.pose.position.x = 0.0967145
  pose_goal.pose.position.y = 0.28001118
  pose_goal.pose.position.z = 0.3196369615044568

  move_to_poses(move_group_arm, [pose_goal])

  #circle_poses = generate_circle_poses(pose_goal, 0.1, 10)
  #move_to_poses(move_group_arm, circle_poses, dry_run=False, manual=False)

  

  """
  circle_poses = generate_circle_poses(pose_goal, 0.01, 1)
  move_to_poses(move_group_arm, circle_poses, dry_run=False)
  # Plan to a pose goal
  # -190 mm is the safe pose. +x is to the left of the manipulator in table plane, +y is to the front of the manipulator in table plane, +z is upwards
  #input("Press enter to plan to goal pose...")
  
  #circle_poses = generate_circle_poses(pose_goal, 0.02, 1)
  #move_to_poses(move_group_arm, circle_poses, dry_run=True)

  
  
  move_group_arm.set_pose_target(pose_goal)

  # Move to the pose goal
  input("Press enter to move to the goal pose...")
  success = move_group_arm.go(wait=True) 
  # Stop and Clear after execution
  move_group_arm.stop()
  move_group_arm.clear_pose_targets()

  
  # Plan to a pose goalfor z-calibration
  input("Press enter to plan to goal pose...")
  pose_goal = move_group_arm.get_current_pose()
  pose_goal.pose.position.x = -0.1688111302792703
  pose_goal.pose.position.y = 0.3195621031298655
  pose_goal.pose.position.z = 0.2596369615044568
  move_group_arm.set_pose_target(pose_goal)

  # Move to the pose goal
  input("Press enter to move to the goal pose...")
  success = move_group_arm.go(wait=True) 
  # Stop and Clear after execution
  move_group_arm.stop()
  move_group_arm.clear_pose_targets()

  # Plan to a pose-2 goal
  #input("Press enter to plan to goal pose...")
  pose_goal = move_group_arm.get_current_pose()
  pose_goal.pose.position.x = -0.1888111302792703
  pose_goal.pose.position.y = 0.39195621031298655
  pose_goal.pose.position.z = 0.3596369615044568
  start_time = time.time()
  move_group_arm.set_pose_target(pose_goal)
  plan_time = time.time()
  print("Plan time:",plan_time-start_time)

  # Move to the pose goal
  #input("Press enter to move to the goal pose...")
  start_time = time.time()
  success = move_group_arm.go(wait=True) 
  move_time = time.time()
  # Stop and Clear after execution 
  move_group_arm.stop()
  move_group_arm.clear_pose_targets()
  stop_time = time.time()
  print("Move time:",move_time-start_time)
  print("Stop time:",stop_time-move_time)

  # Plan to a pose-3 goal
  #input("Press enter to plan to goal pose...")
  pose_goal = move_group_arm.get_current_pose()
  pose_goal.pose.position.x = -0.1888111302792703
  pose_goal.pose.position.y = 0.35195621031298655
  pose_goal.pose.position.z = 0.3596369615044568
  start_time = time.time()
  move_group_arm.set_pose_target(pose_goal)
  plan_time = time.time()
  print("Plan time:",plan_time-start_time)

  # Move to the pose goal
  #input("Press enter to move to the goal pose...")
  start_time = time.time()
  success = move_group_arm.go(wait=True) 
  move_time = time.time()
  # Stop and Clear after execution 
  move_group_arm.stop()
  move_group_arm.clear_pose_targets()
  stop_time = time.time()
  print("Move time:",move_time-start_time)
  print("Stop time:",stop_time-move_time)


  # Plan to a pose-4 goal
  #input("Press enter to plan to goal pose...")
  pose_goal = move_group_arm.get_current_pose()
  pose_goal.pose.position.x = -0.1988111302792703
  pose_goal.pose.position.y = 0.35195621031298655
  pose_goal.pose.position.z = 0.3596369615044568
  start_time = time.time()
  move_group_arm.set_pose_target(pose_goal)
  plan_time = time.time()
  print("Plan time:",plan_time-start_time)

  # Move to the pose goal
  #input("Press enter to move to the goal pose...")
  start_time = time.time()
  success = move_group_arm.go(wait=True) 
  move_time = time.time()
  # Stop and Clear after execution 
  move_group_arm.stop()
  move_group_arm.clear_pose_targets()
  stop_time = time.time()
  print("Move time:",move_time-start_time)
  print("Stop time:",stop_time-move_time)


  # Plan to a pose-5 goal
  #input("Press enter to plan to goal pose...")
  pose_goal = move_group_arm.get_current_pose()
  pose_goal.pose.position.x = -0.1788111302792703
  pose_goal.pose.position.y = 0.35195621031298655
  pose_goal.pose.position.z = 0.3596369615044568
  start_time = time.time()
  move_group_arm.set_pose_target(pose_goal)
  plan_time = time.time()
  print("Plan time:",plan_time-start_time)

  # Move to the pose goal
  #input("Press enter to move to the goal pose...")
  start_time = time.time()
  success = move_group_arm.go(wait=True) 
  move_time = time.time()
  # Stop and Clear after execution 
  move_group_arm.stop()
  move_group_arm.clear_pose_targets()
  stop_time = time.time()
  print("Move time:",move_time-start_time)
  print("Stop time:",stop_time-move_time)


  # Plan to a pose-6 goal
  input("Press enter to plan to goal pose...")
  pose_goal = move_group_arm.get_current_pose()
  pose_goal.pose.position.x = -0.1688111302792703
  pose_goal.pose.position.y = 0.39195621031298655
  pose_goal.pose.position.z = 0.3596369615044568
  start_time = time.time()
  move_group_arm.set_pose_target(pose_goal)
  plan_time = time.time()
  print("Plan time:",plan_time-start_time)

  # Move to the pose goal
  input("Press enter to move to the goal pose...")
  start_time = time.time()
  success = move_group_arm.go(wait=True) 
  move_time = time.time()
  # Stop and Clear after execution 
  move_group_arm.stop()
  move_group_arm.clear_pose_targets()
  stop_time = time.time()
  print("Move time:",move_time-start_time)
  print("Stop time:",stop_time-move_time)

  
  # Close the gripper
  input("Press enter to close the gripper")
  move_group_hand.set_named_target("close")
  success = move_group_hand.go(wait=True)
  move_group_hand.stop()
  move_group_hand.clear_pose_targets()
  # Open the gripper
  input("Press enter to open the gripper")
  move_group_hand.set_named_target("open")
  success = move_group_hand.go(wait=True)
  move_group_hand.stop()
  move_group_hand.clear_pose_targets()
  # Close the gripper
  input("Press enter to close the gripper")
  move_group_hand.set_named_target("close")
  success = move_group_hand.go(wait=True)
  move_group_hand.stop()
  move_group_hand.clear_pose_targets()
  # Open the gripper
  input("Press enter to open the gripper")
  move_group_hand.set_named_target("open")
  success = move_group_hand.go(wait=True)
  move_group_hand.stop()
  move_group_hand.clear_pose_targets()
  """

  #input("Press enter to close the gripper via robotiq")
  #close_hand()
  #input("Press enter to open the gripper via robotiq")
  #open_hand()
  #input("Press enter to close the gripper via robotiq")
  #close_hand()
  #input("Press enter to open the gripper via robotiq")
  #open_hand()Exit
  return

def circle_output_test():
  p = PoseStamped()
  p.pose.position = Point(-0.16, 0.39, 0.31)
  poses = generate_circle_poses(p,0.01,4)
  move_to_poses(None, poses, manual=True, dry_run=True)


if __name__ == "__main__":
  main()
  #circle_output_test()

