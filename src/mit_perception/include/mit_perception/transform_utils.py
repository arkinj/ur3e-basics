import numpy as np
from scipy.spatial.transform import Rotation as R

# env pose: position (x, y) and angle (theta)
# tag pose: translation (x, y, z) and rotation (R)
# arm pose: position (x, y) -- we only change these i think
# and distance units for all of them ???

# let's pick a common unit to work in: say mm (tag unit?)
# also just say we use arm origin as real origin for now
# and for simplicity we only look at 2D, but can generalize if needed

# env coordinates [0,500]x[0,500] -> real coordinates [-250,250]x[0,500] (mm)

ARM_UNIT = 1000.0 # mm
ENV_UNIT = 1.0 # need to calibrate this?
TAG_UNIT = 1000.0 # okay as long as we specify tag length in mm for detector?

# distance units in mm
# (x, y, theta) in (mm, mm, radians)
ARM_ORIGIN = np.array([0, 0]) # choose arm origin as ref
ENV_ORIGIN = np.array([-250, 150]) # need to calibrate this?
# multiple ref tags
TAG_ORIGINS = {
    0: np.array([ 0,  0], dtype=float),
    1: np.array([74,  0], dtype=float),
    2: np.array([ 0, 74], dtype=float),
    3: np.array([74, 74], dtype=float)}

# NOTE: WE ARE GOING TO ASSUME ALL ANGLES ARE THE SAME FOR NOW
# can also just construct experiment like that so it hsould be fine?
# so the below all assume np array of [x, y] in the respective frames
 
# functions to convert between different frames

def arm2env(arm_pos):
    return (arm_pos * ARM_UNIT + ARM_ORIGIN - ENV_ORIGIN) / ENV_UNIT

def env2arm(env_pos):
    return (env_pos * ENV_UNIT + ENV_ORIGIN - ARM_ORIGIN) / ARM_UNIT

def env2tag(env_pos, tag_id):
    TAG_ORIGIN = TAG_ORIGINS[tag_id]
    return (env_pos * ENV_UNIT + ENV_ORIGIN - TAG_ORIGIN) / TAG_UNIT

def tag2env(tag_pos, tag_id):
    TAG_ORIGIN = TAG_ORIGINS[tag_id]
    return (tag_pos * TAG_UNIT + TAG_ORIGIN - ENV_ORIGIN) / ENV_UNIT

def tag2arm(tag_pos, tag_id):
    TAG_ORIGIN = TAG_ORIGINS[tag_id]
    return (tag_pos * TAG_UNIT + TAG_ORIGIN - ARM_ORIGIN) / ARM_UNIT

def arm2tag(arm_pos, tag_id):
    TAG_ORIGIN = TAG_ORIGINS[tag_id]
    return (arm_pos * ARM_UNIT + ARM_ORIGIN - TAG_ORIGIN) / TAG_UNIT

# functions to convert between frames and real
# some redundancy with above function implementations...

def arm2real(arm_pos):
    return arm_pos * ARM_UNIT + ARM_ORIGIN

def env2real(env_pos):
    return env_pos * ENV_UNIT + ENV_ORIGIN

def tag2real(tag_pos, tag_id):
    TAG_ORIGIN = TAG_ORIGINS[tag_id]
    return tag_pos * TAG_UNIT + TAG_ORIGIN

def real2arm(real_pos):
    return (real_pos - ARM_ORIGIN) / ARM_UNIT

def real2env(real_pos):
    return (real_pos - ENV_ORIGIN) / ENV_UNIT

def real2tag(real_pos, tag_id):
    TAG_ORIGIN = TAG_ORIGINS[tag_id]
    return (real_pos - TAG_ORIGIN) / TAG_UNIT


# we should also handle the april tag and T stuff
# here let's assume the tags are flat and on top of the T
# NOTE: are there potential issue with the arm blocking view of tags?

# for each tag, we need (x, y, theta) offset from T center (in mm, T frame)

# NOTE: order of offsets has to align with order in which tag poses are given

# assume just one tag on center for now
# also just assume we are aligning all angles to the T

# center of mass of T wrt upper left corner of 120x120 bounding box
# T_COM_COORDINATES = (60, 285/7) ?? check this

T_TAG_ORIGINS = {
    # tags A, B, C, D (upper part of T)
    4:  np.array([ -45, -180/7], dtype=float),
    5:  np.array([ -15, -180/7], dtype=float),
    6:  np.array([  15, -180/7], dtype=float),
    7:  np.array([  45, -180/7], dtype=float),
    # tags E, F, G (lower part of T)
    8:  np.array([   0,   30/7], dtype=float),
    9:  np.array([   0,  240/7], dtype=float),
    10: np.array([   0,  450/7], dtype=float)}

def tag_pose_to_T_env_pose(pose, tag_id, ref_tag_id):
    translation, rotation = pose
    offset = T_TAG_ORIGINS[tag_id]
    pos = tag2real(translation[:2], ref_tag_id)
    rot_matrix = rotation.as_matrix()[:2,:2]
    T_pos_real = pos - rot_matrix.dot(offset)
    T_pos_env = real2env(T_pos_real)
    T_angle = rotation.as_rotvec()[2]
    return T_pos_env, T_angle

def tag_poses_to_T_env_pose(tag_poses, ref_tag_id):
    n_tags = len(tag_poses)
    if n_tags == 0:
        return None
    positions = np.zeros((n_tags, 2))
    angles = np.zeros(n_tags)
    # take all pose estimates and just average them
    for i, (tag_id, pose) in enumerate(tag_poses.items()):
        position, angle = tag_pose_to_T_env_pose(pose, tag_id, ref_tag_id)
        positions[i,:] = position
        angles[i] = angle
    avg_angle = np.mean(angles)
    avg_position = np.mean(positions, axis=0)
    return avg_position, avg_angle
    