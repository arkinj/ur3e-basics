import numpy as np
from scipy.spatial.transform import Rotation as R
import math

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
ENV_ORIGIN = np.array([-250-6, 100-6]) # -6 to account for 512x512 inc walls
# multiple ref tags
TAG_ORIGINS = {
    0: np.array([-250+37, 100+37], dtype=float),
    1: np.array([ 250-38, 100+37], dtype=float),
    2: np.array([-250+37, 484-38], dtype=float),
    3: np.array([ 250-36, 484-37], dtype=float)}

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
# T_COM = (60, 285/7) ?? check this
# "center" position is based on, wrt bounding box also
T_CENTER = np.array([60, 30], dtype=float)

T_TAG_ORIGINS = {
    # tags A, B, C, D (upper part of T)
    4:  np.array([ 15,  15], dtype=float) - T_CENTER,
    5:  np.array([ 45,  15], dtype=float) - T_CENTER,
    6:  np.array([ 75,  15], dtype=float) - T_CENTER,
    7:  np.array([105,  15], dtype=float) - T_CENTER,
    # tags E, F, G (lower part of T)
    8:  np.array([ 60,  45], dtype=float) - T_CENTER,
    9:  np.array([ 60,  75], dtype=float) - T_CENTER,
    10: np.array([ 60, 105], dtype=float) - T_CENTER,
    # side tags 
    # upper left single
    11: np.array([  0,  15], dtype=float) - T_CENTER, 
    # top row of 4
    12: np.array([  15,  0], dtype=float) - T_CENTER, 
    13: np.array([  45,  0], dtype=float) - T_CENTER, 
    14: np.array([  75,  0], dtype=float) - T_CENTER, 
    15: np.array([ 105,  0], dtype=float) - T_CENTER, 
    #
    16: np.array([ 120, 15], dtype=float) - T_CENTER, 
    #
    17: np.array([ 105, 30], dtype=float) - T_CENTER, 
    #
    18: np.array([ 75,  45], dtype=float) - T_CENTER, 
    19: np.array([ 75,  75], dtype=float) - T_CENTER, 
    20: np.array([ 75, 105], dtype=float) - T_CENTER, 
    #
    21: np.array([ 60, 120], dtype=float) - T_CENTER, 
    #
    22: np.array([ 45, 105], dtype=float) - T_CENTER, 
    23: np.array([ 45,  75], dtype=float) - T_CENTER, 
    24: np.array([ 45,  45], dtype=float) - T_CENTER,
    # 
    25: np.array([ 15,  30], dtype=float) - T_CENTER, 
}

R_IDENTITY = R.from_rotvec(np.zeros(3)) 

R_CCW_Z = R.from_rotvec(np.pi/2 * np.array([0, 0, 1]))
R_CCW_Y = R.from_rotvec(np.pi/2 * np.array([0, 1, 0]))
R_CCW_X = R.from_rotvec(np.pi/2 * np.array([1, 0, 0]))

R_CW_Z = R.from_rotvec(-np.pi/2 * np.array([0, 0, 1]))
R_CW_Y = R.from_rotvec(-np.pi/2 * np.array([0, 1, 0]))
R_CW_X = R.from_rotvec(-np.pi/2 * np.array([1, 0, 0]))

R_T_L = R_CW_Z * R_CW_Y
R_T_U = R_CW_Z * R_CW_Y * R_CW_Z
R_T_R = R_CW_Z * R_CW_Y * R_CW_Z * R_CW_Z
R_T_D = R_CW_Z * R_CW_Y * R_CW_Z * R_CW_Z * R_CW_Z

T_TAG_ROTATIONS = {
    # flat tags
    4: R_IDENTITY,
    5: R_IDENTITY,
    6: R_IDENTITY,
    7: R_IDENTITY,
    8: R_IDENTITY,
    9: R_IDENTITY,
    10: R_IDENTITY,
    # side tags
    11: R_T_L,
    #
    12: R_T_U,
    13: R_T_U,
    14: R_T_U,
    15: R_T_U,
    #
    16: R_T_R,
    #
    17: R_T_D,
    #
    18: R_T_R,
    19: R_T_R,
    20: R_T_R,
    #
    21: R_T_D,
    #
    22: R_T_L,
    23: R_T_L,
    24: R_T_L,
    #
    25: R_T_D
}

def tag_pose_to_T_env_pose(pose, tag_id, ref_tag_id):
    translation, rotation = pose
    offset = T_TAG_ORIGINS[tag_id]
    pos = tag2real(translation[:2], ref_tag_id)

    rotation = R.inv(T_TAG_ROTATIONS[tag_id]) * rotation
    rot_matrix = R.inv(rotation).as_matrix()[:2,:2]
    # T_trans_real = np.hstack((pos,translation[2:]*TAG_UNIT)) \
    #     - rotation.apply(np.hstack((offset,translation[2:]*TAG_UNIT)))
    # T_pos_real = T_trans_real[:2]
    T_pos_real = pos - rot_matrix.dot(offset)
    T_pos_env = real2env(T_pos_real)
    T_angle = rotation.as_rotvec()[2]
    return T_pos_env, T_angle

def tag_poses_to_T_env_pose(tag_poses, ref_tag_id, quiet=True):
    n_tags = len(tag_poses)
    if n_tags == 0:
        return None
    positions = np.zeros((n_tags, 2))
    angles = np.zeros(n_tags)
    # take all pose estimates and aggregate them
    for i, (tag_id, pose) in enumerate(tag_poses.items()):
        position, angle = tag_pose_to_T_env_pose(pose, tag_id, ref_tag_id)
        positions[i,:] = position
        angles[i] = angle

    # debugging...
    if not quiet:
        print(f" |-- using reference tag {ref_tag_id}:")
        np.set_printoptions(precision=3, suppress=True)
        tag_ids = list(tag_poses.keys())
        for i, (position, angle) in enumerate(zip(positions, angles)):
            tag_id = tag_ids[i]
            dashes = 4 - len(str(tag_id))
            print(f' | |{"-"*dashes} {tag_id} | pos: {env2real(position)}, rot: {math.degrees(angle):3.3f}')
            
    angle = np.median(angles)
    position = np.median(positions, axis=0)
    # angle = np.median(angles)
    # position = np.median(positions, axis=0)

    return position, angle
    