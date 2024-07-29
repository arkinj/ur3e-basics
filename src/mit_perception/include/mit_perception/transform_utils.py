import numpy as np
from scipy.spatial.transform import Rotation as R

# env pose: position (x, y) and angle (theta)
# tag pose: translation (x, y, z) and rotation (R)
# arm pose: position (x, y) -- we only change these i think
# and distance units for all of them ???

# let's pick a common unit to work in: say mm (tag unit?)
# also just say we use arm origin as real origin for now
# and for simplicity we only look at 2D, but can generalize if needed

# env coordinates [0,500]x[0,500] -> arm coordinates [-250,0]x[250,500] (mm)

ARM_UNIT = 1000.0 # mm
ENV_UNIT = 2.0 # need to calibrate this?
TAG_UNIT = 1.0 # okay as long as we specify tag length in mm for detector?

# distance units in mm
# (x, y, theta) in (mm, mm, radians)
ARM_ORIGIN = np.ndarray([0, 0]) # choose arm origin as ref
ENV_ORIGIN = np.ndarray([-250, 250]) # need to calibrate this?
TAG_ORIGIN = np.ndarray([0, -100]) # wild guess, need to calibrate this

# NOTE: WE ARE GOING TO ASSUME ALL ANGLES ARE THE SAME FOR NOW
# can also just construct experiment like that so it hsould be fine?
# so the below all assume np array of [x, y] in the respective frames

def arm2env(arm_pos):
    return (arm_pos * ARM_UNIT + ARM_ORIGIN - ENV_ORIGIN) / ENV_UNIT

def env2arm(env_pos):
    return (env_pos * ENV_UNIT + ENV_ORIGIN - ARM_ORIGIN) / ARM_UNIT

def env2tag(env_pos):
    return (env_pos * ENV_UNIT + ENV_ORIGIN - TAG_ORIGIN) / TAG_UNIT

def tag2env(tag_pos):
    return (tag_pos * TAG_UNIT + TAG_ORIGIN - ENV_ORIGIN) / ENV_UNIT

def tag2arm(tag_pos):
    return (tag_pos * TAG_UNIT + TAG_ORIGIN - ARM_ORIGIN) / ARM_UNIT

def arm2tag(arm_pos):
    return (arm_pos * ARM_UNIT + ARM_ORIGIN - TAG_ORIGIN) / TAG_UNIT


# we should also handle the april tag and T stuff
# here let's assume the tags are flat and on top of the T
# NOTE: are there potential issue with the arm blocking view of tags?

# for each tag, we need (x, y, theta) offset from T center (in mm, T frame)

# NOTE: order of offsets has to align with order in which tag poses are given

# assume just one tag on center for now
# also just assume we are aligning all angles to the T
T_TAG_ORIGINS = [
    np.ndarray([0, 0])
]

def tag_pose_to_T_env_pose(translation, rotation, offset=T_TAG_ORIGINS[0]):
    pos = translation[:2] * TAG_UNIT
    rot_matrix = rotation.as_matrix()[:2,:2]
    T_angle = rotation.as_euler("XYZ")[2]
    T_pos_tag = pos - rot_matrix * offset
    T_pos_env = tag2env(T_pos_tag)
    return T_pos_env, T_angle

def tag_poses_to_T_env_pose(translations, rotations):
    n_tags = len(translations)
    positions = np.zeros(n_tags, 2)
    angles = np.zeros(n_tags)
    # take all pose estimates and just average them
    for idx, t, r in enumerate(zip(translations, rotations)):
        if idx >= len(T_TAG_ORIGINS):
            break
        position, angle = tag_pose_to_T_env_pose(t, r, T_TAG_ORIGINS[idx])
        positions.append(position)
        angles.append(angle)
    avg_angle = np.mean(angles)
    avg_position = np.mean(positions, axis=0)
    return avg_position, avg_angle
    