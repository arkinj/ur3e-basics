import argparse
import json
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import math
import time
import argparse

from mit_perception.apriltag_utils import (
    AprilTag, 
    RealsenseCam, 
    detect_tags)

from mit_perception.perception_utils import(
    preprocess_T_tag_rotations_in_ref_tag_frame,
    get_tag_poses_in_ref_tag_frame)

from mit_perception.transform_utils import (
    TAG_ORIGINS, 
    T_TAG_ORIGINS, 
    T_TAG_ROTATIONS,
    env2real,
    tag_poses_to_T_env_pose)

# sizes in meters, length does not include border
SMALL_TAG_SIZE = 0.023
LARGE_TAG_SIZE = 0.060

# key: tag size, value (tuple of tag_ids)
# by default use T tags as small size, ref tags as large size
TAG_SIZES = {
    SMALL_TAG_SIZE: frozenset(T_TAG_ORIGINS.keys()),
    LARGE_TAG_SIZE: frozenset(TAG_ORIGINS.keys())
}

def visualize(
    color_imgs, 
    depth_imgs=None, 
    show_depth=False):

    if show_depth and depth_img is None:
        print("show_depth set to True, must provide depth image!")
        return
    depth_imgs = [None] * len(color_imgs)

    imgs = []
    
    for color_img, depth_img in zip(color_imgs, depth_imgs):
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        if show_depth:
            depth_image = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET
            )
            depth_image = cv2.resize(
                depth_image, (color_img.shape[1], color_img.shape[0])
            )
            depth_img = depth_image
            img = np.vstack([color_img, depth_img])
        else:
            img = color_img
        imgs.append(img)
    
    if show_depth:
        img = np.hstack(imgs)
    else:
        img = np.vstack(imgs)

    cv2.imshow("Detected tags", img)



############################################
###   BEGIN RELEVANT CODE TO STEP_REAL   ###
############################################

def get_state_estimate_T(
    april_tag: AprilTag, 
    cams: RealsenseCam, 
    ref_ids=TAG_ORIGINS.keys(), 
    mov_ids=T_TAG_ORIGINS.keys(), 
    quiet: bool=True, 
    show_cam: bool=True, 
    show_depth: bool=False):
    """
    gets state estimate of T using april tags and camera

    Args:
        april_tag (AprilTag): april tag
        cam (RealSenseCam): camera
        ref_ids (list[int]): list of reference tag ids 
        mov_ids (list[int]): list of T (moving) tag ids
        quiet (bool): if True, suppress all non-essential logging
        show_cam (bool): if True, show camera output
        show_depth (bool): if True, also show depth camera output

    Returns:
        ok: if state estimate was succesfully determined
        T_env_pose: (xy position, angle), in env frame
    """
    
    tag_dicts = []
    color_imgs = []
    depth_imgs = []
    for cam in cams:
        tags, color_img, depth_img = detect_tags(april_tag, cam, return_imgs=True)
        tag_dict = {tag.tag_id: tag for tag in tags}
        tag_dicts.append(tag_dict)
        color_imgs.append(color_img)
        depth_imgs.append(depth_img)

    positions_fused = []
    angles_fused = []

    for cam_idx, tag_dict in enumerate(tag_dicts):
        # can adjust the detection output if need to clean this up 
        # but this is more compatible with single tag size version for now
        detected_ref_tags = [tag_dict[ref_id] for ref_id in ref_ids if ref_id in tag_dict]
        detected_mov_tags = [tag_dict[mov_id] for mov_id in mov_ids if mov_id in tag_dict]

        ok = False
        T_env_pose = (None, None)

        if len(detected_ref_tags) > 0:
            # NEW: preprocess angles to hopefully handle the few that flip:

            # get T tag poses wrt each reference tag
            tag_poses_wrt_ref = {ref_tag.tag_id:
                get_tag_poses_in_ref_tag_frame(detected_mov_tags, ref_tag, T_TAG_ROTATIONS)
                for ref_tag in detected_ref_tags}

            # debugging...
            # for tag_id, poses in tag_poses_wrt_ref.items():
            #     print(tag_id)
            #     for idx, pose in poses.items():
            #         # print(idx, pose[0], pose[1].as_rotvec(degrees=True))
            #         pos, angle = tag_pose_to_T_env_pose(pose, idx, tag_id)
            #         real_pos = env2real(pos)tag_pose_to_T_env_pose
            #         print(idx, real_pos, angle)
            
            # slow down updates for debugging
            # time.sleep(1)

            if not quiet:
                print("\n=================================================")
                print(f"\n[CAM {cam_idx}] T state estimates from detected T tags:")
            
            # get T poses in env frame (as (x,y), theta)
            T_env_poses = {ref_tag_id:
                tag_poses_to_T_env_pose(poses, ref_tag_id, quiet=True)
                for ref_tag_id, poses in tag_poses_wrt_ref.items()}

            # debugging...
            if not quiet:
                print(f"\n[CAM {cam_idx}] T state estimates from detected ref tags:")
                for tag_id, pose in T_env_poses.items():
                    if pose is not None:
                        print(f' |----- {tag_id} | pos: {env2real(pose[0])}, rot: {math.degrees(pose[1]):3.3f}')

            positions = [pose[0] 
                for pose in list(T_env_poses.values()) if pose is not None]
            angles = [pose[1] 
                for pose in list(T_env_poses.values()) if pose is not None]

            if len(positions) > 0:
                T_env_pose = (
                    # np.mean(np.vstack(positions), axis=0),
                    # np.mean(np.vstack(angles))
                    np.median(np.vstack(positions), axis=0),
                    np.median(np.vstack(angles))
                )
                # debugging
                if not quiet:
                    np.set_printoptions(precision=3, suppress=True)
                    T_env_position = T_env_pose[0]
                    T_env_angle = math.degrees(T_env_pose[1])
                    T_real_position = env2real(T_env_pose[0])
                    T_real_angle = T_env_angle
                    print(f"\n[CAM {cam_idx}] T state estimate:")
                    print(f" |-- position  (env): {T_env_position}")
                    print(f" |----- angle  (env): {T_env_angle:3.3f}")
                    print(f" |-- position (real): {T_real_position}")
                    print(f" |----- angle (real): {T_real_angle:3.3f}")

                ok = True
                positions_fused.append(T_env_pose[0])
                angles_fused.append(T_env_pose[1])

            elif not quiet:
                print(f"[CAM {cam_idx}] no T tags found ;-;\n") 
        elif not quiet:
            print(f"[CAM {cam_idx}] no reference tags found ;-;")

    if len(positions_fused) > 0:
        position = sum(positions_fused) / len(positions_fused)
        angle = sum(angles_fused) / len(angles_fused)
        if not quiet:
            print("\n=================================================")
            print("\n[FUSED] T state estimate:")
            print(f" |-- position  (env): {position}")
            print(f" |----- angle  (env): {angle:3.3f}")
            print(f" |-- position (real): {env2real(position)}")
            print(f" |----- angle (real): {angle:3.3f}")


    else:
        position = None
        angle = None

    # visualization
    if show_cam:
        if show_depth:
            visualize(color_imgs, depth_imgs, show_depth)
        else:
            visualize(color_imgs)
    
    return ok, (position, angle)

def get_state_estimate_T_retry(
    april_tag, 
    cams, 
    ref_ids=TAG_ORIGINS.keys(), 
    mov_ids=T_TAG_ORIGINS.keys(), 
    quiet=True, 
    show_cam=True, 
    show_depth=False, 
    max_attempts=10):
    """
    wrapper for get_state_estimate_T with retry on failure
    """

    for _ in range(max_attempts):
        ok, (position, angle) = \
            get_state_estimate_T(
                april_tag, cams,
                ref_ids, mov_ids,
                quiet, show_cam,
                show_depth)
        if ok:
            return True, (position, angle)
    if not quiet:
        print(f"failed to get state estimate in {max_attempts} attempts...")
        print(f"using stale state estimate ;-;")
    return False, (None, None)

############################################
###    END RELEVANT CODE TO STEP_REAL    ###
############################################





def main():

    default_ref_ids = TAG_ORIGINS.keys()
    default_mov_ids = T_TAG_ORIGINS.keys()

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--show-depth", action="store_true", help="Show depth image.")
    parser.add_argument('-r', '--ref-ids', nargs="+", type=int, default=default_ref_ids)
    parser.add_argument('-m', '--mov-ids', nargs="+", type=int, default=default_mov_ids)
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('-a', '--max-attempts', type=int, default=10)
    args = parser.parse_args()

    if len(args.ref_ids) < 1:
        print("need at least one ref tag id")
        return
    if len(args.mov_ids) < 1:
        print("need at least one moving tag id")
        return

    cams = [
        RealsenseCam(
            "317422075665",
            (1280, 720), # color img size
            (1280, 720), # depth img size
            30), # frame rate
        RealsenseCam(
            "243222071097",
            (1280, 720),
            (1280, 720),
            30)]
    # tag size in meters, or dict of tag_size: tag_ids
    april_tag = AprilTag(tag_size=TAG_SIZES) 

    # cv2.namedWindow("RealsenseAprilTag", cv2.WINDOW_AUTOSIZE)

    while True:
        ok, (position, angle) = get_state_estimate_T_retry(
            april_tag, cams, 
            ref_ids=args.ref_ids, 
            mov_ids=args.mov_ids, 
            quiet=args.quiet, 
            show_depth=args.show_depth,
            max_attempts=args.max_attempts)
        if not ok:
            print(f"state estimate from cam failed")

        k = cv2.waitKey(1)
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
  main()
