import argparse
import json
import os

import cv2
import numpy as np
import pupil_apriltags as apriltag

from scipy.spatial.transform import Rotation as R
import math

import argparse

from mit_perception.apriltag_utils import *
from mit_perception.perception_utils import *
from mit_perception.transform_utils import *

import time

# sizes in meters, length does not include border
SMALL_TAG_SIZE = 0.023
LARGE_TAG_SIZE = 0.060

# key: tag size, value (tuple of tag_ids)
# by default use T tags as small size, ref tags as large size
TAG_SIZES = {
    SMALL_TAG_SIZE: frozenset(T_TAG_ORIGINS.keys()),
    LARGE_TAG_SIZE: frozenset(TAG_ORIGINS.keys())
}

def main():

    default_ref_ids = TAG_ORIGINS.keys()
    default_mov_ids = T_TAG_ORIGINS.keys()

    parser = argparse.ArgumentParser()
    parser.add_argument("--show-depth", action="store_true", help="Show depth image.")
    parser.add_argument('-g', '--generate-images', action='store_true')
    parser.add_argument('-c', '--continuous', action='store_true')
    parser.add_argument('-r', '--ref-ids', nargs="+", type=int, default=default_ref_ids)
    parser.add_argument('-m', '--mov-ids', nargs="+", type=int, default=default_mov_ids)
    args = parser.parse_args()

    if len(args.ref_ids) < 1:
        print("need at least one ref tag id")
        return
    if len(args.mov_ids) < 1:
        print("need at least one moving tag id")
        return

    # """Initialize the camera. Get an image and tag pose."""
    # # Initialize the camera and AprilTag detector
    # pipeline = perception_utils.get_camera_pipeline(
    #     width=1280, height=720, stream_format='bgr'
    # )
    # intrinsics = perception_utils.get_intrinsics(pipeline=pipeline)
    # detector = apriltag.Detector(
    #     # families="tagStandard52h13", quad_decimate=1.0, quad_sigma=0.0, decode_sharpening=0.25
    #     families="tag36h11", quad_decimate=1.0, quad_sigma=0.0, decode_sharpening=0.25
    # )
    cam = RealsenseCam(
        "317422075665",
        (1280, 720), # color img size
        (1280, 720), # depth img size
        30 # frame rate
    )
    # tag size in meters, or dict of tag_size: tag_ids
    april_tag = AprilTag(tag_size=TAG_SIZES) 

    # cv2.namedWindow("RealsenseAprilTag", cv2.WINDOW_AUTOSIZE)

    while True:
        tags, color_img, depth_image = detect_tags(april_tag, cam, return_imgs=True)
        tag_dict = {tag.tag_id: tag for tag in tags}

        # can adjust the detection output if need to clean this up 
        # but this is more compatible with single tag size version for now
        detected_ref_tags = [tag_dict[ref_id] for ref_id in args.ref_ids if ref_id in tag_dict]
        detected_mov_tags = [tag_dict[mov_id] for mov_id in args.mov_ids if mov_id in tag_dict]

        # for tag in tags:
        #     print(tag.tag_id, tag.pose_t.reshape(-1), R.from_matrix(tag.pose_R).as_euler("XYZ"))

        if len(detected_ref_tags) > 0:
            # get T tag poses wrt each reference tag
            tag_poses_wrt_ref = {ref_tag.tag_id:
                get_tag_poses_in_ref_tag_frame(detected_mov_tags, ref_tag)
                for ref_tag in detected_ref_tags}

            # debugging...
            # for tag_id, poses in tag_poses_wrt_ref.items():
            #     print(tag_id)
            #     for idx, pose in poses.items():
            #         # print(idx, pose[0], pose[1].as_rotvec(degrees=True))
            #         pos, angle = tag_pose_to_T_env_pose(pose, idx, tag_id)
            #         real_pos = env2real(pos)
            #         print(idx, real_pos, angle)
            
            # slow down updates for debugging
            # time.sleep(1)
            
            # get T poses in env frame (as (x,y), theta)
            T_env_poses = {ref_tag_id:
                tag_poses_to_T_env_pose(poses, ref_tag_id)
                for ref_tag_id, poses in tag_poses_wrt_ref.items()}

            # debugging...
            # for tag_id, pose in T_env_poses.items():
            #     print(tag_id, pose)

            positions = [pose[0] 
                for pose in list(T_env_poses.values()) if pose is not None]
            angles = [pose[1] 
                for pose in list(T_env_poses.values()) if pose is not None]

            if len(positions) > 0:
                T_env_pose_avg = (
                    np.mean(np.vstack(positions), axis=0),
                    np.mean(np.vstack(angles)))
                print(T_env_pose_avg)
            else:
                print("no T tag found ;-;") 


        # visualization
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        if args.show_depth:
            depth_image = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET
            )

            depth_image = cv2.resize(
                depth_image, (color_img1.shape[1], color_img1.shape[0])
            )
            depth_img = depth_image
            img = np.vstack([color_img, depth_img])
        else:
            img = color_img
        cv2.imshow("Detected tags", img)

        k = cv2.waitKey(1)
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()
            break

        if not args.continuous:
            break

if __name__ == "__main__":
  main()
