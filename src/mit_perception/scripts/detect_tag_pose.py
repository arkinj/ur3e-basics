import argparse
import json
import os

import cv2
import numpy as np
import pupil_apriltags as apriltag
 
import mit_perception.perception_utils as perception_utils
from scipy.spatial.transform import Rotation as R
import math

def main():
  """Initialize the camera. Get an image and tag pose."""
  # Initialize the camera and AprilTag detector
  pipeline = perception_utils.get_camera_pipeline(
    width=1280, height=720, stream_format='bgr'
  )
  intrinsics = perception_utils.get_intrinsics(pipeline=pipeline)
  detector = apriltag.Detector(
    # families="tagStandard52h13", quad_decimate=1.0, quad_sigma=0.0, decode_sharpening=0.25
    families="tag36h11", quad_decimate=1.0, quad_sigma=0.0, decode_sharpening=0.25
  )

  generate_images = False

  if generate_images:
    # Get an image
    image = perception_utils.get_image(
      pipeline=pipeline, display_images=False
    )

    # Detect april tags
    (
      is_tag_detected,
      tag_pose_t,
      tag_pose_r,
      tag_center_pixel,
      tag_corner_pixels,
      tag_family
    ) = perception_utils.get_tag_poses_in_camera_frame(
      detector=detector,
      image=image,
      intrinsics=intrinsics,
      tag_length=2.0, # inches
      tag_active_pixel_ratio= 0.6 # Magic
    )

    print(is_tag_detected)
    print(tag_pose_t)
    print(tag_pose_r)
    print(tag_center_pixel)

    image_labeled = image
    images_labeled = []

    for i in range(len(tag_center_pixel)):
      image_labeled = perception_utils.label_tag_detection(
        image_labeled,
        tag_center_pixel[i],
        tag_corner_pixels[i],
        tag_family[i]
      )
      image_labeled_single_tag = perception_utils.label_tag_detection(
        image,
        tag_center_pixel[i],
        tag_corner_pixels[i],
        tag_family[i]
      )
      images_labeled.append(image_labeled_single_tag)

    # use this identify your moving and stationary tag indexes?
    img_base_path = "src/mit_perception"
    perception_utils.save_image(image,f"{img_base_path}/image.jpg")
    perception_utils.save_image(image_labeled,f"{img_base_path}/image_labeled.jpg")
    for i in range(len(images_labeled)):
      perception_utils.save_image(images_labeled[i],f"{img_base_path}/image_labeled_tag_{i}.jpg")



  continuous = True

  while True:
    image = perception_utils.get_image(
      pipeline=pipeline, display_images=False, silent=True
    )

    # get tags as detection objects

    tags = perception_utils.get_tag_poses_in_camera_frame(
      detector=detector,
      image=image,
      intrinsics=intrinsics,
      tag_length=2.0, # inches
      tag_active_pixel_ratio= 0.6, # Magic
      as_detection=True
    )

    if len(tags) < 2:
      print(f"only {len(tags)} in camera view :(")
      if continuous:
        print("trying again...")
        continue
      else:
        print("exiting...")
        break

    # transforming pose estimates to stationary tag frame
    # reference: https://github.com/dawsonc/tello-x/blob/main/tellox/pilot.py

    tagS = tags[1] # stationary tag (use this frame of reference)
    tagM = tags[0] # moving tag

    # rotations for stationary tag
    R_cam_tagS = R.from_matrix(tagS.pose_R)
    R_tagS_cam = R.inv(R_cam_tagS)

    # translations for stationary tag in camera frame
    p_cam_tagS_cam = tagS.pose_t.reshape(-1)
    p_tagS_cam_cam = -p_cam_tagS_cam
    # camera translation in stationary tag frame
    p_tagS_cam_tagS = R_tagS_cam.apply(p_tagS_cam_cam)

    # rotations for moving tag
    R_cam_tagM = R.from_matrix(tagM.pose_R)
    R_tagM_cam = R.inv(R_cam_tagM)
    # rotation for moving tag in stationary tag frame
    R_tagM_tagS = R_tagM_cam * R_cam_tagS

    # translations for moving tag in camera frame
    p_cam_tagM_cam = tagM.pose_t.reshape(-1)
    # camera to moving tag translation in stationary tag frame
    p_cam_tagM_tagS = R_tagS_cam.apply(p_cam_tagM_cam)

    # moving tag translation in stationary tag frame
    p_tagS_tagM_tagS = p_tagS_cam_tagS + p_cam_tagM_tagS

    # trying something
    p_tagS_tagM_cam = p_tagS_cam_cam + p_cam_tagM_cam
    p_tagS_tagM_tagS_alt = R_tagS_cam.apply(p_tagS_tagM_cam)

    euler_angles = R_tagM_tagS.as_euler("XYZ")

    np.set_printoptions(precision=3, suppress=True)
    # print(p_tagS_tagM_tagS)
    print("tagM translation in tagS frame:")
    print(p_tagS_tagM_tagS_alt)
    print("tagM rotation in tagS frame:")
    print(R_tagM_tagS.as_matrix())
    print("tagM rotation in XYZ euler angles (rad):")
    print(euler_angles)

    if not continuous:
      break

if __name__ == "__main__":
  main()
