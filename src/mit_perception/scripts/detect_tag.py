import argparse
import json
import os

import cv2
import numpy as np
import pupil_apriltags as apriltag
 
import mit_perception.perception_utils as perception_utils

def main():
  """Initialize the camera. Get an image and tag pose."""
  # Initialize the camera and AprilTag detector
  pipeline = perception_utils.get_camera_pipeline(
    width=1280, height=720, stream_format='bgr'
  )
  intrinsics = perception_utils.get_intrinsics(pipeline=pipeline)
  detector = apriltag.Detector(
    families="tagStandard52h13", quad_decimate=1.0, quad_sigma=0.0, decode_sharpening=0.25
  )

  # Get an image
  image = perception_utils.get_image(
    pipeline=pipeline, display_images=False
  )

  # Detect an april tag
  (
    is_tag_detected,
    tag_pose_t,
    tag_pose_r,
    tag_center_pixel,
    tag_corner_pixels,
    tag_family
  ) = perception_utils.get_tag_pose_in_camera_frame(
    detector=detector,
    image=image,
    intrinsics=intrinsics,
    tag_length=6.0, # inches
    tag_active_pixel_ratio= 0.6 # Magic
  )

if __name__ == "__main__":
  main()
