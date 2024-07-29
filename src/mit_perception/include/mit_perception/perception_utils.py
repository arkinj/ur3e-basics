import json
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

def get_camera_pipeline(width, height, stream_format):
  """Starts an RGB image stream from the RealSense. Gets pipeline object."""
  pipeline = rs.pipeline()
  config = rs.config()

  if stream_format == "rgb":
    config.enable_stream(
      stream_type=rs.stream.color, width=width, height=height, format=rs.format.rgb8, framerate=15
    )
  elif stream_format == "bgr":
    config.enable_stream(
      stream_type=rs.stream.color, width=width, height=height, format=rs.format.bgr8, framerate=15
    )
  else:
    raise RuntimeError("Stream format not recognized.")

  pipeline.start(config)

  # Get the sensor
  sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
  # Set the exposure
  sensor.set_option(rs.option.exposure, 156.000)

  return pipeline

def get_intrinsics(pipeline):
  """Gets the intrinsics for the RBG camera from the RealSense."""
  profile = pipeline.get_active_profile()

  color_profile = rs.video_stream_profile(profile.get_stream(stream_type=rs.stream.color))
  color_intrinsics = color_profile.get_intrinsics()

  color_intrinsics_dict = {
    "cx": color_intrinsics.ppx,
    "cy": color_intrinsics.ppy,
    "fx": color_intrinsics.fx,
    "fy": color_intrinsics.fy
  }

  return color_intrinsics_dict

def get_image(pipeline, display_images=False, silent=False):
  """Gets an RGB image from the RealSense."""
  if not silent:
    print("\nGetting image...")
  frames = pipeline.wait_for_frames()
  color_frame = frames.get_color_frame()
  color_image = np.asanyarray(color_frame.get_data())
  if not silent:
    print("Got image.")

  if display_images:
    cv2.namedWindow(winname="RGB Output", flags=cv2.WINDOW_AUTOSIZE)
    cv2.imshow(winname="RGB Outout", mat=color_image)
    cv2.waitKey(delay=2000)
    cv2.destroyAllWindows()

  return color_image

def label_tag_detection(image, tag_center_pixel, tag_corner_pixels, tag_family):
  """Labels a tag detection on an image."""
  image_labeled = image.copy()

  # Draw a circle at the image center
  center = (int(tag_center_pixel[0]), int(tag_center_pixel[1]))
  cv2.circle(img=image_labeled, center=center, radius=5, color=(0,0,0))

  # Draw circles at each corner
  corner_a = (int(tag_corner_pixels[0][0]), int(tag_corner_pixels[0][1]))
  corner_b = (int(tag_corner_pixels[1][0]), int(tag_corner_pixels[1][1]))
  corner_c = (int(tag_corner_pixels[2][0]), int(tag_corner_pixels[2][1]))
  corner_d = (int(tag_corner_pixels[3][0]), int(tag_corner_pixels[3][1]))

  cv2.circle(img=image_labeled, center=corner_a, radius=10, color=(255,0,0))
  cv2.putText(
    img=image_labeled,
    text="corner_a",
    org=(corner_a[0], corner_a[1]-10),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=0.5,
    color=(255,0,0),
    thickness=2,
    lineType=cv2.LINE_AA
  )
  cv2.circle(img=image_labeled, center=corner_b, radius=10, color=(0,255,0))
  cv2.putText(
    img=image_labeled,
    text="corner_b",
    org=(corner_b[0], corner_b[1]-10),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=0.5,
    color=(0,255,0),
    thickness=2,
    lineType=cv2.LINE_AA
  )
  cv2.circle(img=image_labeled, center=corner_c, radius=10, color=(0,0,255))
  cv2.putText(
    img=image_labeled,
    text="corner_c",
    org=(corner_c[0], corner_c[1]-10),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=0.5,
    color=(0,0,255),
    thickness=2,
    lineType=cv2.LINE_AA
  )
  cv2.circle(img=image_labeled, center=corner_d, radius=10, color=(0,0,0))
  cv2.putText(
    img=image_labeled,
    text="corner_d",
    org=(corner_d[0], corner_d[1]-10),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=0.5,
    color=(0,0,0),
    thickness=2,
    lineType=cv2.LINE_AA
  )

  # Draw oriented box on image
  cv2.line(img=image_labeled, pt1=corner_a, pt2=corner_b, color=(0, 255, 0), thickness=2)
  cv2.line(img=image_labeled, pt1=corner_b, pt2=corner_c, color=(0, 255, 0), thickness=2)
  cv2.line(img=image_labeled, pt1=corner_c, pt2=corner_d, color=(0, 255, 0), thickness=2)
  cv2.line(img=image_labeled, pt1=corner_d, pt2=corner_a, color=(0, 255, 0), thickness=2)

  return image_labeled

def save_image(image, filename):
  """Saves an image to file."""
  print("\nSaving image...")
  cv2.imwrite(filename, img=(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)))
  print("Saved image.")
  return

def get_tag_pose_in_camera_frame(detector, image, intrinsics, tag_length, tag_active_pixel_ratio, detect_idx=0):
  """Detects an AprilTag in an image. Gets the pose of the tag in the camera frame."""
  gray_image = cv2.cvtColor(src=image.astype(np.uint8), code=cv2.COLOR_BGR2GRAY)
  tag_active_length = tag_length * 0.0254 * tag_active_pixel_ratio
  detection = detector.detect(
    img=gray_image,
    estimate_tag_pose=True,
    camera_params=[intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]],
    tag_size=tag_active_length
  )

  if detection:
    is_detected = True
    if detect_idx >= len(detection):
      print(f"can't select detect_idx {detect_idx}, only {len(detection)} tags found")
    pos = detection[detect_idx].pose_t.copy().squeeze() # (3, )
    ori_mat = detection[detect_idx].pose_R.copy()
    center_pixel = detection[detect_idx].center
    corner_pixels = detection[detect_idx].corners
    family = detection[detect_idx].tag_family
  else:
    is_detected = False
    pos, ori_mat, center_pixel, corner_pixels, family = None, None, None, None, None

  return is_detected, pos, ori_mat, center_pixel, corner_pixels, family

def get_tag_poses_in_camera_frame(detector, image, intrinsics, tag_length, tag_active_pixel_ratio, as_detection=False):
  """Detects an AprilTag in an image. Gets the pose of the tag in the camera frame."""
  gray_image = cv2.cvtColor(src=image.astype(np.uint8), code=cv2.COLOR_BGR2GRAY)
  tag_active_length = tag_length * 0.0254 * tag_active_pixel_ratio
  detection = detector.detect(
    img=gray_image,
    estimate_tag_pose=True,
    camera_params=[intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]],
    tag_size=tag_active_length
  )

  if detection:
    is_detected = True
    print(f"{len(detection)} tags found :)")
    if as_detection:
      return detection
    else:
      pos = [None] * len(detection)
      ori_mat = [None] * len(detection)
      center_pixel = [None] * len(detection)
      corner_pixels = [None] * len(detection)
      family = [None] * len(detection)
      for i in range(len(detection)):
        pos[i] = detection[i].pose_t.copy().squeeze() # (3, )
        ori_mat[i] = detection[i].pose_R.copy()
        center_pixel[i] = detection[i].center
        corner_pixels[i] = detection[i].corners
        family[i] = detection[i].tag_family
  else:
    is_detected = False
    pos, ori_mat, center_pixel, corner_pixels, family = None, None, None, None, None

  return is_detected, pos, ori_mat, center_pixel, corner_pixels, family

def keypress_images(pipeline, display_image=False):
  count = 0
  while(count < 20):
    input("Press enter to take an image")
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())

    depth_frame = frames.get_depth_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_color_map = cv2.applyColorMap(
      cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET
    )
    if display_image:
      #images = np.hstack((color_image, depth_color_map))
      cv2.namedWindow("RBG-D Output", cv2.WINDOW_AUTOSIZE)
      cv2.imshow("RBG-D Ouptut", color_image)
      cv2.waitKey(1000)
    combined_filename = "/tmp/image_combined" + str(count) + ".jpg"
    color_filename = "/tmp/image" + str(count) + ".jpg"
    cv2.imwrite(combined_filename, np.hstack((color_image, depth_color_map)))
    cv2.imwrite(color_filename, color_image)
    count += 1

  return color_image, depth_image

def save_images(pipeline, display_image=False):
  """Gets RGB and depth images from RealSense."""
  count = 0
  while(count<100):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())

    depth_frame = frames.get_depth_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_color_map = cv2.applyColorMap(
      cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET
    )
    if display_image:
      images = np.hstack((color_image, depth_color_map))
      cv2.namedWindow("RBG-D Output", cv2.WINDOW_AUTOSIZE)
      cv2.imshow("RBG-D Ouptut", images)
      cv2.waitKey(1000)
    filename = "/tmp/image" + str(count) + ".jpg"
    cv2.imwrite(filename, np.hstack((color_image, depth_color_map)))
    count += 1

  return color_image, depth_image

def get_color_and_depth_image(pipeline, display_image=False):
  """Gets RGB and depth images from RealSense."""
  frames = pipeline.wait_for_frames()

  color_frame = frames.get_color_frame()
  color_image = np.asanyarray(color_frame.get_data())

  depth_frame = frames.get_depth_frame()
  depth_image = np.asanyarray(depth_frame.get_data())
  depth_color_map = cv2.applyColorMap(
    cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET
  )

  if display_image:
    images = np.hstack((color_image, depth_color_map))
    cv2.namedWindow("RBG-D Output", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("RBG-D Ouptut", images)
    cv2.waitKey(1000)

  return color_image, depth_image
