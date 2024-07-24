#!/usr/bin/env python
import os
import time

import mit_perception.perception_utils as perception_utils

def main():
  print("Starting the image stream")
  pipeline = perception_utils.get_camera_pipeline(width=1280, height=720, stream_format="rgb")

  print("Sleeping...")
  time.sleep(5)

  print("Saving images")
  perception_utils.keypress_images(pipeline)

if __name__ == "__main__":
  main()

