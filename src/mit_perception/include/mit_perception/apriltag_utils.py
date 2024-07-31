# mostly from https://github.com/clvrai/furniture-bench

import pyrealsense2 as rs
import numpy as np
import cv2
# from dt_apriltags import Detector
from pupil_apriltags import Detector

import matplotlib.pyplot as plt 
import time

import numpy.typing as npt
import argparse
from scipy.spatial.transform import Rotation as R


# TODO: handle pose adjustment for different active pixel ratios?
class AprilTag:
    def __init__(self, tag_size, tag_active_pixel_ratio=1.0):
        self.at_detector = Detector(
            families="tag36h11",
            # families="tagstandard52h13",
            nthreads=4,
            quad_decimate=1.0,
            # quad_sigma=0.0,
            # refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )
        self.tag_size = tag_size
        self.tag_active_pixel_ratio = tag_active_pixel_ratio

    def detect(self, frame, intr_param):
        """Detect AprilTag.

        Args:
            frame: pyrealsense2.frame or Gray-scale image to detect AprilTag.
            intr_param: Camera intrinsics format of [fx, fy, ppx, ppy].
        Returns:
            Detected tags.
        """
        if isinstance(frame, rs.frame):
            frame = np.asanyarray(frame.get_data())
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        tag_active_length = self.tag_size * self.tag_active_pixel_ratio

        detections = self.at_detector.detect(
            frame, True, intr_param, tag_active_length)
        # Filter out bad detections.
        return [detection for detection in detections if detection.hamming < 2] # was 2

    def detect_id(self, frame, intr_param):
        detections = self.detect(frame, intr_param)
        # Make it as a dictionary which the keys are tag_id.
        return {detection.tag_id: detection for detection in detections}


class RealsenseCam:
    def __init__(
        self,
        serial,
        color_res,
        depth_res,
        frame_rate,
        roi=None,
        disable_auto_exposure: bool = False,
    ):
        self.started = False
        self.serial = serial

        if serial is None:
            raise ValueError("camera serial not set!")
            # from furniture_bench.config import config

            # # Find which camera's serial is not set.
            # for i in range(1, 4):
            #     if config["camera"][i]["serial"] is None:
            #         raise ValueError(
            #             f" Camera {i} serial is not set. \n Run export CAM{i}_SERIAL=<serial> before running this script. \n "
            #         )

        self.color_res = color_res
        self.depth_res = depth_res
        self.frame_rate = frame_rate
        # Create a context object. This object owns the handles to all connected realsense devices
        self.pipeline = rs.pipeline()
        # Configure streams
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(
            rs.stream.color, *self.color_res, rs.format.rgb8, self.frame_rate
        )
        config.enable_stream(
            rs.stream.depth, *self.depth_res, rs.format.z16, self.frame_rate
        )
        # Start streaming
        self.roi = roi
        try:
            conf = self.pipeline.start(config)
        except Exception as e:
            print(f"[Error] Could not initialize camera serial: {self.serial}")
            raise e

        self.min_depth = 0.15
        self.max_depth = 2.0
        self.threshold_filter = rs.threshold_filter()
        self.threshold_filter.set_option(rs.option.min_distance, self.min_depth)
        self.threshold_filter.set_option(rs.option.max_distance, self.max_depth)

        # Get intrinsic parameters of color image``.
        profile = conf.get_stream(
            rs.stream.color
        )  # Fetch stream profile for depth stream
        intr_param = (
            profile.as_video_stream_profile().get_intrinsics()
        )  # Downcast to video_stream_profile and fetch intrinsics
        self.intr_param = [intr_param.fx, intr_param.fy, intr_param.ppx, intr_param.ppy]
        self.intr_mat = self._get_intrinsic_matrix()

        # Get the sensor once at the beginning. (Sensor index: 1)
        color_sensor = conf.get_device().first_color_sensor()
        # Set the exposure anytime during the operation
        color_sensor.set_option(rs.option.enable_auto_exposure, True)

        # Set region of interest.
        # color_sensor = conf.get_device().first_roi_sensor()

        if disable_auto_exposure:
            color_sensor.set_option(rs.option.enable_auto_exposure, False)

        if roi is not None:
            # Disable auto exposure.

            # TODO: Fix this.
            roi_sensor = color_sensor.as_roi_sensor()
            roi = roi_sensor.get_region_of_interest()
            roi.min_x, roi.max_x = self.roi[0], self.roi[1]
            roi.min_y, roi.max_y = self.roi[2], self.roi[3]
            # https://github.com/IntelRealSense/librealsense/issues/8004
            roi_success = False
            for _ in range(5):
                try:
                    roi_sensor.set_region_of_interest(roi)
                except:
                    time.sleep(0.1)
                    pass
                else:
                    roi_success = True
                    break
            if not roi_success:
                print("Could not set camera ROI.")

        for _ in range(10):
            # Read dummy observation to setup exposure.
            self.get_frame()
            time.sleep(0.04)

        self.started = True

    def get_frame(self):
        """Read frame from the realsense camera.

        Returns:
            Tuple of color and depth image. Return None if failed to read frame.

            color frame:(height, width, 3) RGB uint8 realsense2.video_frame.
            depth frame:(height, width) z16 realsense2.depth_frame.
        """
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        depth_frame = self.threshold_filter.process(depth_frame)

        if not color_frame or not depth_frame:
            return None, None
        return color_frame, depth_frame

    def get_image(self):
        """Get numpy color and depth image.

        Returns:
            Tuble of numpy color and depth image. Return (None, None) if failed.

            color image: (height, width, 3) RGB uint8 numpy array.
            depth image: (height, width) z16 numpy array.
        """
        color_frame, depth_frame = self.get_frame()
        if color_frame is None or depth_frame is None:
            return None, None
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data()).copy()
        depth_image = np.asanyarray(depth_frame.get_data()).copy()

        return color_image, depth_image

    def _get_intrinsic_matrix(self):
        m = np.zeros((3, 3))
        m[0, 0] = self.intr_param[0]
        m[1, 1] = self.intr_param[1]
        m[0, 2] = self.intr_param[2]
        m[1, 2] = self.intr_param[3]
        return m

    def __del__(self):
        if self.started:
            self.pipeline.stop()


def frame_to_image(color_frame, depth_frame):
    color_image = np.asanyarray(color_frame.get_data()).copy()
    depth_image = np.asanyarray(depth_frame.get_data()).copy()

    return color_image, depth_image


def read_detect(
    april_tag: AprilTag, cam1: RealsenseCam): #, cam2: RealsenseCam, cam3: RealsenseCam):
    color_img1, depth_img1 = cam1.get_image()
    # color_img2, depth_img2 = cam2.get_image()
    # color_img3, depth_img3 = cam3.get_image()
    tags1 = april_tag.detect_id(color_img1, cam1.intr_param)
    # tags2 = april_tag.detect_id(color_img2, cam2.intr_param)
    # tags3 = april_tag.detect_id(color_img3, cam3.intr_param)

    return (
        color_img1,
        depth_img1,
        # color_img2,
        # depth_img2,
        # color_img3,
        # depth_img3,
        tags1,
        # tags2,
        # tags3,
    )

def draw_tags(
    color_image: npt.NDArray[np.uint8], cam: RealsenseCam, tags
) -> npt.NDArray[np.uint8]:
    draw_img = color_image.copy()
    for tag in tags:
        if tag is None:
            continue
        # print tag
        #print(tag)

        # Draw boarder of the tag.
        draw_img = draw_bbox(draw_img, tag)
        # Draw x, y, z axis on the image.
        draw_img = draw_axis(draw_img, tag.pose_R, tag.pose_t, cam.intr_mat).copy()

        # Draw id next to the tag.
        draw_img = cv2.putText(
            draw_img,
            str(tag.tag_id),
            org=(
                tag.corners[0, 0].astype(int) + 10,
                tag.corners[0, 1].astype(int) + 10,
            ),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(0, 0, 255),
        )
    return draw_img


def draw_bbox(color_image: npt.NDArray[np.uint8], tag):
    if tag is None:
        return color_image
    draw_img = color_image.copy()
    for idx in range(len(tag.corners)):
        draw_img = cv2.line(
            draw_img,
            tuple(tag.corners[idx - 1, :].astype(int)),
            tuple(tag.corners[idx, :].astype(int)),
            (0, 255, 0),
        )
    return draw_img


def draw_axis(
    img: npt.NDArray[np.uint8],
    R: npt.NDArray[np.float32],
    t: npt.NDArray[np.float32],
    K: npt.NDArray[np.float32],
    s: float = 0.015,
    d: int = 3,
    rgb=True,
    axis="xyz",
    colors=None,
    trans=False,
    text_label: bool = False,
    draw_arrow: bool = False,
) -> npt.NDArray[np.uint8]:
    """Draw x, y, z axis on the image.

    Args:
        img: Image to draw on.
        R: Rotation matrix.
        t: Translation vector.
        K: Intrinsic matrix.
        s: Length of the axis.
        d: Thickness of the axis.


    Returns:
        Image with the axis drawn.
    """
    draw_img = img.copy()
    # Unit is m
    rotV, _ = cv2.Rodrigues(R)
    # The tag's coordinate frame is centered at the center of the tag,
    # with x-axis to the right, y-axis down, and z-axis into the tag.
    if isinstance(s, float):
        points = np.float32([[s, 0, 0], [0, s, 0], [0, 0, s], [0, 0, 0]]).reshape(-1, 3)
    else:
        # list
        points = np.float32(
            [[s[0], 0, 0], [0, s[1], 0], [0, 0, s[2]], [0, 0, 0]]
        ).reshape(-1, 3)

    axis_points, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    a0 = np.array((int(axis_points[0][0][0]), int(axis_points[0][0][1])))
    a1 = np.array((int(axis_points[1][0][0]), int(axis_points[1][0][1])))
    a2 = np.array((int(axis_points[2][0][0]), int(axis_points[2][0][1])))
    a3 = np.array((int(axis_points[3][0][0]), int(axis_points[3][0][1])))
    if colors is None:
        if rgb:
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        else:
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    else:
        colors = [colors] * 3

    axes_map = {"x": (a0, colors[0]), "z": (a2, colors[2]), "y": (a1, colors[1])}

    for axis_label, (point, color) in axes_map.items():
        if axis_label in axis:
            if draw_arrow:
                draw_img = cv2.arrowedLine(
                    draw_img, tuple(a3), tuple(point), color, d, tipLength=0.5
                )
            else:
                draw_img = cv2.line(draw_img, tuple(a3), tuple(point), color, d)

    # Add labels for each axis
    if text_label:
        cv2.putText(
            draw_img, "X", tuple(a0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[0], 3
        )
        cv2.putText(
            draw_img, "Y", tuple(a2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[1], 3
        )
        cv2.putText(
            draw_img, "Z", tuple(a1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[2], 3
        )

    if trans:
        # Transparency value
        alpha = 0.50
        # Perform weighted addition of the input image and the overlay
        draw_img = cv2.addWeighted(draw_img, alpha, img, 1 - alpha, 0)

    return draw_img
    

def detect_draw(april_tag, cam: RealsenseCam):
    color_frame, depth_frame = cam.get_frame()
    depth_image = np.asanyarray(depth_frame.get_data()).copy()

    img = np.asanyarray(color_frame.get_data()).copy()

    tags = april_tag.detect(color_frame, cam.intr_param)
    for tag in tags:
        if tag.tag_id==168:
            p_global_drone_global, R_global_drone, euler_angles = world_pose_estimate(tag)
            print('tag: ', tag.tag_id)
            print('angles: ', p_global_drone_global)

    #for tag in tags:
    #    p_global_drone_global, R_global_drone, euler_angles = pose_estimate(tag)
    # Visualize tags.
    draw_image = draw_tags(img.copy(), cam, tags)

    return draw_image, depth_image

def detect_tags(april_tag, cam: RealsenseCam, return_imgs=False):
    color_frame, depth_frame = cam.get_frame()
    tags = april_tag.detect(color_frame, cam.intr_param)

    if return_imgs:
        depth_image = np.asanyarray(depth_frame.get_data()).copy()
        img = np.asanyarray(color_frame.get_data()).copy()
        draw_image = draw_tags(img.copy(), cam, tags)
        return tags, draw_image, depth_image
    else:
        return tags

def main():

    # Define arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-depth", action="store_true", help="Show depth image.")
    args = parser.parse_args()


    cam1 = RealsenseCam(
        "317422075665",
        (1280, 720), # color img size
        (1280, 720), # depth img size
        30 # frame rate
    )
    april_tag = AprilTag(tag_size=0.0195)

    cv2.namedWindow("RealsenseAprilTag", cv2.WINDOW_AUTOSIZE)

    while True:
        color_img, depth_image = detect_draw(april_tag, cam1)
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

        # img2 = cv2.cvtColor(color_img2, cv2.COLOR_RGB2BGR)

        k = cv2.waitKey(1)
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()