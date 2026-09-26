#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import os
import cv2
import cv2.aruco as aruco
import numpy as np
import time
import calibration
import detector
import transformation
import load_camera
import visualization as viz
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, PointStamped
from skydroid_msgs.msg import GimbalCommand
from rclpy.qos import QoSProfile, ReliabilityPolicy
import threading


package_share = get_package_share_directory("aruco_detection")

DEFAULT_CALIB = os.path.join(package_share, "camera_calibration", 'data',"elp210", "calib_data_left.json")
DEFAULT_DICT = "DICT_4X4_50"
DEFAULT_MARKER_SIZE = 0.267
# DEFAULT_MARKER_SIZE = 26.7



class Aruco_Pose(Node): 
    def __init__(self):
        super().__init__('pose_estimation_node')
        self.declare_parameter("marker_size", DEFAULT_MARKER_SIZE)
        self.declare_parameter("cam", 2)
        self.declare_parameter("id", -1)
        self.declare_parameter("target_id", 1)
        self.declare_parameter("calib", DEFAULT_CALIB)
        self.declare_parameter("side", "left")
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("detector_type", "aruco")   # fractal  aruco
        self.declare_parameter("fractal_config", "FRACTAL_5L_6")
        self.declare_parameter("calib_size", [854,480])
        self.declare_parameter("threaded_capture", True)
        
        self.marker_size = self.get_parameter("marker_size").get_parameter_value().double_value
        self.cam = self.get_parameter("cam").value
        self.id = self.get_parameter("id").get_parameter_value().integer_value
        self.calib = self.get_parameter("calib").get_parameter_value().string_value
        self.side = self.get_parameter("side").get_parameter_value().string_value
        self.publish_debug_image = self.get_parameter("publish_debug_image").value
        self.detector_type = self.get_parameter("detector_type").value
        self.target_id = self.get_parameter("target_id").value
        self.fractal_config = self.get_parameter("fractal_config").value
        self.calib_size = self.get_parameter("calib_size").value
        self.threaded_capture = self.get_parameter("threaded_capture").value




        
        self.url = "rtsp://192.168.144.108:554/stream=1"
        self.bridge = CvBridge()
        self.frame_id = "ngoc_camera"
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        qos = QoSProfile(depth=1, reliability = (ReliabilityPolicy.BEST_EFFORT))
        
        self.gimbal_timer = self.create_timer(1.0, self.start_gimbal)        
        self.gimbal_pub = self.create_publisher(GimbalCommand, "/gimbal_command", 10)
        self.pub_image = self.create_publisher(Image, "/image_ngoc", qos)
        # self.pub_pose = self.create_publisher(PoseStamped, "/aruco_pose_node/pose", qos)  
        self.pub_pose = self.create_publisher(PoseStamped, "/pose_ngoc", qos)     #~/pose
        self.sub_img = self.create_subscription(Image, '/skydroid/rgb/image_raw', self.image_callback, qos)



        self.camera_matrix, self.dist_coeffs = calibration.load_calibration(self.calib, self.side)
        print("===== CALIBRATION K =====")
        print(self.camera_matrix)
        print("=========================")
        self.detector_type = str(self.detector_type).lower()
        if self.detector_type not in ("aruco", "fractal"):
            raise RuntimeError(
                f"detector_type must be 'aruco' or 'fractal' "
                f"(got: {self.detector_type})"
            )
        target = self.target_id
        self.target_id = None if target < 0 else target

        dict_name = DEFAULT_DICT
        if self.detector_type == "aruco" and not hasattr(aruco, dict_name):
            raise RuntimeError(f"Unknown dictionary: {dict_name}")
        
        # dict_id = getattr(aruco, dict_name) if self.detector_type == "aruco" else None

        self.detector = detector.MakerDetector(dict_id = getattr(aruco, DEFAULT_DICT),
                                            marker_size = self.marker_size,
                                            camera_matrix = self.camera_matrix,
                                            dist_coeffs = self.dist_coeffs,
                                            detector_type=self.detector_type ,
                                            sigma = 100)

        # if not self.cap.isOpened():
        #     raise RuntimeError(f"Khong mo duoc camera {self.cam}")
        

        self.n_frame = 0
        self.n_detected = 0
        self._last_cb = None
        self._fps = 0.0
        # Tạo timer
        
        marker_family = (dict_name if self.detector_type == "aruco"
                         else "fractal_config")
        self.get_logger().info(f"Ready: {self.detector_type}, marker "
                               f"{self.marker_size * 1:.1f}m, {marker_family}, "
                               f"target_id={self.target_id}, frame_id '{self.frame_id}'")
        # self.camera_matrix, self.dist_coeffs = calibration.load_calibration(
        #     DEFAULT_CALIB, self.side)
        
        frame_size = self._open_camera(self.url, self.calib_size, self.threaded_capture)
        self.timer = self.create_timer(1/30, self.timer_callback)


    def tilt_down(self):
            msg = GimbalCommand()
            msg.control_mode = 0
            msg.mode = 0
            msg.ptz_cmd = 2
            self.gimbal_pub.publish(msg)
            self.get_logger().info('ptz_titl_down')
    
    def start_gimbal(self):
        self.tilt_down()

        self.gimbal_timer.cancel()

        self.get_logger().info("Gimbal initial tilt down sent")
    
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

            with self.frame_lock:
                self.latest_frame = frame
                self.latest_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f"CvBridge error: {e}")

    def timer_callback(self):
       # ok, hello = self.cam.read()

        # if not ok:
        #     return

        t0 = time.perf_counter()

        with self.frame_lock:
            if self.latest_frame is None:
                return
            frame = self.latest_frame.copy()
            stamp = self.latest_stamp


        # self.bridge = None
        # if self.publish_debug_image:
        #     if CvBridge is None:
        #         self.get_logger().warn("cv_bridge missing -> publish_debug_image disabled")
        #         self.publish_debug_image = False
        #     else:
        #         from sensor_msgs.msg import Image
        #         self.bridge = CvBridge()
        #         self.pub_image = self.create_publisher(Image, "/image_ngoc", 10)


        self.n_frame += 1

        if self._last_cb is not None:
            dt = t0 - self._last_cb
            if dt > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps else 1.0 / dt
        self._last_cb = t0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t_detect0 = time.perf_counter()
    
        detections = self.detector.process(
            gray,
            self.target_id
        )

        detect_ms = (time.perf_counter() - t_detect0) * 1000

        print(
    f"[PERF] detect={detect_ms:.1f} ms | "
    f"callback_fps={self._fps:.1f}"
    f"frame ={frame.shape}"
    f"K //////"
    f"{self.detector.camera_matrix}"

)
        

        if detections:
            self.n_detected +=1
            det = detections[0]

            print("det", det)
            print("detection", detections)
            self._publish_pose(stamp, det)

        if self.publish_debug_image:
            self._publish_debug(stamp, frame, detections, t0, detect_ms )

    def _open_camera(self, source, calib_size, threaded):

        self.cam = load_camera.open_camera(source, width=calib_size[0], 
                                           height=calib_size[1],threaded=threaded)




        
        self.get_logger().info(f"Camera: {self.cam}")

        # The requested resolution is only a request: cap.set() fails silently
        # on many USB cameras. Compare against what actually came back, because
        # a mismatch makes every distance wrong by the resolution ratio.
        frame_size = self.cam.resolution
        if frame_size != calib_size:
            self.get_logger().warn(f"Camera returns {frame_size[0]}x{frame_size[1]} instead of the "
                f"calibrated {calib_size[0]}x{calib_size[1]}")
            # self.camera_matrix = calibration.fit_camera_matrix(self.camera_matrix,calib_size,frame_size)
        return frame_size

    
    def _publish_pose(self, stamp, det):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        t = det.tvec.ravel()
        msg.pose.position.x = float(t[0])
        msg.pose.position.y = float(t[1])
        msg.pose.position.z = float(t[2])

        print("x,y,z", (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z))
        
        qx, qy, qz, qw = transformation.rotmat_to_quat(cv2.Rodrigues(det.rvec)[0])
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.pub_pose.publish(msg)

    def _publish_debug(self, stamp, frame, detections, t0, detect_ms):


        y = 30
        if detections:
            for det in detections:
                y = viz.draw_detection(frame, det, self.camera_matrix,
                                        self.dist_coeffs, self.marker_size, y=y)
                
        else:
            viz.draw_no_marker(frame, y)
        frame_ms = (time.perf_counter() - t0) * 1000
        viz.draw_stats(frame, fps=self._fps, frame_ms=frame_ms,
                       detect_ms=detect_ms, detected=self.n_detected, total=self.n_frame)

        viz.draw_markers(frame, detections)
        viz.draw_debug_origins(frame, self.camera_matrix, detections)


        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id #frame_id o day la 1 cai ten giup phan biet giua cac camera cua 1 drone 
        self.pub_image.publish(msg)

        viz.draw_markers(frame, detections)
        viz.draw_debug_origins(frame, self.camera_matrix, detections)

        print("DEBUG IMAGE:", frame.shape)
        print("DEBUG K:")
        print(self.camera_matrix)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Aruco_Pose()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f"[aruco_pose_node] {e}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
