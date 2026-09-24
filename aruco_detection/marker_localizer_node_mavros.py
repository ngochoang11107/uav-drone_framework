#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import time
import numpy as np
from px4_msgs. msg import VehicleOdometry
from geometry_msgs.msg import PoseStamped, PointStamped
# from sensor_msgs.msg import Imu
from rclpy.qos import QoSProfile, ReliabilityPolicy
from collections import deque
from std_msgs.msg import Bool, Float32
import math
from transformation import quat_to_rotmat, marker_yaw_enu, CameraExtrinsic

class MarkerLocalizer(Node):
    def __init__(self):
        super().__init__('marker_localizer_node')
        self.declare_parameter("marker_pose_topic", "/pose_ngoc")
        self.declare_parameter("marker_pose_topic", "/aruco_pose_node/pose")        #/pose_ngoc 
        self.declare_parameter("local_pose_topic", "/mavros/local_position/pose")
        self.declare_parameter("imu_topic", "/mavros/imu/data")

        self.declare_parameter("attitude_buffer_s", 2.0)
        self.declare_parameter("max_attitude_age_s", 0.15)
        self.declare_parameter("mount_roll_deg", 0.0)
        self.declare_parameter("mount_pitch_deg", 0.0)
        self.declare_parameter("mount_yaw_deg", 0.0)
        self.declare_parameter("cam_offset", [0.0, 0.0, 0.0])
        self.declare_parameter("level_frame_id", "base_link_level")
        self.declare_parameter("marker_timeout_s", 0.5)
        self.declare_parameter("map_frame_id", "map")
        self.declare_parameter("attitude_source", "imu") #imu, local_position 
        self.declare_parameter("camera_latency_s", 0.0)

        p = self.get_parameter
        self.buffer_s = float(p("attitude_buffer_s").value)
        self.max_attitude_age = float(p("max_attitude_age_s").value)
        self.level_frame_id = p("level_frame_id").value
        self.marker_timeout = float(p("marker_timeout_s").value)
        self.map_frame_id = p("map_frame_id").value
        self.attitude_source = str(p("attitude_source").value).lower()
        self.camera_latency = float(p("camera_latency_s").value)
        self._attitude = deque()
        self._last_marker = None
        

        offset = list(p("cam_offset").value)
        self.extrinsic = CameraExtrinsic(mount_roll=float(p("mount_roll_deg").value),
                                         mount_pitch=float(p("mount_pitch_deg").value),
                                         mount_yaw=float(p("mount_yaw_deg").value), offset=offset)

        qos = QoSProfile(depth=1, reliability = (ReliabilityPolicy.BEST_EFFORT))
        # self.sub1 = self.create_subscription(PoseStamped, "/pose_ngoc", self.sub_callback, qos)
        # self.sub2 = self.create_subscription(PoseStamped, "/mavros/local_position/pose", self.on_mavros_pose, qos)
        
        if self.attitude_source == "imu":
            self.create_subscription(VehicleOdometry, p("imu_topic").value, self.on_imu, qos)
            self.get_logger().warn( f"attitude_source=attitude_imu" ({p('attitude_topic').value}))
        elif self.attitude_source == "local_position":
            self.create_subscription(VehicleOdometry, p("local_pose_topic").value, self.on_mavros_pose, qos)
        else:
            raise RuntimeError("attitude_source phai la 'local_position' hoac " f"'imu' (nhan duoc: {self.attitude_source})")

        self.create_subscription(PoseStamped, p("marker_pose_topic").value, self.on_marker_pose, qos)
        self.pub_valid = self.create_publisher(Bool, "~/valid", qos)
        self.pub_rel = self.create_publisher(PointStamped, "/marker_rel", qos)      # marker tương đối với drone
        self.pub_pose = self.create_publisher(PoseStamped, "/marker_pose", qos)     # marker tuyệt đối trong ENU/map
        self.pub_height = self.create_publisher(Float32, "/height", qos)
        self.create_timer(self.marker_timeout / 2.0, self._check_marker_fresh)

        self.get_logger().info(
            f"MarkerLocalizer ENU/FLU | mount rpy=("
            f"{p('mount_roll_deg').value}, {p('mount_pitch_deg').value}, "
            f"{p('mount_yaw_deg').value}) deg | offset={offset} m | "
            f"latency={self.camera_latency * 1000:.0f} ms")

    def stamp_to_sec(self, stamp):
        return stamp.sec + stamp.nanosec * 1e-9
    
    # def sub_callback(self, msg):
    #     print(f"aruco_pose_position: {msg.pose.position}")
    #     print(f"aruco_pose_orientation: {msg.pose.orientation}")

    # def sub_call(self, msg):
    #     print(f"mavros_position:{msg.pose.position}")
    #     print(f"mavros_orientation:{msg.pose.orientation}")
# --------------------
    def on_mavros_pose(self, msg):
        q = msg.pose.orientation
        pos = msg.pose.position
        self._push_attitude(self.stamp_to_sec(msg.header.stamp),
                            (q.x, q.y, q.z, q.w), 
                            (pos.x, pos.y, pos.z))

    def _push_attitude(self, t, q_xyzw, drone_pos):
        self._attitude.append((t, q_xyzw, drone_pos))

        cut_off = t - self.buffer_s
        while self._attitude and self._attitude[0][0] < cut_off:
            self._attitude.popleft()

    def on_imu(self, msg):
        q = msg.orientation
        self._push_attitude(self.stamp_to_sec(msg.header.stamp),(q.x, q.y, q.z, q.w), (0.0, 0.0, 0.0))

    def _lookup(self, t):
        if not self._attitude:
            return None
        best = min(self._attitude, key=lambda s: abs(s[0] - t))
        return best if abs(best[0] - t) <= self.max_attitude_age else None


    def on_marker_pose(self, msg):
        t_capture = self.stamp_to_sec(msg.header.stamp) #- self.camera_latency
        sample = self._lookup(t_capture)
        if sample is None:
            self.pub_valid.publish(Bool(data=False))
            self.get_logger().warn("Khong co attitude khop timestamp -> bo pose " f"(buffer {len(self._attitude)} mau)",
                                   throttle_duration_sec = 2.0)
            return

        _, q_xyzw, drone_pos = sample

        t = msg.pose.position
        p_cam = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float)
        #marker -> cam trong he cam optical ( z huong truoc cam, x sang phai, y vao trong nguoi)
        p_body_flu = (self.extrinsic.R_flu_from_cam @ p_cam + self.extrinsic.offset)
        #drone -> marker trong FLU
        R_enu_from_flu = quat_to_rotmat(*q_xyzw)
        p_rel = R_enu_from_flu @ p_body_flu
        #drone-> marker trong ENU
        self._last_marker = time.monotonic()

        header_level = msg.header
        header_level.frame_id = self.level_frame_id

        rel = PointStamped()
        rel.header = header_level
        rel.point.x, rel.point.y, rel.point.z = map(float, p_rel)
        self.pub_rel.publish(rel)
        print("toi dayyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
        print("toa do marker tuong doi voi drone", p_rel)

        out = PoseStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.map_frame_id
        out.pose.position.x = float(drone_pos[0] + p_rel[0])
        out.pose.position.y = float(drone_pos[1] + p_rel[1])
        out.pose.position.z = float(drone_pos[2] + p_rel[2])
        print("out.pose.position.x", out.pose.position.x)

        o = msg.pose.orientation
        print("oooooo", o)
        yaw = marker_yaw_enu(quat_to_rotmat(o.x, o.y, o.z, o.w), q_xyzw, self.extrinsic)
        out.pose.orientation.z = math.sin(yaw/2.0)
        out.pose.orientation.w = math.cos(yaw/2.0)
        self.pub_pose.publish(out)
    # pub_pose la toa do cua marker trong enu
        # print("toi goi")
        # print("out", out)
        self.pub_height.publish(Float32(data=float(-p_rel[2])))
        #do 
        self.pub_valid.publish(Bool(data=True))

    def _check_marker_fresh(self):
        if self._last_marker is None:
            return
        if time.monotonic() - self._last_marker > self.marker_timeout:
            self.pub_valid.publish(Bool(data=False))
            self.get_logger().warn("Mat marker", throttle_duration_sec=2.0)
        
def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MarkerLocalizer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f"[marker_localizer_node] {e}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()