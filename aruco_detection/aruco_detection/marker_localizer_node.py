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
from transformation import quat_to_rotmat, marker_yaw_ned, CameraExtrinsic

class MarkerLocalizer(Node):
    def __init__(self):
        super().__init__('marker_localizer_node')
        self.declare_parameter("marker_pose_topic", "/pose_ngoc")
        # self.declare_parameter("marker_pose_topic", "/aruco_pose_node/pose")        #/pose_ngoc 
        self.declare_parameter("px4_attitude_topic", "/fmu/out/vehicle_odometry")

        self.declare_parameter("attitude_buffer_s", 2.0)
        self.declare_parameter("max_attitude_age_s", 0.15)
        self.declare_parameter("mount_roll_deg", 0.0)
        self.declare_parameter("mount_pitch_deg", 0.0)
        self.declare_parameter("mount_yaw_deg", 0.0)
        self.declare_parameter("cam_offset", [0.0, 0.0, 0.0])
        self.declare_parameter("level_frame_id", "drone_ned")
        self.declare_parameter("marker_timeout_s", 0.5)
        self.declare_parameter("map_frame_id", "map")
        self.declare_parameter("attitude_source", "vehicle_attitude")
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

        if self.attitude_source == "vehicle_attitude":
            self.create_subscription(VehicleOdometry, p("px4_attitude_topic").value, self.on_odometry, qos)
            self.get_logger().info( f"Subscribed PX4 VehicleOdometry")
        else:
            raise RuntimeError("????????????????????")

        self.create_subscription(PoseStamped, p("marker_pose_topic").value, self.on_marker_pose, qos)
        self.pub_valid = self.create_publisher(Bool, "~/valid", qos)
        self.pub_rel = self.create_publisher(PointStamped, "/marker_rel_ngoc", qos)      # marker tương đối với drone
        self.pub_pose = self.create_publisher(PoseStamped, "/marker_pose_ngoc", qos)     # marker tuyệt đối trong ENU/map
        self.pub_height = self.create_publisher(Float32, "/height_ngoc", qos)
        self.create_timer(self.marker_timeout / 2.0, self._check_marker_fresh)

        self.get_logger().info(
            f"MarkerLocalizer ENU/FLU | mount rpy=("
            f"{p('mount_roll_deg').value}, {p('mount_pitch_deg').value}, "
            f"{p('mount_yaw_deg').value}) deg | offset={offset} m | "
            f"latency={self.camera_latency * 1000:.0f} ms")

    def px4_timestamp_to_sec(self, timestamp_us): 
        """ PX4 timestamp: uint64 microseconds Convert to seconds. """ 
        return float(timestamp_us) * 1e-6

    def stamp_to_sec(self, stamp):
        return (float(stamp.sec) + float(stamp.nanosec)*1e-9)
    # def sub_callback(self, msg):
    #     print(f"aruco_pose_position: {msg.pose.position}")
    #     print(f"aruco_pose_orientation: {msg.pose.orientation}")

    # def sub_call(self, msg):
    #     print(f"mavros_position:{msg.pose.position}")
    #     print(f"mavros_orientation:{msg.pose.orientation}")
# --------------------
    # def on_mavros_pose(self, msg):
        # q = msg.pose.orientation
        # pos = msg.pose.position
        # self._push_attitude(stamp_to_sec(msg.header.stamp),
        #                     (q.x, q.y, q.z, q.w), 
        #                     (pos.x, pos.y, pos.z))


    def on_odometry(self, msg): #drone trong FRD/NED
        # PX4 q = [w, x, y, z]
        q = msg.q
        self.qw = q[0]
        self.qx = q[1]
        self.qy = q[2]
        self.qz = q[3]

        # quat_to_rotmat() expects [x, y, z, w]
        pos = msg.position
        posx = pos[0]
        posy = pos[1]
        posz = pos[2]

        t = self.px4_timestamp_to_sec(msg.timestamp_sample)
        q_wxyz = (self.qw, self.qx, self.qy, self.qz)
        drone_pose_ned = np.array([posx, posy, posz])
        self._push_attitude(t, q_wxyz, drone_pose_ned)

    def _push_attitude(self, t, q_wxyz, drone_pos_ned):
        self._attitude.append((t, q_wxyz, drone_pos_ned))

        cut_off = t - self.buffer_s
        while self._attitude and self._attitude[0][0] < cut_off:
            self._attitude.popleft()


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

        _, q_wxyz, drone_pos_ned = sample

        # Camera optical ( x: right, y: down, z: forward)
        p_cam = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float)
        #marker -> cam trong he cam optical ( z huong truoc cam, x sang phai, y vao trong nguoi)
        p_frd = self.extrinsic.to_body(p_cam)
        # marker -> cam trong frd
        qw, qx, qy, qz = q_wxyz
        R_ned_from_frd = quat_to_rotmat(qx, qy, qz, qw)
        p_ned_rel = R_ned_from_frd @ p_frd      # marker nam o dau so voi drone theo he ned
        marker_ned = drone_pos_ned + p_ned_rel          # marker nam o dau trong world NED


        print("========== DEBUG NED ==========")
        print("drone_pos_ned =", drone_pos_ned)
        print("p_frd         =", p_frd)
        print("p_ned_rel     =", p_ned_rel)
        print("marker_ned    =", marker_ned)
        print("===============================")



        #drone-> marker trong ENU
        self._last_marker = time.monotonic()

        header_level = msg.header
        header_level.frame_id = self.level_frame_id

        rel = PointStamped()
        rel.header = header_level
        rel.point.x, rel.point.y, rel.point.z = map(float, p_ned_rel)
        self.pub_rel.publish(rel)
        print("toi dayyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
        print("toa do marker tuong doi so voi drone trong ned", p_ned_rel)

        out = PoseStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.map_frame_id
        out.pose.position.x, out.pose.position.y, out.pose.position.z = map(float, marker_ned)

        # out.pose.position.x = float(drone_pos_ned[0] + p_ned_rel[0])
        # out.pose.position.y = float(drone_pos[1] + p_ned_rel[1])
        # out.pose.position.z = float(drone_pos[2] + p_ned_rel[2])
        print("toa do marker trong ned")

        o = msg.pose.orientation
        print("oooooo", o)
        yaw = marker_yaw_ned(quat_to_rotmat(o.x, o.y, o.z, o.w), q_wxyz, self.extrinsic)
        out.pose.orientation.z = math.sin(yaw/2.0)
        out.pose.orientation.w = math.cos(yaw/2.0)

        self.pub_pose.publish(out)      # pub_pose la toa do cua marker trong ned
        self.pub_height.publish(Float32(data=float(-p_ned_rel[2])))
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