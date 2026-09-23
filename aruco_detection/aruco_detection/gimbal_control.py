#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from skydroid_msgs.msg import GimbalCommand

class GimbalController(Node):
    def __init__(self):
        super().__init__('gimbal_controller')
        self.gimbal_pub = self.create_publisher(GimbalCommand, '/gimbal_command', 10)
        self.get_logger().info('okee ')
        self.center()
     #ptz: pan-tilt-zoom

    def stop(self):
        msg = GimbalCommand()
        msg.control_mode = 0
        msg.mode = 0
        msg.ptz_cmd = 0
        self.gimbal_pub.publish(msg)
        self.get_logger().info('ptz_stop')

    def tilt_up(self):
        msg = GimbalCommand()
        msg.control_mode = 0
        msg.mode = 0
        msg.ptz_cmd = 1
        self.gimbal_pub.publish(msg)
        self.get_logger().info('ptz_titl_up')

    def tilt_down(self):
        msg = GimbalCommand()
        msg.control_mode = 0
        msg.mode = 0
        msg.ptz_cmd = 2
        self.gimbal_pub.publish(msg)
        self.get_logger().info('ptz_titl_down')

    def center(self):
        msg = GimbalCommand()
        msg.control_mode = 0
        msg.mode = 0
        msg.ptz_cmd = 3
        self.gimbal_pub.publish(msg)
        self.get_logger().info('ptz_center')

    def follow(self):
        msg = GimbalCommand()
        msg.control_mode = 0
        msg.mode = 0
        msg.ptz_cmd = 4
        self.gimbal_pub.publish(msg)
        self.get_logger().info('ptz_follow')

    def lock(self):
        msg = GimbalCommand()
        msg.control_mode = 0
        msg.mode = 0
        msg.ptz_cmd = 5
        self.gimbal_pub.publish(msg)
        self.get_logger().info('ptz_lock')

    #position

    def pose_pitch_yaw(self, pitch, yaw, pitch_speed, yaw_speed):

        msg = GimbalCommand()
        msg.control_mode = 0
        msg.mode = 1
        msg.enable_pitch = True
        msg.enable_yaw = True
        msg.pitch_deg = pitch
        msg.yaw_deg = yaw
        msg.pitch_speed_dps = pitch_speed
        msg.yaw_speed_dps = yaw_speed
        self.gimbal_pub.publish(msg)
        self.get_logger().info('pose_pitch_yaw')

    def pose_pitch(self, pitch, yaw, pitch_speed, yaw_speed):

        msg = GimbalCommand()
        msg.control_mode = 0
        msg.mode = 1
        msg.enable_pitch = True
        msg.enable_yaw = False
        msg.pitch_deg = pitch
        msg.yaw_deg = yaw
        msg.pitch_speed_dps = pitch_speed
        msg.yaw_speed_dps = yaw_speed
        self.gimbal_pub.publish(msg)
        self.get_logger().info('pose_pitch')

    def pose_yaw(self, pitch, yaw, pitch_speed, yaw_speed):

        msg = GimbalCommand()
        msg.control_mode = 0
        msg.mode = 1
        msg.enable_pitch = False
        msg.enable_yaw = True
        msg.pitch_deg = pitch
        msg.yaw_deg = yaw
        msg.pitch_speed_dps = pitch_speed
        msg.yaw_speed_dps = yaw_speed
        self.gimbal_pub.publish(msg)
        self.get_logger().info('pose_yaw')

    def pose_home(self, pitch, yaw, pitch_speed, yaw_speed):
    
        msg = GimbalCommand()
        msg.control_mode = 0
        msg.mode = 1
        msg.enable_pitch = True
        msg.enable_yaw = True
        msg.pitch_deg = pitch
        msg.yaw_deg = yaw
        msg.pitch_speed_dps = pitch_speed
        msg.yaw_speed_dps = yaw_speed
        self.gimbal_pub.publish(msg)
        self.get_logger().info('pose_home')

    def vel_pitch(self, pitch_vel, yaw_vel):
        msg = GimbalCommand()
        msg.control_mode = 0
        msg.mode = 2
        msg.pitch_vel_dps = pitch_vel
        msg.yaw_vel_dps = yaw_vel
        self.gimbal_pub.publish(msg)
        self.get_logger().info('vel_pitch')

    def vel_stop(self, pitch_vel, yaw_vel):
        msg = GimbalCommand()
        msg.control_mode = 0
        msg.mode = 2
        msg.pitch_vel_dps = pitch_vel
        msg.yaw_vel_dps = yaw_vel
        self.gimbal_pub.publish(msg)
        self.get_logger().info('vel_pitch')

    def manual(self):
        msg = GimbalCommand()
        msg.control_mode = 0
        msg.mode = 0
        msg.ptz_cmd = 3
        self.gimbal_pub.publish(msg)
        self.get_logger().info('manual')

    def auto(self, pitch, yaw ):
        msg = GimbalCommand()
        msg.control_mode = 1
        msg.mode = 1
        msg.enable_pitch = True
        msg.enable_yaw = True
        msg.pitch_deg = pitch
        msg.yaw_deg = yaw
        self.gimbal_pub.publish(msg)
        self.get_logger().info('auto')

def main(args=None):
    rclpy.init(args=args)
    node = GimbalController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()