import math
import numpy as np


def rotmat_to_quat(R):
    """Tra ve (x, y, z, w) theo thu tu cua geometry_msgs/Quaternion."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return x, y, z, w

def quat_to_rotmat(x, y, z, w):
    """(x, y, z, w) -> 3x3. Thu tu geometry_msgs, KHONG phai px4_msgs."""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])

def euler_to_rotmat(roll, pitch, yaw):
    """Z-Y-X noi tai, don vi DO."""
    r, p, y_ = math.radians(roll), math.radians(pitch), math.radians(yaw)
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y_), math.sin(y_))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])

def rotation_to_euler(rmat):
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        if sy > 1e-6:
            roll = math.atan2(rmat[2, 1], rmat[2, 2])
            pitch = math.atan2(-rmat[2, 0], sy)
            yaw = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            roll = math.atan2(-rmat[1, 2], rmat[1, 1])
            pitch = math.atan2(-rmat[2, 0], sy)
            yaw = 0.0
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

# # x: phai, y: xuong, z: ra truoc
# def camera_in_marker(rmat, tvec):
#     return(-rmat.T @ np.asarray(tvec).reshape(3, 1)).ravel()

# # MAVROS: 
# # ENU: x: EAST, Y: NORTH, Z: UP
# # FLU: x: Forward, y: Left, Z: UP
# # quaternion   : [x, y, z, w]
# R_FLU_FROM_CAM_DOWN = np.array([[0.0, -1.0,  0.0],
#                                 [-1.0, 0.0,  0.0],
#                                 [0.0,  0.0, -1.0]])


def camera_in_marker(rmat, tvec):
    return(-rmat.T @ np.asarray(tvec).reshape(3, 1)).ravel()

# FMU: 
# FRD: x: Front, Y: Right, Z: Down
# NED: x: North, y: East, Z: Down
# quaternion   : [w, x, y, z]
R_FRD_FROM_CAM_DOWN = np.array([[0.0,  1.0, 0.0],
                                [-1.0, 0.0, 0.0],
                                [0.0,  0.0, 1.0]])

class CameraExtrinsic:

    def __init__(self, mount_roll=0.0, mount_pitch=0.0, mount_yaw =0.0, offset=(0.0, 0.0, 0.0)):
        self.R_frd_from_cam = (euler_to_rotmat(mount_roll, mount_pitch, mount_yaw)@ R_FRD_FROM_CAM_DOWN)
        self.offset = np.asarray(offset, dtype=float).reshape(3)

    def to_body(self, p_cam):
        p_cam = np.asarray(p_cam, dtype=float).reshape(3)
        return self.R_frd_from_cam @ p_cam + self.offset

def marker_yaw_ned(R_cam_from_marker, q_wxyz, extrinsic):

    # PX4 q = [w, x, y, z]
    # orientation/rotation của drone dưới dạng quaternion trong frd.
    qw, qx, qy, qz = q_wxyz

    # FRD -> NED
    # rotation matrix tương ứng với quaternion giup chuyen vector frd sang ned
    R_ned_from_frd = quat_to_rotmat(qx, qy, qz, qw)


    # Camera -> FRD
    # rotation cố định của camera so với drone
    R_frd_from_cam = extrinsic.R_frd_from_cam

    # Camera -> NED
    # phep rotation chuyen camera optical sang ned
    R_ned_from_cam = (R_ned_from_frd @ R_frd_from_cam)

    # Marker -> NED
    # R_cam_from_marker rotation sinh ra tu ArUco/PnP
    # Rotation của frame Marker sang frame NED
    R_ned_from_marker = (R_ned_from_cam @ R_cam_from_marker)

    # yaw trong NED
    '''
    R_ned_from_marker
    =
    R_ned_from_frd
    x R_frd_from_cam
    x R_cam_from_marker
    '''
    # lấy yaw orientation của marker trong hệ NED từ rotation matrix R_ned_from_marker.
    return math.atan2(
        R_ned_from_marker[1, 0],
        R_ned_from_marker[0, 0]
    )
def marker_relative_ned(p_cam, q_wxyz, extrinsic):
    # Camera Optical -> FRD
    p_frd = extrinsic.to_body(p_cam)
    # PX4 q = [w, x, y, z]
    qw, qx, qy, qz = q_wxyz
    R_ned_from_frd = quat_to_rotmat(qx,qy,qz,qw)
    p_ned = R_ned_from_frd @ p_frd

    return p_ned, R_ned_from_frd