import cv2, json
import numpy as np

class Calibrator:
    def __init__(self, checkerboard_dims=(6,9), square_size=19):
        self.checkerboard_dims = checkerboard_dims
        self.square_size = square_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objpoints = []
        self.imgpoints = []

        self.objp = np.zeros((checkerboard_dims[0] * checkerboard_dims[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)
        self.objp *= square_size*1.0

    def process_frame(self, frame, gray_frame):
        ret, corners = cv2.findChessboardCorners(gray_frame, self.checkerboard_dims, None)
        if ret:
            self.corners2 = cv2.cornerSubPix(gray_frame, corners, (11, 11), (-1, -1), self.criteria)            
            frame = cv2.drawChessboardCorners(frame, self.checkerboard_dims, self.corners2 , ret)
        return ret, frame
    
    def save_corners(self):
        self.objpoints.append(self.objp)
        self.imgpoints.append(self.corners2)

    def calibrate(self, image_shape):
        if len(self.objpoints) < 10:
            raise ValueError("Not enough valid frames captured for calibration.")
        
        initial_fx = 1800.0  # Your desired fixed horizontal focal length
        initial_fy = 1800.0  # Let's say you want a square pixel aspect ratio initially
        image_width = 1440
        image_height = 1080
        initial_cx = image_width / 2
        initial_cy = image_height / 2
        initial_camera_matrix = np.array([[initial_fx, 0.0, initial_cx],
                                        [0.0, initial_fy, initial_cy],
                                        [0.0, 0.0, 1.0]])


        flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS |
            cv2.CALIB_FIX_ASPECT_RATIO |
            #cv2.CALIB_FIX_PRINCIPAL_POINT |    # keep (cx, cy) fixed (must be set correctly in K0)
            cv2.CALIB_ZERO_TANGENT_DIST         # p1 = p2 = 0
            # cv2.CALIB_FIX_K1 |                # k1 is fixed
            # cv2.CALIB_FIX_K2 |                # k2 is fixed
            # cv2.CALIB_FIX_K3 |                # k3 is fixed
            # cv2.CALIB_FIX_K4 |                # k4 is fixed
            # cv2.CALIB_FIX_K5 |                # k5 is fixed
            # cv2.CALIB_FIX_K6                  # k6 is fixed
        )

        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints,
            self.imgpoints,
            image_shape,              # (width, height)
            cameraMatrix=initial_camera_matrix,
            distCoeffs=None,
            flags=flags,
            criteria=self.criteria
        )
        return K, dist


def load_calib(path="calib_data_stereo.json"):
    with open(path, "r") as f:
        c = json.load(f)
    K1 = np.array(c["left"]["matrix"])
    D1 = np.array(c["left"]["distortion"])
    K2 = np.array(c["right"]["matrix"])
    D2 = np.array(c["right"]["distortion"])
    R = np.array(c["R"])
    T = np.array(c["T"])
    return K1, D1, K2, D2, R, T

def prepare_undistort(K, D, size_wh, alpha=1.0):
    """Precompute undistort-remap and return K_new, roi, map1, map2."""
    w, h = size_wh
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K_new, (w, h), cv2.CV_16SC2)
    return K_new, roi, map1, map2

def adjust_K_after_crop(P, crop_roi):
    """Adjusts P or K after cropping."""
    x, y, _, _ = crop_roi
    P_adj = P.copy()
    P_adj[0, 2] -= x
    P_adj[1, 2] -= y
    return P_adj

def get_common_roi(roi1: tuple, roi2: tuple) -> tuple:
    """
    Calculates the intersection of two rectangular regions of interest (ROIs).

    Args:
        roi1: The first ROI as a tuple (x, y, w, h).
        roi2: The second ROI as a tuple (x, y, w, h).

    Returns:
        A new tuple (x, y, w, h) representing the common, intersecting ROI.
        Returns a rectangle with zero width or height if there is no overlap.
    """
    # Unpack the coordinates for clarity
    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2

    # Calculate the starting coordinates of the common rectangle
    x_start = max(x1, x2)
    y_start = max(y1, y2)

    # Calculate the ending coordinates of the common rectangle
    x_end = min(x1 + w1, x2 + w2)
    y_end = min(y1 + h1, y2 + h2)

    # Calculate the new width and height. Use max(0, ...) to handle non-overlapping cases.
    common_w = max(0, x_end - x_start)
    common_h = max(0, y_end - y_start)
    
    return (x_start, y_start, common_w, common_h)


def prjMat2K(P):
    """Decomposes a projection matrix P into intrinsic matrix K and extrinsic matrix [R|T]."""
    K = P[:, :3]
    return K
