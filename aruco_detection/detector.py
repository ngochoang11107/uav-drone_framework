import cv2
import cv2.aruco as aruco
import numpy as np
import time
from geometry_msgs.msg import PoseStamped
# from std_msgs.msg import Float32, Bool
from dataclasses import dataclass #manage Configurations & Parameters

# try:
#     import nanofractal as nf
# except ImportError:
#     nf = None


# from rclpy.qos import QoSProfile, ReliabilityPolicy
# from aruco_detection.calibration import (fit_camera_matrix, load_calibration, parse_size)
# from aruco_detection.transforms import rotmat_to_quat
# from aruco_detection import visualization as viz


@dataclass
class Detection:
    marker_id: int
    corners: np.ndarray      # (4, 2) pixel, thu tu TL, TR, BR, BL
    rvec: np.ndarray         # (3, 1)
    tvec: np.ndarray         # (3, 1), cung don vi voi marker_size              
    # ambiguity: float         # cang gan 1.0 thi goc xoay cang kho tin

def make_detector(dict_id):
            if hasattr(aruco, "ArucoDetector"):
                dictionary = aruco.getPredefinedDictionary(dict_id)
                detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
                return detector.detectMarkers

            dictionary = aruco.Dictionary_get(dict_id)
            parameters = aruco.DetectorParameters_create()
            return lambda gray: aruco.detectMarkers(gray, dictionary, parameters=parameters)

def estimate_pose(corners, marker_size, camera_matrix, dist_coeffs):
            half = marker_size / 2.0
            obj_points = np.array([
                [-half,  half, 0],   # goc 0: tren-trai
                [ half,  half, 0],   # goc 1: tren-phai
                [ half, -half, 0],   # goc 2: duoi-phai
                [-half, -half, 0],   # goc 3: duoi-trai
            ], dtype=np.float32)

            rvecs, tvecs, ratios = [], [], []
            for c in corners:
                img_points = c.reshape(4, 2).astype(np.float32)
                n, rv, tv, errs = cv2.solvePnPGeneric(
                    obj_points, img_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if n == 0:
                    rvecs.append(None)
                    tvecs.append(None)
                    ratios.append(None)
                    continue
                e = [float(np.ravel(x)[0]) for x in errs]
                best = int(np.argmin(e))
                rvecs.append(rv[best])
                tvecs.append(tv[best])
                ratios.append(min(e) / max(e) if n > 1 and max(e) > 0 else 0.0)
            return rvecs, tvecs, ratios



def ssr(gray, sigma=100.0):
        img = gray.astype(np.float32) + 1.0            # +1 tranh log(0)
        cv2.log(img, img)                              # 1) LOG TRUOC (in-place)
        # ksize tinh tu sigma dung cong thuc cua tac gia (bat le bang |1)
        ksize = int(round((sigma - 0.8) / 0.15 + 2.0)) | 1
        blur = cv2.GaussianBlur(img, (ksize, ksize), sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)  # 2) blur log-domain
        retinex = img - blur                           # 3) tru trong log-domain
        out = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
        return out.astype(np.uint8)

class MakerDetector():
    def __init__(self, dict_id, marker_size, camera_matrix, dist_coeffs , detector_type,  sigma=100.0):

        self.dict_id = dict_id
        self.marker_size = marker_size
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.detector_type = detector_type
        self.sigma = sigma
        self.id = 1
        fractal_config="FRACTAL_5L_6"
        self.use_ssr = False


        self.detector_type = str(detector_type).lower()

        # self.detect_width =  854   #640#460
        # self.detect_height =  480  #360#270

        # sx = self.detect_width / 1280.0
        # sy = self.detect_height / 720.0

        # self.camera_matrix_small = scale_camera_matrix(self.camera_matrix, sx, sy)


        # if self.detector_type not in ("aruco", "fractal"):
        #     raise RuntimeError(f"detector_type must be 'aruco' or 'fractal' "f"(got: {detector_type})")
        # self.detect = None
        # self.fractal_detector = None
        # if self.detector_type == "aruco":
        #     if dict_id is None:
        #         raise RuntimeError("dict_id is required for ArUco detection")
        #     self.detect = make_detector(dict_id)
        # else:
        #     if nf is None:
        #         raise RuntimeError(
        #             "Fractal mode requires nanofractal: pip install nanofractal")
        #     try:
        #         self.fractal_detector = nf.FractalDetector(fractal_config,marker_size=marker_size,)
        #     except ValueError as error:
        #         raise RuntimeError(str(error)) from error


        self.detect = make_detector(dict_id)


        # self.fractal_detector = nf.FractalDetector(fractal_config,marker_size=marker_size,)
        # self.detect = make_detector(dict_id)
        

    def process(self, gray, target_id=None):
        image = ssr(gray) if self.use_ssr else gray 
        # if self.detector_type == "fractal":
        #     return self._process_fractal(image, target_id)
        return self._process_aruco(image, target_id)
    
    # def _process_aruco(self, image, target_id):
    #     # ------------------------------------------------
    
    #     corners, ids, _ = self.detect(image)

    #     if ids is None or len(ids) == 0:
    #         return []
        
    #     rvecs, tvecs, ratios = estimate_pose(
    #         corners, self.marker_size, self.camera_matrix_small, self.dist_coeffs)

    #     out = []
    #     for i, marker_id in enumerate(ids.flatten()):
    #         marker_id = int(marker_id)
    #         if target_id is not None and marker_id != target_id:
    #             continue
    #         if rvecs[i] is None:
    #             continue

    #         out.append(Detection(marker_id=marker_id,
    #                              corners=corners[i].reshape(4, 2),
    #                              rvec=rvecs[i],
    #                              tvec=tvecs[i]))
    #                             #  ambiguity=ratios[i]))
    #     return out

    def _process_aruco(self, image, target_id=None):
        t0 = time.perf_counter()
        corners, ids, _ = self.detect(image)
        t1 = time.perf_counter()
        if ids is None or len(ids) == 0:
            return []

        ids_flat = ids.flatten()

        if target_id is not None:

            target_indices = np.where(ids_flat == target_id)[0]

            if len(target_indices) == 0:
                return []

            corners = [corners[i] for i in target_indices]
            ids = ids[target_indices]

        t2 = time.perf_counter()

        rvecs, tvecs, ratios = estimate_pose(corners, self.marker_size, self.camera_matrix,self.dist_coeffs)

        t3 = time.perf_counter()

        print(
            f"[ARUCO DETECT] {(t1-t0)*1000:.1f} ms | "
            f"[PNP] {(t3-t2)*1000:.1f} ms"
        )

        out = []

        for i, marker_id in enumerate(ids.flatten()):

            if rvecs[i] is None:
                continue

            out.append(Detection(marker_id=int(marker_id), corners=np.asarray(corners[i]).reshape(4, 2), rvec=rvecs[i],tvec=tvecs[i]))

        return out
    # --------------------------------------------------------------

    # def _process_fractal(self, image, target_id):
    #     image = np.ascontiguousarray(image, dtype=np.uint8)
    #     result = self.fractal_detector.detect(image,with_inner_points=True)
    #     if result.ids.size == 0:
    #         return []

    #     # FractalDetector represents one composite marker pose. Its detector
    #     # can expose an ID, but there is one pose for the complete composite.
    #     marker_id = int(result.ids[0])
    #     if target_id is not None and marker_id != target_id:
    #         return []

    #     pose = self.fractal_detector.estimate_pose(
    #         result,
    #         np.ascontiguousarray(self.camera_matrix, dtype=np.float64),
    #         np.ascontiguousarray(self.dist_coeffs, dtype=np.float64).reshape(-1),)
    #     if pose is None:
    #         return []

    #     rvec, tvec, reprojection_error = pose
    #     rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    #     tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    #     # tvec = self._smooth_tvec(marker_id, tvec)

    #     return [Detection(
    #         marker_id=marker_id,
    #         corners=result.corners[0].reshape(4, 2),
    #         rvec=rvec,
    #         tvec=tvec
    #         # ambiguity=None,
    #         # reprojection_error=float(reprojection_error),
    #     )]
def scale_camera_matrix(camera_matrix, sx, sy):
    K = camera_matrix.copy().astype(np.float64)
    K[0,0] *= sx
    K[1,1] *= sy
    K[0,2] *= sx
    K[1,2] *= sy

    return K
