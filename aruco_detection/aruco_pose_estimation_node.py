#!/usr/bin/env python3
"""Detect ArUco marker tu webcam va uoc luong pose (rvec/tvec) so voi camera.

Chay:
    python3 aruco_pose_estimation.py --marker-size 26.7
    python3 aruco_pose_estimation.py --cam 2 --id 1

Nhan 'q' de thoat.
"""
import rclpy
from rclpy.node import Node
import json
import math
import os
import cv2
import cv2.aruco as aruco
import numpy as np
from ament_index_python.packages import get_package_share_directory

package_share = get_package_share_directory("aruco_detection")

DEFAULT_CALIB = os.path.join(package_share, "camera_calibration", "data", "elp210", "calib_data_left.json")
DEFAULT_DICT = "DICT_4X4_50"
DEFAULT_MARKER_SIZE = 26.7


class Pose_estimation(Node): 

    def __init__(self):
        super().__init__('pose_estimation_node')
        self.declare_parameter("marker_size", DEFAULT_MARKER_SIZE)
        self.declare_parameter("cam", 2)
        self.declare_parameter("id", -1)
        self.declare_parameter("calib", DEFAULT_CALIB)
        self.declare_parameter("side", "left")

        self.marker_size = self.get_parameter("marker_size").get_parameter_value().double_value
        self.cam = self.get_parameter("cam").get_parameter_value().integer_value
        self.id = self.get_parameter("id").get_parameter_value().integer_value
        self.calib = self.get_parameter("calib").get_parameter_value().string_value
        self.side = self.get_parameter("side").get_parameter_value().string_value
        self.detect = self.make_detector(getattr(aruco, DEFAULT_DICT))

        self.camera_matrix, self.dist_coeffs = self.load_calibration(self.calib, self.side)

        self.cap = cv2.VideoCapture(self.cam)

        if not self.cap.isOpened():
            raise RuntimeError(f"Khong mo duoc camera {self.cam}")

        
        # Tạo timer
        self.timer = self.create_timer(0.03, self.timer_callback)

    def timer_callback(self):
        ok, frame = self.cap.read()
        if not ok:
            return
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_proc = self.ssr( gray, sigma=100.0)
        corners, ids, _ = self.detect(gray)

        if ids is not None:
            rvecs, tvecs, _ = self.estimate_pose(corners, self.marker_size, self.camera_matrix, self.dist_coeffs)
            aruco.drawDetectedMarkers(frame, corners, ids)

            y = 30
            for i, marker_id in enumerate(ids.flatten()):
                if self.id != -1 and marker_id != self.id:
                    continue

                rvec = rvecs[i]
                tvec = tvecs[i]
            
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_size * 0.5)

                x, yy, z = tvec.ravel()
                distance = np.linalg.norm(tvec)

                rmat, _ = cv2.Rodrigues(rvec)
                roll, pitch, yaw = self.rotation_to_euler(rmat)

                lines = [
                    f"id={marker_id} dist={distance:.1f} cm",
                    f"x={x:.1f} y={yy:.1f} z={z:.1f} cm",
                    f"roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f} deg",
                ]

                for line in lines:
                    cv2.putText(
                        frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )
                    y += 25
                y += 10
        else:
            cv2.putText(frame, "Khong thay marker", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("aruco_pose_estimation", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rclpy.shutdown()
            return

    def load_calibration(self, path, side):
        with open(path) as f:
            data = json.load(f)
        camera_matrix = np.array(data[side]["matrix"], dtype=np.float64)
        dist_coeffs = np.array(data[side]["distortion"], dtype=np.float64).reshape(-1, 1)
        return camera_matrix, dist_coeffs

    def make_detector(self, dict_id):
        if hasattr(aruco, "ArucoDetector"):
            dictionary = aruco.getPredefinedDictionary(dict_id)
            detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
            return detector.detectMarkers

        dictionary = aruco.Dictionary_get(dict_id)
        parameters = aruco.DetectorParameters_create()
        return lambda gray: aruco.detectMarkers(gray, dictionary, parameters=parameters)

    def rotation_to_euler(self, rmat):
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
    
    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()
        
    def estimate_pose(self, corners, marker_size, camera_matrix, dist_coeffs):
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
    def ssr(self, gray, sigma=100.0):
    # """Single-Scale Retinex 

    # Nguyen ly: anh = reflectance x illumination. Illumination la thanh phan
    # thay doi cham (bong do, gradient sang). SSR tach reflectance (thong tin
    # marker) ra khoi illumination.

    # QUAN TRONG
    #   1) LOG TRUOC:  img = log(I)
    #   2) BLUR trong log-domain:  blur = G * log(I)
    #   3) TRU:  retinex = log(I) - G*log(I)
    # sigma = 10 la gia tri tac gia dung cung (camera Arducam 800x600). Neu
    # camera cua ban do phan giai khac nhieu thi co the chinh nhe, nhung khong
    # lon nhu 100.
        # """
        img = gray.astype(np.float32) + 1.0            # +1 tranh log(0)
        cv2.log(img, img)                              # 1) LOG TRUOC (in-place)
        # ksize tinh tu sigma dung cong thuc cua tac gia (bat le bang |1)
        ksize = int(round((sigma - 0.8) / 0.15 + 2.0)) | 1
        blur = cv2.GaussianBlur(img, (ksize, ksize), sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)  # 2) blur log-domain
        retinex = img - blur                           # 3) tru trong log-domain
        out = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
        return out.astype(np.uint8)

def main(args=None):
    rclpy.init(args=args)
    node = Pose_estimation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # Đảm bảo shutdown chỉ gọi 1 lần
        if rclpy.ok(): 
            rclpy.shutdown()

if __name__ == "__main__":
    main()