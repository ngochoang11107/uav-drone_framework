from utils_calib import Calibrator
import cv2, json
import numpy as np
from utils_configs import IMG_CALIB_DIR

np.set_printoptions(precision=3, suppress=True)

def process_frames(calibrator:Calibrator, frame_ids, side='left', show_corners=False):
    print(f"Processing {side} images for calibration...")
    counter = 0
    lost_frames = 0

    for i in frame_ids:
        imgL = cv2.imread(f"{IMG_CALIB_DIR}/{side}/{i:06d}.jpg")
        # shape of image
        h, w = imgL.shape[:2]
        calibrator.img_shape = [w, h]
        ret, frame = calibrator.process_frame(imgL, cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY))
        if ret:
            calibrator.save_corners()
        if not ret:
            print(f"Frame {i:06d} failed to find corners.")
            lost_frames += 1

        if show_corners:
            cv2.imshow(f"Corners - {side}", frame)
            cv2.waitKey(1)
        counter +=1
    print(f"Captured {counter} frames with {lost_frames} lost frames.")
    
    print(f"Calculating calibration parameters. This may take a while...")
    
    h,  w  = calibrator.img_shape
    calibrator.img_shape = (w, h)  # OpenCV expects (width,
    mtx, dist = calibrator.calibrate((w, h))
    return mtx, dist, counter, lost_frames


if __name__ == "__main__":
    board_shape = (9, 6)
    checker_size = 14  # in mm

    # define each mono calibrator
    calib_left = Calibrator(board_shape, checker_size)
    calib_right = Calibrator(board_shape, checker_size)

    # choose k frames among N collected frames for calibration - if N is large
    frame_ids = np.random.choice(299, 100, replace=False) # N=299, k=100

    # First, calibrate each camera individually.
    KL, distL, _, _ = process_frames(calib_left, frame_ids, 'left', show_corners=True) 
    KR, distR, _, _ = process_frames(calib_right, frame_ids, 'right', show_corners=True) 

    KL_single = KL.copy()
    KR_single = KR.copy()
    distL_single = distL.copy()
    distR_single = distR.copy()
    print("Single camera calibration done.")
    print(f"Left K1:\n{KL_single}\nLeft D1:\n{distL_single}")
    print(f"Right K2:\n{KR_single}\nRight D2:\n{distR_single}")

    print("Starting stereo calibration...")

    # stereo calibration
    flags = (
        cv2.CALIB_FIX_INTRINSIC |
        #cv2.CALIB_USE_INTRINSIC_GUESS |
        #cv2.CALIB_FIX_PRINCIPAL_POINT |    # keep (cx,cy) fixed
        cv2.CALIB_FIX_ASPECT_RATIO |        # fx/fy ratio fixed (square pixels)
        cv2.CALIB_SAME_FOCAL_LENGTH |       # fxR=fxL, fyR=fyL        
        cv2.CALIB_ZERO_TANGENT_DIST         # p1=p2=0        
        #cv2.CALIB_FIX_K1 |                 # fix k1
        #cv2.CALIB_FIX_K2 |                 # fix k2
        #cv2.CALIB_FIX_K3 |                 # fix k3        
        )

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    rms, K1o, D1o, K2o, D2o, R, T, E, F = cv2.stereoCalibrate(
        calib_left.objpoints,            # list of (N x 3) object points
        calib_left.imgpoints,            # list of (N x 2) left image points
        calib_right.imgpoints,           # list of (N x 2) right image points
        KL, distL,                       # initial left intrinsics/dist
        KR, distR,                       # initial right intrinsics/dist
        calib_left.img_shape,            # (width, height)
        flags=flags,
        R=np.eye(3),
        T=np.array([-0.231, 0.0, 0.0]), # change this based on your setup as an initial guess
        criteria=criteria
    )
    print(f"Stereo Calibration RMS error: {rms}")
    # T = T/1000.0 # convert T from mm to meters
    # T[0] *= -1 # depending on your setup, you might need to invert the X axis

    print(f"Left K1:\n{K1o}\nLeft D1:\n{D1o}")
    print(f"Right K2:\n{K2o}\nRight D2:\n{D2o}")

    # save stereo calibration results
    print("Saving stereo calibration results...")
    stereo_calib_data = {
        "left_single":{
            "matrix": KL_single.tolist(),
            "distortion": distL_single.tolist(),
        },
        "right_single":{
            "matrix": KR_single.tolist(),
            "distortion": distR_single.tolist(),
        },
        "left": {
            "matrix": K1o.tolist(),
            "distortion": D1o.tolist(),
        },  
        "right": {
            "matrix": K2o.tolist(),
            "distortion": D2o.tolist(),
        },
        "R": R.tolist(),
        "T": T.tolist(),
    }
    print("Writing stereo calibration data to calib_data_stereo.json")

    with open("calib_data_stereo.json", "w") as f:
        json.dump(stereo_calib_data, f, indent=4)
    print("Stereo calibration data saved to calib_data_stereo.json")

    cv2.destroyAllWindows()
