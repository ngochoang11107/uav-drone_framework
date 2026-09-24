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
    calibrator = Calibrator(board_shape, checker_size)
    

    # choose k frames among N collected frames for calibration - if N is large
    frame_ids = np.random.choice(299, 100, replace=False) # N=299, k=100

    # First, calibrate each camera individually.
    KL, distL, _, _ = process_frames(calibrator, frame_ids, 'left', show_corners=True) 
    

    KL_single = KL.copy()
    distL_single = distL.copy()
   
    print("Single camera calibration done.")
    print(f"Left K1:\n{KL_single}\nLeft D1:\n{distL_single}")
   

    print("Starting stereo calibration...")
   

    # save stereo calibration results
    print("Saving stereo calibration results...")
    stereo_calib_data = {
        "camera_name":{
            "matrix": KL_single.tolist(),
            "distortion": distL_single.tolist(),
        },        
    }

    with open("calib_data_mono.json", "w") as f:
        json.dump(stereo_calib_data, f, indent=4)
    print("Mono calibration data saved to calib_data_mono.json")

    cv2.destroyAllWindows()

