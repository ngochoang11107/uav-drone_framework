import cv2
import time
from utils_read_cam import open_camera, ThreadedCamera
from utils_timer_tictok import TimerTicTok
from utils_configs import CamConfig, CheckerboardConfig, IMG_CALIB_DIR
import os


os.makedirs(f"{IMG_CALIB_DIR}/left", exist_ok=True)
os.makedirs(f"{IMG_CALIB_DIR}/right", exist_ok=True)

# this is frame rate measurement utility
tictok = TimerTicTok(window_seconds=1.0)

# open both cameras in threaded mode, left and right.
tcL = ThreadedCamera(open_camera(CamConfig.LEFT_CAM))
tcL.start()
tcR = ThreadedCamera(open_camera(CamConfig.RIGHT_CAM))
tcR.start()

# default last frame ids = -1
last_id_L = last_id_R = -1

counter = 0
while True:
    # read frames from both cameras using the threaded camera interface
    retL, frameL, tsL, idL = tcL.read()
    retR, frameR, tsR, idR = tcR.read()

    # Only proceed when BOTH cameras produced a NEW frame since last loop
    if idL == last_id_L or idR == last_id_R:
        time.sleep(0.0002)
        continue

    # concatenate the two frames side by side for display
    wide_view = cv2.hconcat([frameL, frameR])
    wide_view = cv2.resize(wide_view, (1280, 400)) # resize for display.
    cv2.imshow('Wide View', wide_view)
    input = cv2.waitKey(1)
    if input & 0xFF == ord('q'):
        break

    # find the checkerboard corners in both frames
    retL, cornersL = cv2.findChessboardCorners(frameL, CheckerboardConfig.PATTERN_SIZE)
    retR, cornersR = cv2.findChessboardCorners(frameR, CheckerboardConfig.PATTERN_SIZE)

    # if the corners are found, save the images to disk
    if (retL and retR):
        cv2.imwrite(f"{IMG_CALIB_DIR}/left/{counter:06d}.jpg", frameL)
        cv2.imwrite(f"{IMG_CALIB_DIR}/right/{counter:06d}.jpg", frameR)
        counter += 1
        print(f"Captured frame {counter:06d}")
        time.sleep(0.1)

cv2.destroyAllWindows()

