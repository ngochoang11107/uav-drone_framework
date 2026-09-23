"""
This script opens two cameras in threaded mode and displays their raw frames side by side.
"""
import cv2
import time
from utils_read_cam import open_camera, ThreadedCamera
from utils_timer_tictok import TimerTicTok
from utils_configs import CamConfig

# this is frame rate measurement utility
tictok = TimerTicTok(window_seconds=1.0)

# open both cameras in threaded mode, left and right.
tcL = ThreadedCamera(open_camera(CamConfig.LEFT_CAM))
tcL.start()
tcR = ThreadedCamera(open_camera(CamConfig.RIGHT_CAM))
tcR.start()

# default last frame ids = -1
last_id_L = last_id_R = -1

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


cv2.destroyAllWindows()











