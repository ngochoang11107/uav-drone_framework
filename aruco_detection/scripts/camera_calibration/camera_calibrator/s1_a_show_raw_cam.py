"""
This script opens a single camera and displays its raw frames in a window.
"""

import cv2
from utils_read_cam import open_camera
from utils_timer_tictok import TimerTicTok
from utils_configs import CamConfig
tictok = TimerTicTok(window_seconds=1.0)


if __name__ == "__main__":
    cam = open_camera(CamConfig.RIGHT_CAM) # 0= right cam # 4 leftcam

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        cv2.imshow('Webcam', frame)
        tictok.update()
        tictok.pprint()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

