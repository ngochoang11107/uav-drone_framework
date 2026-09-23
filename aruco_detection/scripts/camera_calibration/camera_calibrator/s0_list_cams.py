"""
This script lists all available USB cameras connected to the system.
"""

import cv2

def list_cameras(max_cameras=10):
    available_cameras = []
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    return available_cameras

def main():
    cameras = list_cameras()
    
    if cameras:
        print(f"Available USB cameras: {cameras}")
    else:
        print("No USB cameras found.")

if __name__ == "__main__":
    main()
