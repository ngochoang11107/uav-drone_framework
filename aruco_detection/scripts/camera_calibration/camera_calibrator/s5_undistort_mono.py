import cv2
import numpy as np
from utils_calib import load_calib
np.set_printoptions(precision=3, suppress=True)

def main():
    K1, D1, K2, D2, R, T = load_calib()

    # ===================================================================================
    # prepare undistort maps
    h, w = 800, 1280
    img_shape = (w, h)  # (width, height)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                                            K1, D1, K2, D2,
                                            img_shape,
                                            R, T,
                                            flags=cv2.CALIB_ZERO_DISPARITY,
                                            alpha=1      # 0=crop to valid pixels, -1=keep all pixels
                                        )
    # Left camera maps
    mapLx, mapLy = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_shape, cv2.CV_16SC2)
    # Right camera maps
    mapRx, mapRy = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_shape, cv2.CV_16SC2)

    # decompose P -> K(intrinsic) and T
    K1 = P1[:, :3]
    K2 = P2[:, :3]
    print(f"K1 (after rectification):\n{K1}")
    print(f"K2 (after rectification):\n{K2}")

    # ===================================================================================
    # read images
    N = 1
    for i in range(N):
        frameL = cv2.imread(f"images_test/left/{i:06d}.jpg")
        frameR = cv2.imread(f"images_test/right/{i:06d}.jpg")
        if frameL is None or frameR is None:
            print(f"Image {i:06d} not found, skipping...")
            break
        if frameL.shape[:2] != (h, w) or frameR.shape[:2] != (h, w):
            print(f"Image shape expected: {(h, w)}, got: {frameL.shape[:2]} and {frameR.shape[:2]}")
            break
        
        # ==============================================================================
        # rectify images
        frameL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
        frameR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)

        # show rectified images
        cv2.imshow("Left Rectified", frameL)
        cv2.imshow("Right Rectified", frameR)

        input = cv2.waitKey(0)
        if input == 27 or input == ord('q'):  # ESC key
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
