
import cv2, json  
import numpy as np  
import matplotlib.pyplot as plt  
np.set_printoptions(precision=2, suppress=True)  # Format numpy output for readability

# Load the first calibration image from the left camera 
frame_photos = list(range(1,41))
for i in frame_photos:
    filepath = f"/home/ngoc/drone_ws/src/skydroid/skydroid/camera/{i:02}.jpg"
    # print(filepath)
    img = cv2.imread(filepath)

img = cv2.imread(f"/home/ngoc/drone_ws/src/skydroid/skydroid/camera/{i:02}.jpg")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))        
plt.show()


h, w = img.shape[:2] 

board_shape = (8, 6)  # 9 columns and 6 rows of inner corners

checker_size = 35  # in mm


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((board_shape[0]*board_shape[1], 3), np.float32) # (54,3) . . . 

objp[:, :2] =  np.mgrid[0:board_shape[0], 0: board_shape[1]].T.reshape(-1, 2)

objp *= checker_size  # scale to mm
objpoints = []  # 3D points in board coords 
imgpoints = []  # 2D points in image plane
img_shape = (w, h)

frame_ids = list(range(1,41))  # Adjust as needed for more images
for i in frame_ids:
    # Load the current image
    frame = cv2.imread(f"/home/ngoc/drone_ws/src/skydroid/skydroid/camera/{i:02}.jpg")
    
    # Find chessboard corners in the image
    ret, corners = cv2.findChessboardCorners(frame, board_shape, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    # Refine corner locations to subpixel accuracy
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners_ref = cv2.cornerSubPix(gray_frame, corners, (11, 11), (-1, -1), criteria)
    
    # Draw the refined corners on the image for visualization 
    frame = cv2.drawChessboardCorners(frame, board_shape, corners_ref, ret)
    

# Loop through a small set of calibration images to visualize corner detection
frame_ids = list(range(1,41))  # Adjust as needed for more images
for i in frame_ids:
    # Load the current image
    frame = cv2.imread(f"/home/ngoc/drone_ws/src/skydroid/skydroid/camera/{i:02}.jpg")
    
    # Find chessboard corners in the image
    ret, corners = cv2.findChessboardCorners(frame, board_shape, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    # Refine corner locations to subpixel accuracy
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners_ref = cv2.cornerSubPix(gray_frame, corners, (11, 11), (-1, -1), criteria)
    
    # Draw the refined corners on the image for visualization 
    frame = cv2.drawChessboardCorners(frame, board_shape, corners_ref, ret)


    # Store the object points and image points for calibration
    objpoints.append(objp.copy())
    imgpoints.append(corners_ref)
    
# Set calibration flags
flags = cv2.CALIB_FIX_K3 | cv2.CALIB_USE_INTRINSIC_GUESS  # Fix k3 distortion, use initial guess for intrinsics

# Initial camera matrix setup (intrinsic parameters)
initial_fx = 1000.0  # Initial horizontal focal length
initial_fy = 1000.0  # Initial vertical focal length (square pixels)
image_width = 1280
image_height = 720
initial_cx = image_width / 2  # Principal point x
initial_cy = image_height / 2  # Principal point y
initial_camera_matrix = np.array([[initial_fx, 0, initial_cx],
                                [0, initial_fy, initial_cy],
                                [0, 0, 1]], dtype=np.float32)

# Run camera calibration using collected object and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, 
        cameraMatrix=initial_camera_matrix, distCoeffs=None,
        flags=flags, 
        criteria=criteria
    )

# Print calibration results
print("\n=== Calibration Results (LEFT) ===")
print(f"RMS reprojection error: {ret:.4f}")
print("K (intrinsics):\n", mtx)
print("dist (k1 k2 p1 p2 k3):\n", dist.ravel())

# Undistort the image using the computed camera matrix and distortion coefficients
undistorted = cv2.undistort(frame, mtx, dist)

# Visualize the original and undistorted images side by side
plt.figure(figsize=(16, 9))
plt.subplot(1, 2, 1)
plt.axis('off')
plt.title("Original")
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.axis('off')
plt.title("Undistorted")
plt.imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
plt.show()