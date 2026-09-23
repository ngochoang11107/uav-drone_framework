class CamConfig:
    LEFT_CAM = 2   # change this after running "python s0_list_cams.py"
    RIGHT_CAM = 0  # change this after running "python s0_list_cams.py"
 

class CheckerboardConfig:
    PATTERN_SIZE = (9, 6)  # number of inner corners per a chessboard row and column
    SQUARE_SIZE = 0.025    # size of a square in your defined unit (m, ft, in, etc.)


# Make directories to save checkerboard images for calibration
IMG_CALIB_DIR = "images_calib"










