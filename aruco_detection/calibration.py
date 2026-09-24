import json
import numpy as np  


def load_calibration(path, side):
    with open(path) as f:
        data = json.load(f)
    camera_matrix = np.array(data[side]["matrix"], dtype=np.float64)
    dist_coeffs = np.array(data[side]["distortion"], dtype=np.float64).reshape(-1, 1)
    return camera_matrix, dist_coeffs


def fit_camera_matrix(camera_matrix, calib_size, frame_size):
    """Scale intrinsics from the calibration resolution to the running one."""
    cw, ch = calib_size
    fw, fh = frame_size
    if (cw, ch) == (fw, fh):
        return camera_matrix

    sx, sy = fw / cw, fh / ch
    if abs(sx - sy) > 0.01:
        raise RuntimeError(
            f"Aspect ratio mismatch (sx={sx:.3f} sy={sy:.3f}): the camera is "
            f"cropping rather than scaling, so the intrinsics cannot be scaled "
            f"correctly. Recalibrate at {fw}x{fh}.")

    scaled = camera_matrix.copy()
    scaled[0, 0] *= sx
    scaled[0, 2] *= sx
    scaled[1, 1] *= sy
    scaled[1, 2] *= sy
    return scaled
