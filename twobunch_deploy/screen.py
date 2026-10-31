"""Image processing: the region of the camera the analysis uses, oriented and cropped like the DAQ images (S2).

Planned (S3, S5): background, median / hot-pixel / threshold filter, off-shot row correction, energy cut,
band statistics and the step reading. They re-implement the parts of Yiheng's XTCAV analysis (xtcav.py) and
of twobunch-s2e-rl's online/features.py that the deployed analysis uses; tests check them against both on
DAQ frames.
"""
from __future__ import annotations

import numpy as np

# DTOTR2 camera settings in the DAQ runs the deployed analysis was built on (TEST_15668, E300_15671,
# E300_15673; metadata.DTOTR2 of each run). Keys are the PV suffixes, as in pvs.CAMERA_SETTINGS.
TRAINED = {"MinX_RBV": 396, "MinY_RBV": 70, "SizeX_RBV": 184, "SizeY_RBV": 894,
           "X_ORIENT": "Positive", "Y_ORIENT": "Positive", "IS_ROTATED": 1,
           "AcquireTime_RBV": 0.0001, "Gain_RBV": 0.0, "RESOLUTION": 30.5}

# The deployed analysis's hand crop, (left, upper, right, lower) on the oriented trained ROI
# (xtcav_real_features.HAND_CROP). Its rows run past the trained ROI's 184, and PIL's crop fills those with 0.
# At the training orientation it takes sensor columns 396-579 and rows 270-719.
HAND_CROP = (200, 0, 650, 300)


def trained_frame(data: np.ndarray, camera: dict) -> np.ndarray:
    """The hand crop of one raw frame, as the deployed analysis cuts it from the DAQ images.

    `data` is the flat array the camera published and `camera` the readbacks read with it
    (machine.camera_settings()). The crop is taken in sensor pixels, so it holds the same pixels whatever
    ROI the camera is set to. Raises ValueError if the array isn't the driver's own output (port PS107, not
    reversed, the ROI's size: no binning), or if the camera's ROI doesn't contain the crop.
    """
    if camera["NDArrayPort_RBV"] != "PS107":
        raise ValueError(f"the camera array comes from port {camera['NDArrayPort_RBV']}, not the driver (PS107)")
    if (camera["ReverseX_RBV"], camera["ReverseY_RBV"]) != ("No", "No"):
        raise ValueError(f"the driver reverses the image: {camera}")
    roi = camera["SizeX_RBV"], camera["SizeY_RBV"]
    if (camera["ArraySizeX_RBV"], camera["ArraySizeY_RBV"]) != roi or \
            (camera["ArraySize0_RBV"], camera["ArraySize1_RBV"]) != roi:
        raise ValueError(f"the camera array is not the size of the camera ROI: {camera}")
    raw = data.reshape(camera["SizeY_RBV"], camera["SizeX_RBV"])    # rows are y: x is the fast axis

    # sensor row and column of each pixel of the trained ROI, oriented and cropped like the DAQ images
    t = TRAINED
    rows, cols = np.mgrid[t["MinY_RBV"]:t["MinY_RBV"] + t["SizeY_RBV"],
                          t["MinX_RBV"]:t["MinX_RBV"] + t["SizeX_RBV"]]
    left, upper, right, lower = HAND_CROP
    rows, cols = (orient(a, camera["X_ORIENT"], camera["Y_ORIENT"], camera["IS_ROTATED"])[upper:lower, left:right]
                  for a in (rows, cols))
    # the same pixels in the camera's array
    rows, cols = rows - camera["MinY_RBV"], cols - camera["MinX_RBV"]
    inside = (rows >= 0) & (rows < raw.shape[0]) & (cols >= 0) & (cols < raw.shape[1])
    if rows.shape[1] != right - left or not inside.all():
        raise ValueError(f"the camera ROI does not contain the hand crop: {camera}")
    frame = np.zeros((lower - upper, right - left), raw.dtype)       # the rows past the trained ROI stay 0
    frame[:len(rows)] = raw[rows, cols]
    return frame


def orient(image: np.ndarray, x_orient: str, y_orient: str, is_rotated: int) -> np.ndarray:
    """Flip and transpose a raw image as the ops display and the DAQ images do (Yiheng's image.orientImage).

    A Negative X_ORIENT flips the columns and a Negative Y_ORIENT the rows; then IS_ROTATED = 1 transposes.
    """
    if {x_orient, y_orient} - {"Positive", "Negative"}:
        raise ValueError(f"unexpected orientation: X_ORIENT {x_orient!r}, Y_ORIENT {y_orient!r}")
    if x_orient == "Negative":
        image = image[:, ::-1]
    if y_orient == "Negative":
        image = image[::-1, :]
    return image.T if is_rotated == 1 else image
