"""Image processing: the region of the camera the analysis uses, oriented and cropped like the DAQ images (S2);
the background, noise processing, energy cut and band statistics of one frame (S3).

Planned (S5): off-shot row correction, the step's energy cut and the step reading.

S3 re-implements the parts of Yiheng Ye's XTCAV analysis (xtcav.py: preprocess_shot, process_noisy_image) and of
twobunch-s2e-rl's online/features.py (valley_threshold, band_stats) that the deployed analysis uses. Tests check
them against both on DAQ frames.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import ndimage, signal

# DTOTR2 camera settings in the DAQ runs the deployed analysis was built on (TEST_15668, E300_15671,
# E300_15673; metadata.DTOTR2 of each run). Keys are the PV suffixes, as in pvs.CAMERA_SETTINGS.
TRAINED = {"MinX_RBV": 396, "MinY_RBV": 70, "SizeX_RBV": 184, "SizeY_RBV": 894,
           "X_ORIENT": "Positive", "Y_ORIENT": "Positive", "IS_ROTATED": 1,
           "AcquireTime_RBV": 0.0001, "Gain_RBV": 0.0, "RESOLUTION": 30.5}

# The deployed analysis's hand crop, (left, upper, right, lower) on the oriented trained ROI
# (xtcav_real_features.HAND_CROP). Its rows run past the trained ROI's 184, and PIL's crop fills those with 0.
# At the training orientation it takes sensor columns 396-579 and rows 270-719.
HAND_CROP = (200, 0, 650, 300)
# The hand crop's rows that hold pixels: the oriented trained ROI is SizeX_RBV = 184 rows tall (trained_frame
# refuses an orientation that doesn't transpose). The rest is padding.
PIXEL_ROWS = TRAINED["SizeX_RBV"]

# The deployed analysis's noise processing (twobunch-s2e-rl scripts/xtcav_real_features.py)
HOT_PIXEL = 1000    # counts: a pixel above this is replaced by the median of its 3 x 3 neighbourhood
THRESHOLD = 35      # counts: then pixels below this are set to 0

# The training's band statistics (online/features.py)
EDGE_BAND = 0.3e-3  # m: intensity this close to the crop's x edges counts in the edge fraction
SENS_ROWS = 2       # rows: the cut-sensitivity offsets move the cut by this much


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


def x_positions(resolution: float) -> np.ndarray:
    """Each hand-crop column's x (m), from 0: the column times the pixel size `resolution` (um, the camera's
    RESOLUTION), as the DAQ classes scale it."""
    return resolution * 1e-6 * np.arange(HAND_CROP[2] - HAND_CROP[0])


def pedestal(crops: np.ndarray) -> np.ndarray:
    """A constant background: the median of every pixel of the hand crops `crops` (frames, rows, columns), and 0
    in the padding rows. With beam on, the beam covers only a few percent of the crop (2.2-2.4% of the pixels
    are 35 counts or more above the median in TEST_15668, E300_15671 and E300_15673), so the median is the dark
    level."""
    image = np.zeros(crops.shape[1:])
    image[:PIXEL_ROWS] = np.median(crops[:, :PIXEL_ROWS])
    return image


def process(crop: np.ndarray, background: np.ndarray) -> np.ndarray:
    """The deployed noise processing of one hand crop: subtract the background, replace each pixel above
    HOT_PIXEL by the median of its 3 x 3 neighbourhood, then set the pixels below THRESHOLD to 0.

    `crop` is trained_frame's output and `background` a float array of the same shape, 0 in the padding rows: the
    padding stays 0, as in the DAQ chain, which subtracts before cropping. The DAQ chain keeps the subtracted
    frame in float32 (PIL) where this keeps float64; with an integer background the results are equal.
    """
    data = crop.astype(float) - background
    data = np.where(data > HOT_PIXEL, ndimage.median_filter(data, size=3), data)
    return np.where(data < THRESHOLD, 0.0, data)


class Valley(NamedTuple):
    cut: int | None         # the cut row; None if there is no valley
    smoothed: np.ndarray    # the y projection smoothed by a 2-row Gaussian
    peaks: np.ndarray       # rows of the smoothed projection's peaks
    between: tuple          # the two peaks the cut lies between; () if there is no valley


def valley(projection: np.ndarray) -> Valley:
    """The training's energy cut on a processed frame's y projection (online/features.valley_threshold).

    Smooth the projection by a 2-row Gaussian. Take its most prominent peak and, on the side of the projection's
    centre of mass, the nearest peak with at least 5% of that prominence. The cut is the row between them where
    the smoothed projection is lowest.
    """
    p = ndimage.gaussian_filter1d(np.asarray(projection, float), 2.0)
    peaks, _ = signal.find_peaks(p)
    if len(peaks) < 2:
        return Valley(None, p, peaks, ())
    prominence = signal.peak_prominences(p, peaks)[0]
    top = peaks[np.argmax(prominence)]
    centre = np.average(np.arange(len(p)), weights=np.clip(p, 0, None) + 1e-30)
    side = np.sign(centre - top)
    candidates = peaks[(np.sign(peaks - top) == side) & (prominence >= 0.05 * prominence.max())]
    if len(candidates) == 0:
        return Valley(None, p, peaks, ())
    near = candidates[np.argmin(np.abs(candidates - top))]
    low, high = sorted((int(top), int(near)))
    return Valley(low + int(np.argmin(p[low:high + 1])), p, peaks, (int(top), int(near)))


def band_stats(image: np.ndarray, x: np.ndarray, cut: int) -> dict | None:
    """The training's band statistics of one processed frame (online/features.band_stats) with the energy cut at
    row `cut`. The rows up to and including the cut are one band and the rest the other; the witness is the band
    with less intensity. `x` is each column's x (m, x_positions). None if a band is empty.

    Returns cut_row; witness_low (the witness is the rows up to the cut); x_d and x_w, the drive and witness x
    centroids (m); var_d, the drive's x variance (m^2); frac_d, the drive's fraction of the intensity; edge_frac,
    the fraction of the intensity within EDGE_BAND of the crop's x edges; off_cut_hi and off_cut_lo, x_d - x_w
    with the cut SENS_ROWS rows higher and lower (the witness on the same side).
    """
    image = np.clip(np.asarray(image, float), 0, None)
    rows = np.arange(image.shape[0])
    low = rows <= cut
    witness_low = bool(image[low].sum() < image[~low].sum())
    witness = low if witness_low else ~low
    drive_x, witness_x = image[~witness].sum(axis=0), image[witness].sum(axis=0)
    q_d, q_w = drive_x.sum(), witness_x.sum()
    if q_d <= 0 or q_w <= 0:
        return None
    x_d = np.average(x, weights=drive_x)
    both_x = drive_x + witness_x
    k = max(1, int(round(EDGE_BAND / abs(float(x[1] - x[0])))))      # columns in an edge band
    out = {"cut_row": int(cut), "witness_low": witness_low, "x_d": float(x_d),
           "x_w": float(np.average(x, weights=witness_x)), "var_d": float(np.average((x - x_d) ** 2, weights=drive_x)),
           "frac_d": float(q_d / (q_d + q_w)),
           "edge_frac": float((both_x[:k].sum() + both_x[-k:].sum()) / max(both_x.sum(), 1e-30))}
    for key, shift in (("off_cut_hi", SENS_ROWS), ("off_cut_lo", -SENS_ROWS)):
        low = rows <= cut + shift
        witness = low if witness_low else ~low
        d, w = image[~witness].sum(axis=0), image[witness].sum(axis=0)
        out[key] = (float(np.average(x, weights=d) - np.average(x, weights=w))
                    if d.sum() > 0 and w.sum() > 0 else float("nan"))
    return out
