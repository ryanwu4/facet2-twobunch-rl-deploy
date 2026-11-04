"""Image processing: the region of the camera the analysis uses, oriented and cropped like the DAQ images (S2);
the background, noise processing, energy cut and band statistics of one frame (S3); the row correction, the step's
energy cut and the step reading (S5).

S3 and S5 re-implement the parts of Yiheng Ye's XTCAV analysis (xtcav.py: preprocess_shot, process_noisy_image,
the row correction) and of twobunch-s2e-rl's online/features.py (valley_threshold, band_stats, step_cut,
step_features) that the deployed analysis uses (scripts/xtcav_online_replay.py, v7 settings). Tests check them
against both on DAQ frames, and the step readings against the v7 replay.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import constants, ndimage, optimize, signal

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

# The row correction (xtcav.py's build_off_reference, fit_row_distortion, compute_distortion_shifts, apply_row_shift)
ROW_SIGNAL = 0.005  # a row of the off reference is fitted if it holds more than this fraction of its intensity

# The step reading (online/features.py's step_cut and step_features, with xtcav_online_replay.py's settings)
MIN_FRAMES = 15                 # kept frames with band statistics per polarity (the policy was trained on 30)
SPACING_FLOOR = 60e-6           # m: a valid reading's |spacing| is above this
EDGE_MAX = 0.01                 # a valid reading's edge fraction is below this
MIN_INTENSITY = 1e5             # counts: a valid reading's kept frames of each polarity have a median summed
                                # intensity of at least this: beam on the screen (beam frames 5-9e5; without beam,
                                # a pedestal leaves about 200 counts of noise, in which the valley rule finds cuts)
UM_PER_DEG_RF = constants.c / 11.424e9 / 360 * 1e6     # um of z per degree of the X-band TCAV's phase


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


class Reference(NamedTuple):
    image: np.ndarray       # the mean of the step's processed off frames
    centroids: np.ndarray   # each row's fitted x centroid (column); nan where the row isn't fitted
    shifts: np.ndarray      # columns each row is shifted by (shift_rows)


def gaussian(x, centre, height, sigma, offset):
    """The Gaussian fitted to each row of the off reference (Yiheng's analysis.fit_gaussian)."""
    return height * np.exp(-0.5 * ((x - centre) / sigma) ** 2) + offset


def row_reference(off: list[np.ndarray]) -> Reference | None:
    """The row-correction reference from a step's processed TCAV-off frames. Their mean; a Gaussian fitted in x to
    each row of it holding more than ROW_SIGNAL of its intensity, whose centre in the crop is the row's centroid;
    and each row's shift, which moves its centroid to the crop's centre column. A row not fitted takes the centroid
    interpolated linearly from the fitted rows, or the nearest fitted row's past either end. None if no row is fitted.

    Re-implements xtcav.py's build_off_reference, fit_row_distortion and compute_distortion_shifts with their defaults
    (the shift is relative to the column where the mean's x projection peaks, plus the shift that centres that
    column: the peak column cancels out).
    """
    image = np.mean(off, axis=0)
    width = image.shape[1]
    columns = np.arange(width, dtype=float)
    centroids = np.full(image.shape[0], np.nan)
    for row, line in enumerate(image):
        if not line.sum() > ROW_SIGNAL * image.sum():
            continue
        p0 = (float(np.argmax(line)), float(np.max(line)), width / 10, float(np.median(line)))
        try:
            centre = optimize.curve_fit(gaussian, columns, line, p0=p0, maxfev=1000)[0][0]
        except RuntimeError:                        # the fit didn't converge
            continue
        if 0 < centre < width:
            centroids[row] = centre
    fitted = np.flatnonzero(np.isfinite(centroids))
    if len(fitted) == 0:
        return None
    filled = np.interp(np.arange(len(centroids)), fitted, centroids[fitted])
    return Reference(image, centroids, (width - 1) / 2 - filled)


def shift_rows(image: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Each row of `image` shifted right by its `shifts` (columns; fractional shifts interpolate linearly), with 0
    where columns come in from outside the image (xtcav.apply_row_shift)."""
    rows, columns = np.indices(image.shape)
    return ndimage.map_coordinates(image.astype(float), [rows, columns - shifts[:, None]], order=1, mode="constant",
                                   cval=0.0)


def step_cut(plus: list[np.ndarray]) -> int | None:
    """The step's energy cut, for its +V and -V frames, from its +V frames (processed, row-corrected): the median of
    the frames' own valley cuts, rounded; None if fewer than half of them have one (online/features.step_cut). A -V
    frame's own valley can lock onto another dip: the TCAV's energy kick changes sign with the polarity."""
    cuts = [c for c in (valley(frame.sum(axis=1)).cut for frame in plus) if c is not None]
    if not cuts or 2 * len(cuts) < len(plus):
        return None
    return int(round(float(np.median(cuts))))


def reading(plus: list[dict], minus: list[dict], streak_um_per_deg: float) -> dict:
    """The step reading from the band statistics of its kept +V and -V frames (online/features.step_features). Each
    statistic's median per polarity gives:
    - spacing: the drive-witness offsets' half-difference, (x_d - x_w at +V minus at -V) / 2 S;
    - drive_sigma_z: the square root of the mean of the two drive x variances, over |S| (screen resolution included);
    - drive_charge_fraction: the mean of the two;
    - edge_frac: the larger of the two;
    - cut_sens: half the change in spacing between the cut SENS_ROWS rows higher and lower.
    The streak S, screen m per m of z, is the calibration `streak_um_per_deg` (at the pixel size of x) over
    UM_PER_DEG_RF; +V streaks by +S and -V by -S. Lengths in m.
    """
    streak = streak_um_per_deg / UM_PER_DEG_RF

    def median(stats, key):
        return float(np.median([s[key] for s in stats]))

    offset_plus = float(np.median([s["x_d"] - s["x_w"] for s in plus]))
    offset_minus = float(np.median([s["x_d"] - s["x_w"] for s in minus]))
    hi = (median(plus, "off_cut_hi") - median(minus, "off_cut_hi")) / (2 * streak)
    lo = (median(plus, "off_cut_lo") - median(minus, "off_cut_lo")) / (2 * streak)
    return {"spacing": (offset_plus - offset_minus) / (2 * streak),
            "drive_sigma_z": float(np.sqrt(0.5 * (median(plus, "var_d") + median(minus, "var_d")))) / abs(streak),
            "drive_charge_fraction": 0.5 * (median(plus, "frac_d") + median(minus, "frac_d")),
            "edge_frac": max(median(plus, "edge_frac"), median(minus, "edge_frac")),
            "cut_sens": 0.5 * abs(hi - lo)}


def x_rms(image: np.ndarray, x: np.ndarray) -> float | None:
    """The beam's x RMS width (m) in a processed frame; None without intensity. In the DAQ runs about 0.2 mm
    unstreaked and 0.7-1.1 mm streaked: the off frames' widths show whether the TCAV was off for the beam."""
    projection = image.sum(axis=0)
    if projection.sum() <= 0:
        return None
    centre = np.average(x, weights=projection)
    return float(np.sqrt(np.average((x - centre) ** 2, weights=projection)))


class StepReading(NamedTuple):
    reference: Reference | None     # from the step's off frames; None if it has none, or no row could be fitted
    cut: int | None                 # the step's energy cut (step_cut)
    stats: list                     # each + and - frame's band_stats at the cut (dropped ones too); None for the rest
    intensity: list                 # each +, - and off frame's summed intensity (processed; + and - corrected);
                                    # None for the rest
    x_rms: list                     # each +, - and off frame's beam x RMS (m, x_rms), likewise
    medians: dict                   # "+" and "-": the median of the polarity's kept frames, processed and corrected
    reading: dict | None            # reading() of the kept frames; None unless both polarities have one with stats
    checks: dict                    # what a valid reading needs, each True or False


def read_step(crops: np.ndarray, tags: np.ndarray, drop: np.ndarray, background: np.ndarray, x: np.ndarray,
              streak_um_per_deg: float) -> StepReading:
    """One step's reading from its frames, as the deployed analysis reads a DAQ step (xtcav_online_replay.py, v7):
    1. process each +, - and off frame (process);
    2. the row-correction reference from the off frames (row_reference), and each + and - frame's rows shifted by it;
    3. the step's cut from the kept + frames (step_cut), and each + and - frame's band statistics at that cut;
    4. the reading from the kept frames with statistics (reading).
    The reading is valid if all its checks pass: at least MIN_FRAMES such frames per polarity, a reference, a cut,
    |spacing| above SPACING_FLOOR and the edge fraction below EDGE_MAX (these two v7's own rule), and beam: each
    polarity's kept frames have a median summed intensity of at least MIN_INTENSITY. Without a reference the + and -
    frames are read uncorrected, and the reading is invalid.

    `crops` are the hand crops, `tags` and `drop` from acquire, `x` each column's x (m, x_positions) and
    `streak_um_per_deg` the TCAV calibration at x's pixel size.
    """
    processed = {i: process(crop, background) for i, (crop, t) in enumerate(zip(crops, tags)) if t in ("+", "-", "off")}
    off = [processed[i] for i in processed if tags[i] == "off"]
    reference = row_reference(off) if off else None
    # frame index -> processed frame; the + and - frames corrected
    frames = {i: image if reference is None or tags[i] == "off" else shift_rows(image, reference.shifts)
              for i, image in processed.items()}
    kept = {sign: [i for i in frames if tags[i] == sign and not drop[i]] for sign in ("+", "-")}
    cut = step_cut([frames[i] for i in kept["+"]])
    stats = [band_stats(frames[i], x, cut) if i in frames and tags[i] != "off" and cut is not None else None
             for i in range(len(tags))]
    good = {sign: [stats[i] for i in kept[sign] if stats[i] is not None] for sign in ("+", "-")}
    result = reading(good["+"], good["-"], streak_um_per_deg) if good["+"] and good["-"] else None
    intensity = [float(frames[i].sum()) if i in frames else None for i in range(len(tags))]
    widths = [x_rms(frames[i], x) if i in frames else None for i in range(len(tags))]
    checks = {f"+ frames >= {MIN_FRAMES}": len(good["+"]) >= MIN_FRAMES,
              f"- frames >= {MIN_FRAMES}": len(good["-"]) >= MIN_FRAMES,
              "row-correction reference": reference is not None,
              "step cut": cut is not None,
              f"|spacing| > {1e6 * SPACING_FLOOR:.0f} um":
                  result is not None and abs(result["spacing"]) > SPACING_FLOOR,
              f"edge fraction < {EDGE_MAX}": result is not None and result["edge_frac"] < EDGE_MAX,
              f"beam: median intensity >= {MIN_INTENSITY:.0e}":
                  all(kept[sign] and np.median([intensity[i] for i in kept[sign]]) >= MIN_INTENSITY
                      for sign in ("+", "-"))}
    medians = {sign: np.median([frames[i] for i in kept[sign]], axis=0) for sign in kept if kept[sign]}
    return StepReading(reference, cut, stats, intensity, widths, medians, result, checks)
