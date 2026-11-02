"""Plots. They draw only values computed by screen.py.

S2  geometry(): a full frame as the ops display orients it, with the hand crop outlined
S3  background(): a background and its brightest frame; peek(): one frame through the processing, with its energy
    cut and band statistics
Planned (S4, S5): peek of several frames by TCAV polarity, watch(), reading.show()
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import screen

DRIVE, WITNESS, CUT = "#eb6834", "#2a9df4", "#1fc7b6"


def geometry(data: np.ndarray, camera: dict, title: str = ""):
    """Left: the whole frame, oriented like the ops display, with the hand crop outlined. Right: the hand
    crop as the analysis gets it (screen.trained_frame). Grey scale from the 1st to the 99.9th percentile, so
    a few hot pixels don't set it."""
    orientation = camera["X_ORIENT"], camera["Y_ORIENT"], camera["IS_ROTATED"]
    raw_shape = camera["SizeY_RBV"], camera["SizeX_RBV"]
    full = screen.orient(data.reshape(raw_shape), *orientation)
    # the hand crop's pixels: crop the pixel numbers 1, 2, ...; the crop's padding is 0
    taken = screen.trained_frame(np.arange(1, data.size + 1), camera)
    inside = np.zeros(raw_shape, bool)
    inside.flat[taken[taken > 0] - 1] = True
    inside = screen.orient(inside, *orientation)
    trained = screen.trained_frame(data, camera)
    vmin, vmax = np.percentile(full, [1, 99.9])

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.5), width_ratios=[full.shape[1] / full.shape[0], 1.5])
    a.imshow(full, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    a.contour(inside, levels=[0.5], colors="#eb6834", linewidths=1.5)
    a.set_title("whole frame, display orientation\nhand crop outlined", fontsize=9)
    b.imshow(trained, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    b.set_title(f"hand crop, {trained.shape[0]} x {trained.shape[1]}\n(rows from 184 are padding)", fontsize=9)
    for ax in (a, b):
        ax.set_xlabel("column")
        ax.set_ylabel("row")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def background(image: np.ndarray, brightest: np.ndarray, title: str = ""):
    """Left: a background, as subtracted from each hand crop. Right: the hand crop of the frame with the highest
    pixel, where beam would show. Each in grey from its own 1st to 99.9th percentile."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, im, name in zip(axes, (image, brightest), ("background", "brightest frame")):
        vmin, vmax = np.percentile(im[:screen.PIXEL_ROWS], [1, 99.9])
        fig.colorbar(ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest"), ax=ax,
                     label="counts")
        ax.set_title(f"{name} (rows from {screen.PIXEL_ROWS} are padding)", fontsize=9)
        ax.set_xlabel("column")
        ax.set_ylabel("row")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def peek(crop: np.ndarray, processed: np.ndarray, valley: screen.Valley, stats: dict | None, x: np.ndarray,
         title: str = ""):
    """Left: the hand crop as cut. Middle: the frame after screen.process, with the energy cut, the rows
    SENS_ROWS above and below it (the cut-sensitivity check), the witness band shaded, the drive and witness x
    centroids, the drive's +-1 sigma and the EDGE_BAND edge bands. Right: the smoothed y projection the cut was
    found on, with its peaks; the cut lies between the two ringed ones. x in mm."""
    mm = 1e3 * x
    half = (mm[1] - mm[0]) / 2
    extent = (mm[0] - half, mm[-1] + half, crop.shape[0] - 0.5, -0.5)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True, width_ratios=[1, 1, 0.5])
    vmin, vmax = np.percentile(crop[:screen.PIXEL_ROWS], [1, 99.9])
    a.imshow(crop, cmap="gray", vmin=vmin, vmax=vmax, extent=extent, aspect="auto", interpolation="nearest")
    a.set_title("hand crop (counts)", fontsize=9)
    b.imshow(processed, cmap="gray", extent=extent, aspect="auto", interpolation="nearest")
    b.set_title("processed: cut, witness band shaded, centroids, drive +-1 sigma, edge bands", fontsize=9)
    band = 1e3 * screen.EDGE_BAND
    b.axvspan(extent[0], extent[0] + band, color="#d6c94a", alpha=0.3)
    b.axvspan(extent[1] - band, extent[1], color="#d6c94a", alpha=0.3)
    rows = np.arange(len(valley.smoothed))
    c.plot(valley.smoothed, rows, color="k", lw=1)
    c.plot(valley.smoothed[valley.peaks], valley.peaks, "o", ms=4, mfc="none", color="0.5")
    if valley.between:
        c.plot(valley.smoothed[list(valley.between)], valley.between, "o", ms=9, mfc="none", color="k")
    c.set_title("y projection, smoothed", fontsize=9)
    if valley.cut is not None:
        for ax in (b, c):
            ax.axhline(valley.cut, color=CUT, lw=1.2, label="cut")
            for shift in (-screen.SENS_ROWS, screen.SENS_ROWS):
                ax.axhline(valley.cut + shift, color=CUT, lw=0.8, ls=":")
    if stats is not None:
        witness_rows = (-0.5, stats["cut_row"] + 0.5) if stats["witness_low"] else \
            (stats["cut_row"] + 0.5, screen.PIXEL_ROWS - 0.5)
        b.axhspan(*witness_rows, color=WITNESS, alpha=0.15)
        b.axvline(1e3 * stats["x_d"], color=DRIVE, lw=1.5, label="x_d")
        for sign in (-1, 1):
            b.axvline(1e3 * (stats["x_d"] + sign * np.sqrt(stats["var_d"])), color=DRIVE, lw=0.8, ls="--")
        b.axvline(1e3 * stats["x_w"], color=WITNESS, lw=1.5, label="x_w")
        b.legend(fontsize=8, loc="lower right")
    for ax in (a, b):
        ax.set_xlabel("x (mm)")
    a.set_ylabel("row")
    c.set_xlabel("intensity")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig
