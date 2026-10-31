"""Plots. They draw only values computed by screen.py.

S2  geometry(): a full frame as the ops display orients it, with the hand crop outlined
Planned (S3, S5): peek(), watch(), reading.show()
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import screen


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
