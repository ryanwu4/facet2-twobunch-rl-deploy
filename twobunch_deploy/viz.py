"""Plots. They draw only values computed by screen.py.

S2  geometry(): a full frame as the ops display orients it, with the hand crop outlined
S3  background(): a background and its brightest frame; peek(): one frame through the processing, with its energy
    cut and band statistics
S5  step(): one step's median frames, each frame's drive-witness offset, and the TCAV readbacks and tags; S6, S7 add
    the L1 phase setpoints and the L2 subbooster phase setpoints during the step, and the other L1 and L2 PVs as text
S6  phases(): each L1 and L2 PV against time
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import pvs, screen
from .session import PULSE_ID_PERIOD, PULSES_PER_SECOND

DRIVE, WITNESS, CUT = "#eb6834", "#2a9df4", "#1fc7b6"
# TCAV tags (acquire.tag): colour and marker; frames tagged "unmatched" have no readback to plot
TAGS = {"+": ("#2a78d6", "^"), "-": ("#e87ba4", "v"), "neither": ("#eda100", "o"), "off": ("#555555", "s")}
L1_COLOURS = ("#1b9e77", "#7570b3")     # klystrons 11 and 21
# the LI12, LI13 and LI14 subboosters: colour and marker, hollow and of different sizes, as they move together
L2_STYLES = (("#d95f02", "o", 9), ("#e7298a", "s", 6), ("#66a61e", "^", 4))


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


def step(record: dict, medians: dict, updates: list):
    """One step (session.step's record and its capture's updates). Top: the median of each polarity's kept frames
    (processed, row-corrected), with the step's cut, the witness band shaded and the kept frames' median drive and
    witness x centroids. Second: each + and - frame's x_d - x_w, kept frames filled and dropped ones hollow, each
    polarity's median of the kept frames as a line. Third: each frame's TCAV amplitude and phase readback by tag,
    unmatched frames as x on the time axis. Time from the capture's first frame, by pulse ID; the dropped frames'
    spans shaded. Fourth: every update of the two L1 phase setpoints (PREQ) by its timestamp (one from before the first
    frame, as a monitor's first update can be, at 0), their medians as lines, exact zeros off the axis. Bottom: every
    update of the LI12-14 subbooster phase setpoints (PDES) the same way, each minus its median. The other L1 and L2 PVs
    as text, first -> last if they changed."""
    frames = record["per_frame"]
    pulse_ids = np.array(frames["pulse_id"])
    t = (pulse_ids - pulse_ids[0]) % PULSE_ID_PERIOD / PULSES_PER_SECOND
    tags, drop = np.array(frames["tag"]), np.array(frames["drop"])
    mm = 1e3 * screen.x_positions(record["camera"]["RESOLUTION"])
    half = (mm[1] - mm[0]) / 2

    fig = plt.figure(figsize=(13, 18))
    grid = fig.add_gridspec(5, 2, height_ratios=[1.4, 1, 1, 1, 1])
    images = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    offsets = fig.add_subplot(grid[1, :])
    readbacks = [fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])]
    l1 = fig.add_subplot(grid[3, :])
    l2 = fig.add_subplot(grid[4, :])
    for ax, sign in zip(images, ("+", "-")):
        p = record["polarity"][sign]
        ax.set_title(f"{sign}V: median of {p['kept']} kept frames ({p['dropped']} dropped)", fontsize=9)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel(f"row (the padding from {screen.PIXEL_ROWS} not shown)")
        if sign not in medians:
            continue
        ax.imshow(medians[sign], cmap="gray", extent=(mm[0] - half, mm[-1] + half, medians[sign].shape[0] - 0.5, -0.5),
                  aspect="auto", interpolation="nearest")
        ax.set_ylim(screen.PIXEL_ROWS - 0.5, -0.5)
        if record["cut_row"] is not None:
            ax.axhline(record["cut_row"], color=CUT, lw=1.2, label="step cut")
        if "x_d" in p:
            witness_rows = (-0.5, record["cut_row"] + 0.5) if p["witness_low"] else \
                (record["cut_row"] + 0.5, screen.PIXEL_ROWS - 0.5)
            ax.axhspan(*witness_rows, color=WITNESS, alpha=0.15)
            ax.axvline(1e3 * p["x_d"], color=DRIVE, lw=1.5, label="x_d")
            ax.axvline(1e3 * p["x_w"], color=WITNESS, lw=1.5, label="x_w")
        ax.legend(fontsize=8, loc="lower right")

    for sign in ("+", "-"):
        colour, marker = TAGS[sign]
        idx = [i for i, s in enumerate(frames["band_stats"]) if s is not None and tags[i] == sign]
        offset = np.array([1e3 * (frames["band_stats"][i]["x_d"] - frames["band_stats"][i]["x_w"]) for i in idx])
        dropped = drop[idx]
        offsets.plot(t[idx][~dropped], offset[~dropped], marker, color=colour, ms=7, ls="none", label=f"{sign}V kept")
        offsets.plot(t[idx][dropped], offset[dropped], marker, color=colour, mfc="none", ms=7, ls="none",
                     label=f"{sign}V dropped")
        if "x_d - x_w" in record["polarity"][sign]:
            offsets.axhline(1e3 * record["polarity"][sign]["x_d - x_w"], color=colour, lw=1.2)
    offsets.set_ylabel("x_d - x_w (mm)")
    offsets.legend(fontsize=8, ncol=4)

    for ax, key, unit in zip(readbacks, ("amplitude", "phase"), ("MV", "deg")):
        values = np.array([np.nan if v is None else v for v in frames[key]])
        for tag, (colour, marker) in TAGS.items():
            ax.plot(t[tags == tag], values[tags == tag], marker, color=colour, ms=5, ls="none", label=tag)
        bottom = np.nanmin(values) if np.isfinite(values).any() else 0
        ax.plot(t[tags == "unmatched"], np.full(np.sum(tags == "unmatched"), bottom), "x", color="k", ms=6, ls="none",
                label="unmatched")
        ax.set_ylabel(f"TCAV {key} ({unit})")
        ax.legend(fontsize=8, ncol=5)
    for ax in [offsets, *readbacks]:
        for i in np.flatnonzero(drop):
            ax.axvspan(t[i] - 0.05, t[i] + 0.05, color="0.85", lw=0)
        ax.set_xlabel("time from the first frame (s)")

    preq = [name for name in pvs.L1 if name.endswith(":PREQ")]
    for name, colour in zip(preq, L1_COLOURS):
        s = record["phases"][name]
        mine = [u for u in updates if u.name == name]
        t_update = [max(u.timestamp - frames["timestamp"][0], 0) for u in mine]
        l1.plot(t_update, [u.value for u in mine], ".", color=colour, ms=3,
                label=f"{name}: {s['updates']} updates, {s.get('zeros', 0)} exact zeros")
        if "median" in s:
            l1.axhline(s["median"], color=colour, lw=1.2)
    nonzero = [u.value for u in updates if u.name in preq and u.value != 0]
    if nonzero:
        l1.set_ylim(min(nonzero) - 0.1, max(nonzero) + 0.1)
    l1.set_xlabel("time from the first frame (s), by timestamp")
    l1.set_ylabel("L1 phase setpoint (deg)")
    l1.legend(fontsize=8, ncol=2)
    l1.set_title("; ".join(f"{name} {changed(record['phases'][name])}" for name in pvs.L1 if name not in preq),
                 fontsize=8)

    # The subbooster setpoints the L2 multiknob moves, each minus its median: on 11-04 they sat about 2 deg apart and
    # moved together by about 0.05 deg. LI11's (0; the multiknob doesn't move it) and the readbacks go in the title.
    moved = [name for name in pvs.L2 if name.endswith(":PDES") and not name.startswith("LI11:")]
    for name, (colour, marker, size) in zip(moved, L2_STYLES):
        s = record["phases"][name]
        mine = [u for u in updates if u.name == name]
        t_update = [max(u.timestamp - frames["timestamp"][0], 0) for u in mine]
        l2.plot(t_update, [u.value - s["median"] for u in mine], marker, color=colour, mfc="none", ms=size, ls="none",
                label=f"{name}: median {s['median']:.3f}, {s['updates']} updates" if mine else f"{name}: no update")
    l2.axhline(0, color="0.5", lw=1)
    l2.set_xlabel("time from the first frame (s), by timestamp")
    l2.set_ylabel("L2 subbooster setpoint\nminus its median (deg)")
    l2.legend(fontsize=8, ncol=3)
    rest = [[name for name in pvs.L2 if name not in moved and name.endswith(kind)] for kind in (":PDES", ":PHAS")]
    l2.set_title("\n".join("; ".join(f"{name} {changed(record['phases'][name])}" for name in names) for names in rest),
                 fontsize=8)

    r = record["reading"]
    failed = [name for name, ok in record["checks"].items() if not ok]
    fig.suptitle(f"step: {'valid' if record['valid'] else 'invalid (' + '; '.join(failed) + ')'}\n"
                 + ("no reading" if r is None else
                    f"spacing {r['spacing_um']:.1f} um, drive sigma_z {r['drive_sigma_z_um']:.1f} um, drive fraction "
                    f"{r['drive_charge_fraction']:.3f}, edge fraction {r['edge_frac']:.4f}, cut sensitivity "
                    f"{r['cut_sens_um']:.1f} um")
                 + f"; tags {record['tags']}", fontsize=10)
    fig.tight_layout()
    return fig


def changed(summary: dict) -> str:
    """A PV over a capture (acquire.summarise): its first value, "first -> last" if it changed, or "no update"."""
    if not summary["updates"]:
        return "no update"
    first, last = summary["first"], summary["last"]
    return f"{first:.3f}" if first == last else f"{first:.3f} -> {last:.3f}"


def phases(updates: list, names: list[str], start: float, seconds: float):
    """S6, S7 (session.phases_test): every update of each PV in `names` against time from `start`, one panel per kind
    of PV (the part of the name after the last colon). An update from before `start` (a monitor's first update is the
    PV's current value) is drawn at 0; exact zeros (BSA dropouts) are off the axis."""
    kinds = {}
    for name in names:
        kinds.setdefault(name.rsplit(":", 1)[1], []).append(name)
    fig, axes = plt.subplots((len(kinds) + 1) // 2, 2, figsize=(13, 10), sharex=True, squeeze=False)
    for ax, members in zip(axes.flat, kinds.values()):
        for name in members:
            mine = [u for u in updates if u.name == name]
            ax.plot([max(u.timestamp - start, 0) for u in mine], [u.value for u in mine], ".-", ms=3, lw=0.6,
                    drawstyle="steps-post", label=f"{name}: {len(mine)} updates")
        nonzero = [u.value for u in updates if u.name in members and u.value != 0]
        if nonzero:
            ax.set_ylim(min(nonzero) - 0.1, max(nonzero) + 0.1)
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel(f"time from the start of the {seconds} s (s)")
    fig.suptitle(f"L1 and L2 PVs over {seconds} s", fontsize=10)
    fig.tight_layout()
    return fig
