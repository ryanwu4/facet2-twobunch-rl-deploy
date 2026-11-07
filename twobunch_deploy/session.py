"""Calls the notebooks make. S0: where and with what a session runs, and saving records under data/.
S1: capture_test() of the camera frames and pulse IDs. S2: geometry_test() of the camera settings and the
hand crop. S3: backgrounds (take_background, pedestal, load_background), peek() and processing_test().
S4: tcav_test() of the TCAV readbacks and tags; peek() shows the newest frame's tag. S5: step(). S6, S7: phases_test()
of the L1 and L2 PVs; step() records them. S8: drift_test() of the bunch-length and drift PVs and optics_test() of the
magnets; step() records both. S9: check_pvs() and status() for the console; watch() and peek_strip() to look at the
frames between steps. W2 (v2): hold_test() and staircase() for the write sessions with the beam stopped at TD11; the
writes themselves are in actuate.py."""
from __future__ import annotations

import datetime
import getpass
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
import time
from collections import Counter
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

# (module the package imports, its distribution name for the version)
DEPENDENCIES = [("numpy", "numpy"), ("scipy", "scipy"), ("matplotlib", "matplotlib"), ("epics", "pyepics")]
# Channel Access settings. pyepics sets EPICS_CA_MAX_ARRAY_BYTES itself on its first CA call when it is
# unset, so check_env reads these before importing anything.
EPICS_VARS = ["EPICS_CA_ADDR_LIST", "EPICS_CA_AUTO_ADDR_LIST", "EPICS_CA_SERVER_PORT",
              "EPICS_CA_MAX_ARRAY_BYTES", "PYEPICS_LIBCA"]
# Pulse IDs count the 360 Hz pulses from 0 to 131039, then wrap. At 10 Hz, consecutive frames are 36 apart.
PULSE_ID_PERIOD = 131040
PULSE_IDS_PER_FRAME = 36
PULSES_PER_SECOND = 360


def check_env() -> dict:
    """Host, user, Python, package versions and EPICS settings of this session, and the code it runs.

    Imports the runtime dependencies (importing pyepics loads no Channel Access library) and reads no PV.
    """
    epics_env = {k: os.environ.get(k) for k in EPICS_VARS}
    env = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "host": socket.gethostname(),
           "user": getpass.getuser(), "python": sys.version.split()[0], "executable": sys.executable,
           "platform": platform.platform(), "git_commit": git_commit(), "code_sha256": code_sha256(),
           "epics_env": epics_env, "packages": {}, "import_errors": {}}
    for module, dist in DEPENDENCIES:
        try:
            importlib.import_module(module)
            env["packages"][dist] = importlib.metadata.version(dist)
        except Exception as e:                      # record every failure, not only the first
            env["import_errors"][dist] = repr(e)
    return env


def git_commit() -> str:
    """Commit checked out in REPO, read from .git without running git; "unknown" if it can't be read."""
    try:
        head = (REPO / ".git" / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head                             # detached HEAD
        ref = head.removeprefix("ref: ")
        if (REPO / ".git" / ref).exists():
            return (REPO / ".git" / ref).read_text().strip()
        for line in (REPO / ".git" / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return "unknown"


def code_sha256() -> str:
    """sha256 over the package's .py files (name and content, sorted by name).

    Identifies the code that runs whether or not it is committed; `pixi run code-hash` and CI print the
    same value for a checkout.
    """
    h = hashlib.sha256()
    for f in sorted((REPO / "twobunch_deploy").glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def save_record(stage: str, name: str, record: dict) -> Path:
    """Write record as JSON to data/<date>_<stage>/<name>_<HHMMSS>.json and return the path."""
    now = datetime.datetime.now()
    path = DATA / f"{now:%Y-%m-%d}_{stage}" / f"{name}_{now:%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1))
    return path


def pulse_id_steps(pulse_ids: list[int]) -> dict[int, int]:
    """How often each step between consecutive pulse IDs occurs, counted across the wrap.

    36 is the next frame and 72 one frame missed. The IOC sometimes stamps a frame with a neighbouring
    pulse ID: in the 33,779 frames of the DAQ runs, two consecutive frames shared an ID 6 times (a 0 step,
    with a 72 just before it in TEST_15668) and a frame was one pulse early twice (a 35 step, then a 37).
    """
    steps = Counter((b - a) % PULSE_ID_PERIOD for a, b in zip(pulse_ids, pulse_ids[1:]))
    return dict(sorted(steps.items()))


def capture_test(seconds: float = 30, n_saved: int = 10) -> dict:
    """S1: capture the DTOTR2 camera for `seconds` and summarise what arrived.

    Saves the summary to data/<date>_S1/capture_test_<HHMMSS>.json and, next to it (.npz), every frame's
    pulse ID, timestamp and arrival time, plus `n_saved` raw frames spread over the capture.
    """
    import numpy as np                  # imported here, so that check_env() runs even without numpy or pyepics
    from . import machine

    frames = machine.frames(seconds)
    summary = {"seconds": seconds, "frames": len(frames)}
    if frames:
        steps = pulse_id_steps([f.pulse_id for f in frames])
        expected = sum(s * n for s, n in steps.items()) // PULSE_IDS_PER_FRAME + 1
        latency = [f.received - f.timestamp for f in frames]
        summary.update({
            "capture_fraction": len(frames) / expected,
            "pulse_id_steps": steps,
            "latency_s": {"first": latency[0], "median": float(np.median(latency)), "max": max(latency)},
            "dtype": str(frames[0].data.dtype),
            "lengths": sorted({f.data.size for f in frames}),
            "min": int(min(f.data.min() for f in frames)),
            "max": int(max(f.data.max() for f in frames)),
        })
    path = save_record("S1", "capture_test", summary)
    if frames:
        saved = frames[::max(1, len(frames) // n_saved)][:n_saved]
        np.savez_compressed(path.with_suffix(".npz"), pulse_id=[f.pulse_id for f in frames],
                            timestamp=[f.timestamp for f in frames], received=[f.received for f in frames],
                            saved_pulse_id=[f.pulse_id for f in saved],
                            saved_data=np.stack([f.data for f in saved]))
    return summary


def geometry_test(seconds: float = 3) -> dict:
    """S2: the camera settings, then the hand crop of every frame captured for `seconds` (screen.trained_frame).

    Saves the summary to data/<date>_S2/geometry_test_<HHMMSS>.json and, next to it, every frame's pulse ID,
    the newest raw frame and every frame's hand crop (.npz), and viz.geometry's figure of the newest frame
    (.png).
    """
    import numpy as np                  # imported here, so that check_env() runs even without them
    from . import machine, screen, viz

    camera = machine.camera_settings()
    frames = machine.frames(seconds)
    if not frames:
        raise RuntimeError(f"no frame in {seconds} s: is the camera publishing?")
    trained = np.stack([screen.trained_frame(f.data, camera) for f in frames])
    summary = {"camera": camera, "trained": screen.TRAINED,
               "differs": [k for k in screen.TRAINED if camera[k] != screen.TRAINED[k]],
               "frames": len(frames), "trained_shape": list(trained.shape[1:])}
    path = save_record("S2", "geometry_test", summary)
    np.savez_compressed(path.with_suffix(".npz"), pulse_id=[f.pulse_id for f in frames], raw=frames[-1].data,
                        trained=trained)
    fig = viz.geometry(frames[-1].data, camera, f"DTOTR2, pulse ID {frames[-1].pulse_id}")
    fig.savefig(path.with_suffix(".png"))
    return summary


class Background(NamedTuple):
    """A background for screen.process, as take_background(), pedestal() and load_background() return it."""
    image: np.ndarray   # subtracted from each hand crop; 0 in the padding rows
    path: str           # the .npz it is saved in
    record: dict        # the JSON record saved next to it: how it was taken, and the camera settings then


def hand_crops(seconds: float):
    """The camera readbacks, every frame captured for `seconds` after them, and each frame's hand crop."""
    import numpy as np
    from . import machine, screen

    camera = machine.camera_settings()
    frames = machine.frames(seconds)
    if not frames:
        raise RuntimeError(f"no frame in {seconds} s: is the camera publishing?")
    return camera, frames, np.stack([screen.trained_frame(f.data, camera) for f in frames])


def take_background(seconds: float = 5) -> Background:
    """S3: a background taken with no beam on DTOTR2 (ops block it, or a beam-off window): the mean of the hand
    crops of every frame captured for `seconds`.

    Nothing here checks that the frames have no beam: the S3 checklist does, from the record and the figure.
    Saves data/<date>_S3/background_<HHMMSS>.json, .npz and .png (save_background).
    """
    camera, frames, crops = hand_crops(seconds)
    return save_background("background", crops.mean(axis=0), camera, frames, crops)


def pedestal(seconds: float = 5) -> Background:
    """S3: a constant background, for a session without a beam-off window: the median of every hand-crop pixel of
    every frame captured for `seconds` with beam on (screen.pedestal).

    Records the fraction of pixels THRESHOLD or more above the median: the beam's share of the crop, a few percent
    in the DAQ runs. Saves data/<date>_S3/pedestal_<HHMMSS>.json, .npz and .png (save_background).
    """
    import numpy as np
    from . import screen

    camera, frames, crops = hand_crops(seconds)
    image = screen.pedestal(crops)
    rows = screen.PIXEL_ROWS
    above = float(np.mean(crops[:, :rows] - image[:rows] >= screen.THRESHOLD))
    return save_background("pedestal", image, camera, frames, crops, fraction_above_threshold=above)


def save_background(name: str, image: np.ndarray, camera: dict, frames: list, crops: np.ndarray,
                    **extra) -> Background:
    """Save a background to data/<date>_S3/<name>_<HHMMSS>: the record (.json: camera settings, frame count, each
    frame's max and mean, the background's min, mean and max, and `extra`), the background with every frame's
    hand crop and pulse ID (.npz), and viz.background's figure (.png)."""
    import numpy as np
    from . import screen, viz

    pixels = image[:screen.PIXEL_ROWS]
    record = {"camera": camera, "frames": len(frames), "frame_max": [int(c.max()) for c in crops],
              "frame_mean": [round(float(c[:screen.PIXEL_ROWS].mean()), 2) for c in crops],
              "background": {"min": float(pixels.min()), "mean": float(pixels.mean()), "max": float(pixels.max())},
              **extra}
    path = save_record("S3", name, record)
    np.savez_compressed(path.with_suffix(".npz"), background=image, crops=crops,
                        pulse_id=[f.pulse_id for f in frames])
    brightest = int(np.argmax(record["frame_max"]))
    fig = viz.background(image, crops[brightest], f"{path.stem}: {len(frames)} frames; "
                         f"brightest pulse ID {frames[brightest].pulse_id}")
    fig.savefig(path.with_suffix(".png"))
    return Background(image, str(path.with_suffix(".npz")), record)


def load_background(path: str | Path) -> Background:
    """S3: a background saved by take_background() or pedestal() (its .npz), with its record."""
    import numpy as np

    path = Path(path)
    with np.load(path) as saved:
        image = saved["background"]
    return Background(image, str(path), json.loads(path.with_suffix(".json").read_text()))


def peek(background: Background, seconds: float = 0.5) -> dict:
    """S3: the newest frame through the processing, drawn with its energy cut and band statistics (newest). S4: with
    its TCAV tag and readbacks.

    Captures for `seconds` with the TCAV readbacks (acquire.capture) and takes the last frame. Saves the summary to
    data/<date>_S3/peek_<HHMMSS>.json and, next to it, the raw frame and its hand crop (.npz) and the figure (.png).
    """
    import numpy as np
    from . import acquire

    capture = acquire.capture(seconds)
    if not capture.frames:
        raise RuntimeError(f"no frame in {seconds} s: is the camera publishing?")
    summary, crop, fig = newest(capture, background)
    path = save_record("S3", "peek", summary)
    np.savez_compressed(path.with_suffix(".npz"), raw=capture.frames[-1].data, crop=crop)
    fig.savefig(path.with_suffix(".png"))
    return summary


def newest(capture, background: Background):
    """A capture's last frame through the processing, for peek() and watch(): its summary, its hand crop and its figure
    (viz.peek). The cut is the frame's own valley and the frame is not row-corrected: one frame has neither a step cut
    nor an off reference (step() has both)."""
    import numpy as np
    from . import screen, viz

    frame, camera = capture.frames[-1], capture.camera
    crop = screen.trained_frame(frame.data, camera)
    start = time.perf_counter()
    processed = screen.process(crop, background.image)
    valley = screen.valley(processed.sum(axis=1))
    x = screen.x_positions(camera["RESOLUTION"])
    stats = None if valley.cut is None else screen.band_stats(processed, x, valley.cut)
    processing_s = time.perf_counter() - start
    amplitude, phase = float(capture.amplitude[-1]), float(capture.phase[-1])
    summary = {"pulse_id": frame.pulse_id, "age_s": frame.received - frame.timestamp, "frames": len(capture.frames),
               "tcav": {"tag": str(capture.tags[-1]), "amplitude": None if np.isnan(amplitude) else amplitude,
                        "phase": None if np.isnan(phase) else phase},
               "background": background.path, "camera": camera, "processing_s": processing_s,
               "intensity": float(processed.sum()), "band_stats": stats}
    title = (f"DTOTR2, pulse ID {frame.pulse_id}, {summary['age_s']:.2f} s old; TCAV {capture.tags[-1]} "
             f"(A {amplitude:.2f} MV, P {phase:.2f} deg); own valley, no row correction\n"
             + ("no valley: no cut" if stats is None else
                f"x_d - x_w {1e3 * (stats['x_d'] - stats['x_w']):.3f} mm, drive fraction {stats['frac_d']:.3f}, "
                f"edge fraction {stats['edge_frac']:.4f}"))
    return summary, crop, viz.peek(crop, processed, valley, stats, x, title)


def watch(background: Background, seconds: float = 30, every_s: float = 1.0) -> list:
    """S9: peek() about every `every_s` for `seconds`, each figure replacing the last in the notebook. Each look
    captures 0.5 s, subscribing to the camera and the monitored PVs, so it takes longer than that. Holds the kernel
    while it runs: only between steps.

    Saves no frames: each look's pulse ID, TCAV tag, x_d - x_w (m) and drive fraction (None without a cut), to
    data/<date>_S3/watch_<HHMMSS>.json. Returns the same list.
    """
    import matplotlib.pyplot as plt
    from IPython.display import clear_output, display
    from . import acquire

    looks = []
    end = time.time() + seconds
    while time.time() < end:
        start = time.time()
        capture = acquire.capture(0.5)
        if capture.frames:
            summary, _, fig = newest(capture, background)
            clear_output(wait=True)
            display(fig)
            plt.close(fig)
            stats = summary["band_stats"]
            looks.append({"pulse_id": summary["pulse_id"], "tag": summary["tcav"]["tag"],
                          "x_d - x_w": None if stats is None else stats["x_d"] - stats["x_w"],
                          "frac_d": None if stats is None else stats["frac_d"]})
        time.sleep(max(0.0, every_s - (time.time() - start)))
    save_record("S3", "watch", {"seconds": seconds, "every_s": every_s, "background": background.path,
                                "looks": looks})
    return looks


def peek_strip(background: Background, seconds: float = 4, streak_um_per_deg: float | None = None) -> dict:
    """S9: every frame of a `seconds` capture through the processing, each with its own valley as in peek(): the
    frames as small images with their tags (viz.strip), and each + and - frame's x_d - x_w against time. Given
    `streak_um_per_deg`, also one spacing from the newest + frame with a cut and the newest - frame, read at the +
    frame's cut (screen.reading): one pair without row correction, so not the controller's reading, which takes the
    medians of a step's frames (step()).

    Saves the summary (each frame's pulse ID, tag, cut and band statistics) to data/<date>_S3/peek_strip_<HHMMSS>.json
    and, next to it, every frame's hand crop, pulse ID and tag (.npz) and the figure (.png).
    """
    import numpy as np
    from . import acquire, screen, viz

    capture = acquire.capture(seconds)
    if not capture.frames:
        raise RuntimeError(f"no frame in {seconds} s: is the camera publishing?")
    x = screen.x_positions(capture.camera["RESOLUTION"])
    crops = np.stack([screen.trained_frame(f.data, capture.camera) for f in capture.frames])
    processed = [screen.process(crop, background.image) for crop in crops]
    cuts = [screen.valley(image.sum(axis=1)).cut for image in processed]
    stats = [None if cut is None else screen.band_stats(image, x, cut) for image, cut in zip(processed, cuts)]
    pair = None
    plus = [i for i, t in enumerate(capture.tags) if t == "+" and stats[i] is not None]
    minus = [i for i, t in enumerate(capture.tags) if t == "-"]
    if streak_um_per_deg is not None and plus and minus:
        p, m = plus[-1], minus[-1]
        at_cut = screen.band_stats(processed[m], x, cuts[p])
        if at_cut is not None:
            r = screen.reading([stats[p]], [at_cut], streak_um_per_deg)
            pair = {"plus_pulse_id": capture.frames[p].pulse_id, "minus_pulse_id": capture.frames[m].pulse_id,
                    "spacing_um": 1e6 * r["spacing"]}
    pulse_ids = [f.pulse_id for f in capture.frames]
    summary = {"seconds": seconds, "frames": len(capture.frames), "tags": dict(Counter(capture.tags)),
               "background": background.path, "camera": capture.camera, "streak_um_per_deg": streak_um_per_deg,
               "one_pair": pair, "per_frame": {"pulse_id": pulse_ids, "tag": [str(t) for t in capture.tags],
                                               "cut": cuts, "band_stats": stats}}
    path = save_record("S3", "peek_strip", summary)
    np.savez_compressed(path.with_suffix(".npz"), crops=crops, pulse_id=pulse_ids, tag=capture.tags.astype(str))
    t = [(pulse_id - pulse_ids[0]) % PULSE_ID_PERIOD / PULSES_PER_SECOND for pulse_id in pulse_ids]
    title = (f"{len(capture.frames)} frames, own valleys, no row correction; "
             + ("no spacing" if pair is None else
                f"one pair's spacing {pair['spacing_um']:.1f} um (not the controller reading)"))
    viz.strip(processed, capture.tags, stats, t, x, title).savefig(path.with_suffix(".png"))
    return summary


def processing_test(backgrounds: list[Background], seconds: float = 10) -> dict:
    """S3: every frame captured for `seconds` through the processing, once with each background. For each: the
    fraction of frames whose own valley gives a cut, the processing time per frame (hand crop to band
    statistics), and the medians of the summed intensity and of the band statistics.

    Saves the summary to data/<date>_S3/processing_test_<HHMMSS>.json and, next to it, every frame's pulse ID
    and hand crop (.npz): with the backgrounds' .npz, each frame's processing can be redone offline.
    """
    import numpy as np
    from . import machine, screen

    camera = machine.camera_settings()
    frames = machine.frames(seconds)
    if not frames:
        raise RuntimeError(f"no frame in {seconds} s: is the camera publishing?")
    x = screen.x_positions(camera["RESOLUTION"])
    summary = {"camera": camera, "frames": len(frames), "backgrounds": []}
    for background in backgrounds:
        stats, intensity, seconds_per_frame = [], [], []
        for f in frames:
            start = time.perf_counter()
            processed = screen.process(screen.trained_frame(f.data, camera), background.image)
            cut = screen.valley(processed.sum(axis=1)).cut
            stats.append(None if cut is None else screen.band_stats(processed, x, cut))
            seconds_per_frame.append(time.perf_counter() - start)
            intensity.append(float(processed.sum()))
        found = [{**s, "x_d - x_w": s["x_d"] - s["x_w"]} for s in stats if s is not None]
        keys = ("x_d", "x_w", "x_d - x_w", "frac_d", "edge_frac")
        summary["backgrounds"].append({
            "background": background.path, "cut_found": len(found) / len(frames),
            "processing_s": {"median": float(np.median(seconds_per_frame)), "max": max(seconds_per_frame)},
            "intensity_median": float(np.median(intensity)),
            "median": {k: float(np.median([s[k] for s in found])) for k in keys} if found else {}})
    path = save_record("S3", "processing_test", summary)
    np.savez_compressed(path.with_suffix(".npz"), pulse_id=[f.pulse_id for f in frames],
                        crops=np.stack([screen.trained_frame(f.data, camera) for f in frames]))
    return summary


def tcav_test(seconds: float = 10) -> dict:
    """S4: capture for `seconds` with the TCAV amplitude and phase readbacks (acquire.capture) and summarise how they
    arrive, in whatever state the TCAV is: each PV's number of updates and the pulse-ID steps between them; the
    fraction of frames with both readbacks of their pulse; the tags and the frames dropped; each readback's minimum,
    median and maximum per tag.

    Saves the summary to data/<date>_S4/tcav_test_<HHMMSS>.json and, next to it (.npz), every frame's pulse ID,
    timestamp, readbacks, tag and drop, and every update.
    """
    import numpy as np
    from . import acquire, pvs

    capture = acquire.capture(seconds)
    if not capture.frames:
        raise RuntimeError(f"no frame in {seconds} s: is the camera publishing?")
    summary = {"seconds": seconds, "frames": len(capture.frames), "updates": {}}
    for name in (pvs.TCAV_AMPLITUDE, pvs.TCAV_PHASE):
        pulse_ids = [u.pulse_id for u in capture.updates if u.name == name]
        summary["updates"][name] = {"count": len(pulse_ids), "pulse_id_steps": pulse_id_steps(pulse_ids)}
    matched = ~(np.isnan(capture.amplitude) | np.isnan(capture.phase))
    summary.update({"matched_fraction": float(matched.mean()), "tags": dict(Counter(capture.tags)),
                    "dropped": int(capture.drop.sum()), "range": {}})
    for t in summary["tags"]:
        if t != "unmatched":
            summary["range"][t] = {k: [float(f(v[capture.tags == t])) for f in (np.min, np.median, np.max)]
                                   for k, v in (("amplitude", capture.amplitude), ("phase", capture.phase))}
    path = save_record("S4", "tcav_test", summary)
    np.savez_compressed(path.with_suffix(".npz"), **frame_arrays(capture))
    return summary


def frame_arrays(capture) -> dict:
    """A capture's per-frame and per-update arrays, for an .npz."""
    import numpy as np

    return {"pulse_id": [f.pulse_id for f in capture.frames], "timestamp": [f.timestamp for f in capture.frames],
            "amplitude": capture.amplitude, "phase": capture.phase, "tag": capture.tags.astype(str),
            "drop": capture.drop, "update_name": [u.name for u in capture.updates],
            "update_pulse_id": [u.pulse_id for u in capture.updates],
            "update_timestamp": [u.timestamp for u in capture.updates],
            "update_value": [u.value for u in capture.updates]}


class Reading(NamedTuple):
    """A step()'s reading: its record (saved as JSON at `path`), each polarity's median frame and the capture's
    updates (machine.Update), for show()."""
    record: dict
    path: str
    medians: dict
    updates: list

    def show(self) -> None:
        """Draw the step (viz.step) and save the figure next to its record (.png)."""
        from . import viz

        viz.step(self.record, self.medians, self.updates).savefig(Path(self.path).with_suffix(".png"))


def step(background: Background, streak_um_per_deg: float, seconds: float = 15, drop_s: float = 1.0) -> Reading:
    """S5: one measurement step. Captures for `seconds` with the TCAV readbacks (acquire.capture: each frame tagged,
    and the + and - frames less than `drop_s` after their polarity window began dropped), then reads the frames as
    the deployed analysis does (screen.read_step) with `background` and the TCAV calibration `streak_um_per_deg`.
    The calibration is in um on the screen per degree at the live pixel size (RESOLUTION): one measured at another
    pixel size must be rescaled first.

    Without both polarities, an off frame or a cut the reading is invalid (its checks say which), but everything the
    frames allow is still computed and recorded. Saves data/<date>_S5/step_<HHMMSS>.json (the reading and its checks,
    counts, the reference fit and its off frames' beam width, the summary of each L1 and L2 PV and each bunch-length
    and drift PV over the capture (acquire.summarise), each magnet read just before it (None if it didn't answer),
    timing, peak memory, and each frame's pulse ID, timestamp, readbacks, tag, drop, intensity, beam x RMS and band
    statistics) and .npz (every frame's hand crop with the same, every update, the reference and the two median
    frames).
    """
    import resource
    import numpy as np
    from . import acquire, machine, pvs, screen

    start = time.perf_counter()
    magnets = machine.describe(list(pvs.MAGNETS))
    capture = acquire.capture(seconds, drop_s)
    if not capture.frames:
        raise RuntimeError(f"no frame in {seconds} s: is the camera publishing?")
    captured = time.perf_counter()
    crops = np.stack([screen.trained_frame(f.data, capture.camera) for f in capture.frames])
    x = screen.x_positions(capture.camera["RESOLUTION"])
    read = screen.read_step(crops, capture.tags, capture.drop, background.image, x, streak_um_per_deg)
    done = time.perf_counter()

    reference = None
    if read.reference is not None:
        fitted = np.isfinite(read.reference.centroids)
        shifts = read.reference.shifts[fitted]
        widths = [1e3 * w for w, t in zip(read.x_rms, capture.tags) if t == "off" and w is not None]
        reference = {"off_frames": int(np.sum(capture.tags == "off")), "rows_fitted": int(fitted.sum()),
                     "shift_columns": {"min": float(shifts.min()), "max": float(shifts.max())},
                     "off_x_rms_mm": {"median": float(np.median(widths)), "max": max(widths)}}
    polarity = {}
    for sign in ("+", "-"):
        kept = [i for i, (t, d) in enumerate(zip(capture.tags, capture.drop)) if t == sign and not d]
        stats = [read.stats[i] for i in kept if read.stats[i] is not None]
        polarity[sign] = {"kept": len(stats), "dropped": int(np.sum((capture.tags == sign) & capture.drop))}
        if kept:
            polarity[sign]["intensity"] = float(np.median([read.intensity[i] for i in kept]))
        if stats:
            polarity[sign].update({k: float(np.median([s[k] for s in stats])) for k in ("x_d", "x_w")})
            polarity[sign]["x_d - x_w"] = float(np.median([s["x_d"] - s["x_w"] for s in stats]))
            polarity[sign]["witness_low"] = bool(np.mean([s["witness_low"] for s in stats]) > 0.5)
    r = read.reading
    record = {
        "background": background.path, "streak_um_per_deg": streak_um_per_deg, "seconds": seconds, "drop_s": drop_s,
        "camera": capture.camera, "frames": len(capture.frames),
        "updates": dict(Counter(u.name for u in capture.updates)), "tags": dict(Counter(capture.tags)),
        "reference": reference, "cut_row": read.cut, "polarity": polarity,
        "reading": None if r is None else {
            "spacing_um": 1e6 * r["spacing"], "drive_sigma_z_um": 1e6 * r["drive_sigma_z"],
            "drive_charge_fraction": r["drive_charge_fraction"], "edge_frac": r["edge_frac"],
            "cut_sens_um": 1e6 * r["cut_sens"]},
        "checks": read.checks, "valid": all(read.checks.values()),
        "phases": {name: acquire.summarise(capture.updates, name) for name in [*pvs.L1, *pvs.L2]},
        "drift": {name: acquire.summarise(capture.updates, name) for name in [*pvs.BLEN, *pvs.DRIFT]},
        "magnets": {name: None if m is None else m["value"] for name, m in magnets.items()},
        "timing_s": {"capture": captured - start, "reading": done - captured},
        "peak_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "per_frame": {"pulse_id": [f.pulse_id for f in capture.frames],
                      "timestamp": [f.timestamp for f in capture.frames],
                      "amplitude": [None if np.isnan(a) else float(a) for a in capture.amplitude],
                      "phase": [None if np.isnan(p) else float(p) for p in capture.phase],
                      "tag": [str(t) for t in capture.tags], "drop": capture.drop.tolist(),
                      "intensity": read.intensity, "x_rms": read.x_rms, "band_stats": read.stats}}
    path = save_record("S5", "step", record)
    arrays = {"crops": crops, **frame_arrays(capture), **{f"median_{s}": m for s, m in read.medians.items()}}
    if read.reference is not None:
        arrays.update(reference_image=read.reference.image, reference_centroids=read.reference.centroids,
                      reference_shifts=read.reference.shifts)
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    return Reading(record, str(path), read.medians, capture.updates)


def phases_test(seconds: float = 60) -> dict:
    """S6, S7: every update of the L1 and L2 PVs (pvs.L1, pvs.L2) during `seconds`, without the camera (monitor_test).
    Saves to data/<date>_S6/phases_test_<HHMMSS>."""
    from . import pvs

    return monitor_test("S6", "phases_test", [*pvs.L1, *pvs.L2], seconds)


def drift_test(seconds: float = 60) -> dict:
    """S8: every update of the bunch-length and drift PVs (pvs.BLEN, pvs.DRIFT) during `seconds`, without the camera
    (monitor_test). Saves to data/<date>_S8/drift_test_<HHMMSS>."""
    from . import pvs

    return monitor_test("S8", "drift_test", [*pvs.BLEN, *pvs.DRIFT], seconds)


def monitor_test(stage: str, name: str, names: list[str], seconds: float) -> dict:
    """Every update of the PVs `names` during `seconds`, without the camera. For each PV, what step() records
    (acquire.summarise) with the standard deviation, minimum and maximum of the finite values, the pulse-ID steps
    between updates and the age of the first update (for a setpoint, the time since it was last written).

    Saves the summary to data/<date>_<stage>/<name>_<HHMMSS>.json and, next to it, every update (.npz) and each PV
    against time (viz.phases, .png).
    """
    import numpy as np
    from . import acquire, machine, viz

    start = time.time()
    with machine.updates(names) as updates:
        time.sleep(seconds)
    updates = list(updates)
    summary = {"seconds": seconds, "pvs": {}}
    for pv in names:
        summary["pvs"][pv] = acquire.summarise(updates, pv)
        mine = [u for u in updates if u.name == pv]
        finite = [u.value for u in mine if np.isfinite(u.value)]
        if finite:
            summary["pvs"][pv].update(std=float(np.std(finite)), min=float(min(finite)), max=float(max(finite)))
        if mine:
            summary["pvs"][pv].update(pulse_id_steps=pulse_id_steps([u.pulse_id for u in mine]),
                                      first_age_s=start - mine[0].timestamp)
    path = save_record(stage, name, summary)
    np.savez_compressed(path.with_suffix(".npz"), name=[u.name for u in updates],
                        pulse_id=[u.pulse_id for u in updates], timestamp=[u.timestamp for u in updates],
                        value=[u.value for u in updates])
    viz.phases(updates, names, start, seconds).savefig(path.with_suffix(".png"))
    return summary


def optics_test() -> dict:
    """S8: one read of each magnet supply (pvs.MAGNETS) against its value in the DAQ runs (pvs.MAGNETS_IN_DAQ): the
    live value, the DAQ's, and the difference, in the supply's units and in percent of the DAQ's (large for the
    correctors, which sit near 0); None for a supply that doesn't answer.

    Saves the summary to data/<date>_S8/optics_test_<HHMMSS>.json.
    """
    from . import machine, pvs

    summary = {}
    for name, m in machine.describe(list(pvs.MAGNETS)).items():
        daq = pvs.MAGNETS_IN_DAQ[name]
        summary[name] = None if m is None else {"value": m["value"], "daq": daq, "units": m["units"],
                                                "difference": m["value"] - daq,
                                                "difference_pct": 100 * (m["value"] - daq) / abs(daq)}
    save_record("S8", "optics_test", summary)
    return summary


def check_pvs() -> dict:
    """S9: one read of every PV in pvs.READ (machine.describe), for the start of a session: how many answer, which
    don't, and each one's value (an array's first element), element count, units, host, alarm severity and age (seconds
    since its timestamp: for a setpoint, since it was last written; for the gateway's PVs, since the gateway's update).

    Saves the summary to data/<date>_S9/check_pvs_<HHMMSS>.json.
    """
    from . import machine, pvs

    got = machine.describe(list(pvs.READ))
    now = time.time()
    summary = {"answered": sum(m is not None for m in got.values()), "pvs": len(got),
               "missing": [name for name, m in got.items() if m is None],
               "read": {name: None if m is None else {**m, "age_s": now - m["timestamp"]} for name, m in got.items()}}
    save_record("S9", "check_pvs", summary)
    return summary


def status() -> dict:
    """S9: today's session from its records on disk (data/<date>_*/), so it survives a kernel restart: how many
    records and when the first and latest were saved, how many steps were taken and how many were valid, and the
    latest step's path, reading, streak calibration and background."""
    today = f"{datetime.date.today():%Y-%m-%d}"
    records = sorted(DATA.glob(f"{today}_*/*.json"), key=lambda p: p.stat().st_mtime)
    step_paths = sorted(DATA.glob(f"{today}_S5/step_*.json"))
    steps = [json.loads(p.read_text()) for p in step_paths]
    saved = [datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds") for p in records]
    summary = {"date": today, "records": len(records), "first": saved[0] if saved else None,
               "latest": saved[-1] if saved else None, "steps": len(steps),
               "valid_steps": sum(step["valid"] for step in steps)}
    if steps:
        last = steps[-1]
        summary["last_step"] = {"path": str(step_paths[-1]), "valid": last["valid"], "reading": last["reading"],
                                "streak_um_per_deg": last["streak_um_per_deg"], "background": last["background"]}
    return summary


def hold_test(seconds: float = 600) -> dict:
    """W2: every update of the L1, L2 and bunch-length PVs during `seconds` with nothing written (monitor_test): with
    PREQ's feedback off, the L1 readbacks' scatter and the BC11 bunch-length drift. Saves to
    data/<date>_V2/hold_test_<HHMMSS>."""
    from . import pvs

    return monitor_test("V2", "hold_test", [*pvs.L1, *pvs.L2, *pvs.BLEN], seconds)


def staircase(kind: str, step: float, n: int, hold_s: float = 10) -> dict:
    """W2: `n` moves of `step` deg on L1 or L2 (actuate.move_l1, move_l2), then one more, which the budget must refuse,
    then `n` of -`step` back; `hold_s` of nothing before each move and after the last. Every update of the L1, L2 and
    bunch-length PVs is recorded throughout. For each move: its ledger entry, and the median of each of those PVs over
    the hold before it and the hold after it (each move's effect on the L1 readbacks, the BC11 bunch length and the
    subbooster setpoints). Needs an armed session.

    Saves the summary to data/<date>_V2/staircase_<kind>_<HHMMSS>.json and every update (.npz).
    """
    import numpy as np
    from . import actuate, machine, pvs

    names = [*pvs.L1, *pvs.L2, *pvs.BLEN]
    moves, refused = [], None

    def move(delta):                    # the ledger entry, with when the move began and ended (this host's clock)
        start = time.time()
        entry = {"L1": actuate.move_l1, "L2": actuate.move_l2}[kind](delta)
        moves.append({**entry, "began": start, "ended": time.time()})

    with machine.updates(names) as updates:
        for delta in [step] * n + [None] + [-step] * n:
            time.sleep(hold_s)
            if delta is None:                                   # one more step up: the budget must refuse it
                try:
                    move(step)
                    move(-step)
                except ValueError as e:
                    refused = str(e)
                continue
            move(delta)
        time.sleep(hold_s)
    updates = list(updates)

    def medians(start, end):            # a PV that didn't update in the hold: the value it was holding
        out = {}
        for pv in names:
            mine = [u for u in updates if u.name == pv and np.isfinite(u.value)]
            values = [u.value for u in mine if start <= u.timestamp < end] or \
                [u.value for u in mine if u.timestamp < start][-1:]
            out[pv] = float(np.median(values)) if values else None
        return out

    summary = {"kind": kind, "step": step, "n": n, "hold_s": hold_s, "refused": refused,
               "moves": [{**m, "hold_before": medians(m["began"] - hold_s, m["began"]),
                          "hold_after": medians(m["ended"], m["ended"] + hold_s)} for m in moves]}
    path = save_record("V2", f"staircase_{kind}", summary)
    np.savez_compressed(path.with_suffix(".npz"), name=[u.name for u in updates],
                        pulse_id=[u.pulse_id for u in updates], timestamp=[u.timestamp for u in updates],
                        value=[u.value for u in updates])
    return summary
