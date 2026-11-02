"""Calls the notebooks make. S0: where and with what a session runs, and saving records under data/.
S1: capture_test() of the camera frames and pulse IDs. S2: geometry_test() of the camera settings and the
hand crop. S3: backgrounds (take_background, pedestal, load_background), peek() and processing_test()."""
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
    """S3: the newest frame through the processing, drawn with its energy cut and band statistics (viz.peek).

    Captures for `seconds` and takes the last frame. The cut is the frame's own valley: S4 adds the TCAV
    polarity and S5 the step's cut. Saves the summary to data/<date>_S3/peek_<HHMMSS>.json and, next to it, the
    raw frame and its hand crop (.npz) and the figure (.png).
    """
    import numpy as np
    from . import screen, viz

    camera, frames, crops = hand_crops(seconds)
    frame, crop = frames[-1], crops[-1]
    start = time.perf_counter()
    processed = screen.process(crop, background.image)
    valley = screen.valley(processed.sum(axis=1))
    x = screen.x_positions(camera["RESOLUTION"])
    stats = None if valley.cut is None else screen.band_stats(processed, x, valley.cut)
    processing_s = time.perf_counter() - start
    summary = {"pulse_id": frame.pulse_id, "age_s": frame.received - frame.timestamp, "frames": len(frames),
               "background": background.path, "camera": camera, "processing_s": processing_s,
               "intensity": float(processed.sum()), "band_stats": stats}
    path = save_record("S3", "peek", summary)
    np.savez_compressed(path.with_suffix(".npz"), raw=frame.data, crop=crop)
    title = (f"DTOTR2, pulse ID {frame.pulse_id}, {summary['age_s']:.2f} s old; "
             + ("no valley: no cut" if stats is None else
                f"x_d - x_w {1e3 * (stats['x_d'] - stats['x_w']):.3f} mm, drive fraction {stats['frac_d']:.3f}, "
                f"edge fraction {stats['edge_frac']:.4f}"))
    fig = viz.peek(crop, processed, valley, stats, x, title)
    fig.savefig(path.with_suffix(".png"))
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
