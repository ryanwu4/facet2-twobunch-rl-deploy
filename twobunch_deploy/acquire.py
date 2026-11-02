"""Frames for one measurement step (S4): every DTOTR2 frame of a capture with the TCAV amplitude and phase readbacks of
its pulse, each frame tagged off / + / -, and the + and - frames right after each polarity flip marked to drop.

The + and - rule is Yiheng Ye's classify_tcav_shots (xtcav.py, after D. Storey's plot_TCAV.m), which the deployed
analysis used. Off is an amplitude below OFF_AMPLITUDE: classify_tcav_shots' off (0.5 MV below the mean on amplitude)
also takes frames in the middle of a flip, with the TCAV on (86% of its off frames in E300_15671 and E300_15673).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from . import machine, pvs
from .session import PULSE_ID_PERIOD, PULSES_PER_SECOND

ON_AMPLITUDE = 10.0     # MV: the frames above it give the mean on amplitude
OFF_AMPLITUDE = 1.0     # MV: below it the TCAV is off
AMPLITUDE_TOL = 0.25    # MV: + and - frames are within this of the mean on amplitude
PHASE_TOL = 1.0         # deg: + frames are within this of +90, - frames of -90


class Capture(NamedTuple):
    camera: dict            # the camera readbacks, read just before the frames (machine.camera_settings)
    frames: list            # every frame published during the capture (machine.Frame)
    updates: list           # every update of the TCAV readbacks during the capture (machine.Update)
    amplitude: np.ndarray   # each frame's TCAV amplitude (MV); nan if no update has its pulse (readback)
    phase: np.ndarray       # each frame's TCAV phase (deg); likewise
    tags: np.ndarray        # each frame's tag (tag)
    drop: np.ndarray        # True for the + and - frames less than drop_s after their polarity window began (dropped)


def capture(seconds: float, drop_s: float = 1.0) -> Capture:
    """The camera readbacks, then every frame published during the next `seconds` with the TCAV readbacks of its
    pulse; each frame tagged, and the frames to drop after each flip marked."""
    camera = machine.camera_settings()
    with machine.updates([pvs.TCAV_AMPLITUDE, pvs.TCAV_PHASE]) as updates:
        frames = machine.frames(seconds)
    updates = list(updates)
    amplitude, phase = (readback(frames, updates, name) for name in (pvs.TCAV_AMPLITUDE, pvs.TCAV_PHASE))
    tags = tag(amplitude, phase)
    return Capture(camera, frames, updates, amplitude, phase, tags,
                   dropped([f.pulse_id for f in frames], tags, drop_s))


def readback(frames: list, updates: list, name: str) -> np.ndarray:
    """Each frame's value of the PV `name`: the update with the frame's pulse ID, if its timestamp is within 1 s of
    the frame's (pulse IDs repeat every 6 min, and a monitor's first update can be old); nan if there is none. Of
    two updates with the same pulse ID, the later counts."""
    by_pulse = {u.pulse_id: u for u in updates if u.name == name}
    values = []
    for f in frames:
        u = by_pulse.get(f.pulse_id)
        values.append(u.value if u is not None and abs(u.timestamp - f.timestamp) < 1 else np.nan)
    return np.array(values, float)


def tag(amplitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Each frame's TCAV tag:
    - "off": the amplitude is below OFF_AMPLITUDE;
    - "+" and "-": the amplitude is within AMPLITUDE_TOL of the mean of the amplitudes above ON_AMPLITUDE, and the
      phase within PHASE_TOL of +90 and -90 (classify_tcav_shots; without an amplitude above ON_AMPLITUDE, none);
    - "neither": any other readback, e.g. during a flip;
    - "unmatched": the frame has no readback.
    """
    on = amplitude > ON_AMPLITUDE
    mean_on = amplitude[on].mean() if on.any() else np.nan
    near = np.abs(amplitude - mean_on) < AMPLITUDE_TOL
    tags = np.full(len(amplitude), "neither", dtype=object)
    tags[near & (np.abs(phase - 90) < PHASE_TOL)] = "+"
    tags[near & (np.abs(phase + 90) < PHASE_TOL)] = "-"
    tags[amplitude < OFF_AMPLITUDE] = "off"
    tags[np.isnan(amplitude) | np.isnan(phase)] = "unmatched"
    return tags


def dropped(pulse_ids: list[int], tags: np.ndarray, drop_s: float) -> np.ndarray:
    """True for each + and - frame less than `drop_s` after its polarity window began, counted in pulse IDs (360 a
    second, across the wrap): the screen keeps settling for about 1 s after the phase readback locks (10-19 analysis
    of 177 DAQ flips).

    A window begins at a + or - frame whose most recent earlier frame tagged +, - or off has another tag (frames
    tagged neither or unmatched are skipped, so a short phase excursion doesn't restart it), or at the capture's first
    such frame, as the window may have begun just before the capture. Off frames are never dropped.
    """
    drop = np.zeros(len(tags), bool)
    last = start = None
    for i, (pulse_id, t) in enumerate(zip(pulse_ids, tags)):
        if t not in ("+", "-", "off"):
            continue
        if t != last:
            last, start = t, pulse_id
        drop[i] = t != "off" and (pulse_id - start) % PULSE_ID_PERIOD < drop_s * PULSES_PER_SECOND
    return drop
