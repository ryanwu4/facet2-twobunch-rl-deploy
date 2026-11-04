"""Frames for one measurement step (S4): every DTOTR2 frame of a capture with the TCAV amplitude and phase readbacks of
its pulse, each frame tagged off / + / -, and the + and - frames right after each polarity window began marked to
drop. S6, S7: every update of the L1 and L2 PVs during the same capture, and a summary of each.

The tags are Yiheng Ye's classify_tcav_shots (xtcav.py, after D. Storey's plot_TCAV.m), which the deployed analysis
used. The readbacks don't show the TCAV's off windows as off. Ryan Loney's XTCAV toggler, whose pattern the DAQ runs
match, turns the TCAV off for the beam by putting its klystron on standby and triggering the amplitude and phase
detector on the standby pulse. So in an off window the readbacks stay near the on amplitude, with the phase slewing to
the next polarity, while the beam is unstreaked: in the DAQ runs, about 75% of the frames tagged neither, and 90-99%
of those tagged off, are unstreaked. The off tag catches the amplitude dip at the trigger switch, a few frames per
off window.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from . import machine, pvs
from .session import PULSE_ID_PERIOD, PULSES_PER_SECOND

ON_AMPLITUDE = 10.0     # MV: the frames above it give a first mean on amplitude
AMPLITUDE_TOL = 0.25    # MV: + and - frames are within this of that first mean
PHASE_TOL = 1.0         # deg: + frames are within this of +90, - frames of -90
OFF_DIP = 0.5           # MV: off frames are more than this below the mean amplitude of the + and - frames
NEITHER_RUN = 5         # frames: a run of at least this many frames tagged neither ends a polarity window (dropped)


class Capture(NamedTuple):
    camera: dict            # the camera readbacks, read just before the frames (machine.camera_settings)
    frames: list            # every frame published during the capture (machine.Frame)
    updates: list           # every update of the TCAV readbacks and the L1 and L2 PVs during the capture (machine.Update)
    amplitude: np.ndarray   # each frame's TCAV amplitude (MV); nan if no update has its pulse (readback)
    phase: np.ndarray       # each frame's TCAV phase (deg); likewise
    tags: np.ndarray        # each frame's tag (tag)
    drop: np.ndarray        # True for the + and - frames less than drop_s after their polarity window began (dropped)


def capture(seconds: float, drop_s: float = 1.0) -> Capture:
    """The camera readbacks, then every frame published during the next `seconds` with the TCAV readbacks of its
    pulse; each frame tagged, and the frames to drop after each flip marked. The L1 and L2 PVs (pvs.L1, pvs.L2) are
    monitored alongside (summarise)."""
    camera = machine.camera_settings()
    with machine.updates([pvs.TCAV_AMPLITUDE, pvs.TCAV_PHASE, *pvs.L1, *pvs.L2]) as updates:
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


def summarise(updates: list, name: str) -> dict:
    """The updates of the PV `name` during a capture (S6, S7): how many, their median, the first (the PV's value when
    the capture began: a monitor's first update is the current value) and the last, and how many read exactly 0. In
    TEST_15668, FAST_PACT reads 0 on 0.6-0.8% of the shots (BSA dropouts), which barely move the median."""
    values = [u.value for u in updates if u.name == name]
    if not values:
        return {"updates": 0}
    return {"updates": len(values), "median": float(np.median(values)), "first": float(values[0]),
            "last": float(values[-1]), "zeros": values.count(0)}


def tag(amplitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Each frame's TCAV tag, by classify_tcav_shots' rule:
    - "+" and "-": the amplitude within AMPLITUDE_TOL of the mean of the amplitudes above ON_AMPLITUDE, and the
      phase within PHASE_TOL of +90 and -90;
    - "off": the amplitude more than OFF_DIP below the mean amplitude of the + and - frames (without any, of the
      amplitudes above ON_AMPLITUDE; without those either, the TCAV is off throughout and the mean is ON_AMPLITUDE);
    - "neither": any other readback, e.g. in an off window, or while the phase converges after one;
    - "unmatched": the frame has no readback.
    Without an amplitude above ON_AMPLITUDE no frame is + or - (classify_tcav_shots then takes the mean of all the
    amplitudes, and so can tag frames of a TCAV that is off as + or -).
    """
    on = amplitude > ON_AMPLITUDE
    first = amplitude[on].mean() if on.any() else np.nan
    near = np.abs(amplitude - first) < AMPLITUDE_TOL
    plus, minus = near & (np.abs(phase - 90) < PHASE_TOL), near & (np.abs(phase + 90) < PHASE_TOL)
    mean_on = amplitude[plus | minus].mean() if (plus | minus).any() else first if on.any() else ON_AMPLITUDE
    tags = np.full(len(amplitude), "neither", dtype=object)
    tags[plus], tags[minus] = "+", "-"
    tags[amplitude < mean_on - OFF_DIP] = "off"
    tags[np.isnan(amplitude) | np.isnan(phase)] = "unmatched"
    return tags


def dropped(pulse_ids: list[int], tags: np.ndarray, drop_s: float) -> np.ndarray:
    """True for each + and - frame less than `drop_s` after its polarity window began, counted in pulse IDs (360 a
    second, across the wrap): the screen keeps settling for about 1 s after the phase readback locks (10-19 analysis
    of 177 DAQ flips).

    A window begins at a + or - frame
    - whose most recent earlier frame tagged +, - or off has another tag;
    - or that follows a run of at least NEITHER_RUN frames tagged neither (unmatched frames skipped): an off window,
      or the phase converging after one, in which a stray frame can read within 1 deg of +-90;
    - or that is the capture's first tagged frame, as the window may have begun just before the capture.
    A shorter run of neither (a phase excursion) doesn't begin one. Off frames are never dropped.
    """
    drop = np.zeros(len(tags), bool)
    last = start = None
    neither = 0                                     # frames tagged neither since the last +, - or off
    for i, (pulse_id, t) in enumerate(zip(pulse_ids, tags)):
        if t == "neither":
            neither += 1
        if t not in ("+", "-", "off"):
            continue
        if t != last or neither >= NEITHER_RUN:
            last, start = t, pulse_id
        neither = 0
        drop[i] = t != "off" and (pulse_id - start) % PULSE_ID_PERIOD < drop_s * PULSES_PER_SECOND
    return drop
