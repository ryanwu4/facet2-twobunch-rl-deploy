"""Channel Access reads through pyepics. v1 is read-only: this module only ever reads.

S1  frames(): every DTOTR2 frame published during a capture, with its pulse ID
S2  get(): one read of a scalar PV in pvs.READ; camera_settings(): the DTOTR2 camera's ROI, orientation, exposure,
    gain, pixel size, and where its array comes from and its size
"""
from __future__ import annotations

import time
from typing import NamedTuple

import epics
import numpy as np

from . import pvs


class Frame(NamedTuple):
    pulse_id: int       # timestamp nanoseconds & 0x1ffff
    timestamp: float    # EPICS timestamp of the frame, POSIX seconds
    received: float     # when the callback got it, POSIX seconds on this host
    data: np.ndarray    # the raw flat array as published; S2 reshapes and orients it


def frames(seconds: float) -> list[Frame]:
    """Every DTOTR2 frame the camera publishes during the next `seconds`.

    Subscribes for this call only and disconnects at the end. Each frame comes from one monitor callback,
    so its data, timestamp and pulse ID belong to the same update. The first frame is the IOC's current
    one when the subscription starts: it is old if the camera has stopped (compare received and timestamp).
    """
    got = []

    def keep(value, nanoseconds, timestamp, **_):   # runs in pyepics' Channel Access thread: only append
        got.append(Frame(nanoseconds & 0x1ffff, timestamp, time.time(), value))

    # A frame is larger than pyepics monitors by default, so the monitor is requested explicitly. DBE_VALUE
    # only: alarm-state changes would resend the current frame.
    pv = epics.PV(pvs.CAMERA, auto_monitor=epics.dbr.DBE_VALUE, callback=keep)
    time.sleep(seconds)
    pv.disconnect()
    return got


def get(name: str, as_string: bool = False):
    """One read of a scalar PV, without subscribing. Raises ValueError if the name isn't in pvs.READ, so
    only reviewed PVs are read, and TimeoutError if the PV doesn't answer within 2 s.

    as_string gives an enum's label (e.g. "Positive") instead of its index.
    """
    if name not in pvs.READ:
        raise ValueError(f"{name} is not in pvs.READ")
    pv = epics.PV(name, auto_monitor=False)
    value = pv.get(timeout=2, as_string=as_string)
    pv.disconnect()
    if value is None:
        raise TimeoutError(f"no answer from {name}")
    return value


def camera_settings() -> dict:
    """The DTOTR2 camera readbacks in pvs.CAMERA_SETTINGS, keyed by PV suffix (e.g. "MinX_RBV").

    The orientations and reverses are enums, read as their labels ("Positive", "No").
    """
    enums = ("_ORIENT", "ReverseX_RBV", "ReverseY_RBV")
    return {name.rsplit(":", 1)[1]: get(name, as_string=name.endswith(enums)) for name in pvs.CAMERA_SETTINGS}
