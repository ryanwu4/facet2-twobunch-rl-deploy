"""Channel Access reads through pyepics. v1 is read-only: this module only ever reads.

S1  frames(): every DTOTR2 frame published during a capture, with its pulse ID
Planned:
  S2  frame geometry: reshape and orient like the DAQ images
  S4  get(name): one scalar PV; refuses names not in pvs.READ
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
