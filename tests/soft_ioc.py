"""Channel Access server for the offline tests: serves DAQ frames as the DTOTR2 camera PV, and the camera
readbacks.

    python tests/soft_ioc.py <DAQ h5 file> <first> <last>

Each line on stdin is a mode of conftest.MODES ("roi", "full" or "srv01"). The IOC sets the camera readbacks
to the mode's, publishes frame `first` with its own EPICS timestamp, prints "ready", waits 2 s for the test
to subscribe, then publishes frames first+1 .. last-1 at 10 Hz and keeps the last one.
Each DAQ frame (TEST_15668's, at the trained ROI) sits at its ROI in a sensor of dark pixels, and the IOC
serves the mode's ROI of that sensor: in "roi" mode, exactly the DAQ's raw arrays. Frames are served as the
flat array the camera publishes, as 16-bit integers (the live type, S1). The test sets the addresses and
ports through the environment, all on 127.0.0.1.
"""
import asyncio
import sys

import h5py
import numpy as np
from caproto import ChannelDouble, ChannelEnum, ChannelInteger, ChannelString, ChannelType
from caproto.server import PVGroup, pvproperty, run

from conftest import MODES
from twobunch_deploy import pvs, screen

path, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with h5py.File(path) as f:
    daq_frames = f["entry/data/data"][first:last].astype("int16")      # (frames, SizeY, SizeX)
    attrs = f["entry/instrument/NDAttributes"]
    stamps = list(zip(attrs["NDArrayEpicsTSSec"][first:last].tolist(),
                      attrs["NDArrayEpicsTSnSec"][first:last].tolist()))

T = screen.TRAINED
sensor = np.full((last - first, 964, 1292), 31, "int16")                  # 31: the live dark level (S1)
sensor[:, T["MinY_RBV"]:T["MinY_RBV"] + T["SizeY_RBV"], T["MinX_RBV"]:T["MinX_RBV"] + T["SizeX_RBV"]] = daq_frames


def frames(c):
    """The camera ROI `c` of each sensor frame, flat as the camera publishes it."""
    roi = sensor[:, c["MinY_RBV"]:c["MinY_RBV"] + c["SizeY_RBV"], c["MinX_RBV"]:c["MinX_RBV"] + c["SizeX_RBV"]]
    return roi.reshape(len(roi), -1)


def readback(key):
    value = MODES["roi"][key]
    if key.endswith("_ORIENT"):
        return ChannelEnum(value=value, enum_strings=["Positive", "Negative"])
    if key.startswith("Reverse"):
        return ChannelEnum(value=value, enum_strings=["No", "Yes"])
    if isinstance(value, str):
        return ChannelString(value=value)
    if isinstance(value, int):
        return ChannelInteger(value=value)
    return ChannelDouble(value=value)


settings = {name: readback(name.rsplit(":", 1)[1]) for name in pvs.CAMERA_SETTINGS}


class Camera(PVGroup):
    image = pvproperty(name=pvs.CAMERA, dtype=ChannelType.INT, value=frames(MODES["roi"])[0],
                       max_length=1292 * 964, read_only=True)

    @image.startup
    async def image(self, instance, async_lib):
        while mode := (await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)).strip():
            for name, channel in settings.items():
                await channel.write(MODES[mode][name.rsplit(":", 1)[1]])
            data = frames(MODES[mode])
            await instance.write(data[0], timestamp=stamps[0])     # (EPICS seconds, nanoseconds): exact
            print("ready", flush=True)
            await async_lib.library.sleep(2)
            for frame, stamp in zip(data[1:], stamps[1:]):
                await instance.write(frame, timestamp=stamp)
                await async_lib.library.sleep(0.1)


run({**Camera(prefix="").pvdb, **settings}, interfaces=["127.0.0.1"])
