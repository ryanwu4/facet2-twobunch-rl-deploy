"""Channel Access server for the offline tests: serves DAQ frames as the DTOTR2 camera PV, the camera readbacks,
the DAQ's TCAV amplitude and phase readbacks of each frame's shot, stand-ins for the L1 phase setpoints
(conftest.PREQ_STAND_IN), and the other L1 and L2 PVs (conftest.constant_values).

    python tests/soft_ioc.py <DAQ h5 file> <first> <last>

Each line on stdin is a mode of conftest.MODES ("roi", "full", "srv01" or "step"). The IOC sets the camera
readbacks to the mode's, publishes frame `first` with its own EPICS timestamp, prints "ready", waits 2 s for the
test to subscribe, then publishes frames first+1 .. last-1 at 10 Hz and keeps the last one. In "step" mode it
publishes every frame of the file instead, at the trained ROI.
Each DAQ frame (TEST_15668's, at the trained ROI) sits at its ROI in a sensor of dark pixels, and the IOC
serves the mode's ROI of that sensor: in "roi" mode, exactly the DAQ's raw arrays. Frames are served as the
flat array the camera publishes, as 16-bit integers (the live type, S1). Just before each frame, the IOC
publishes the TCAV readbacks and the PREQs of its shot (conftest.tcav_readbacks, preq_values) with the frame's
timestamp; a frame without a shot gets none. The other L1 and L2 PVs keep their value. The test sets the addresses and
ports through the environment, all on 127.0.0.1.
"""
import asyncio
import sys

import h5py
import numpy as np
from caproto import ChannelDouble, ChannelEnum, ChannelInteger, ChannelString, ChannelType
from caproto.server import PVGroup, pvproperty, run

from conftest import MODES, PREQ_STAND_IN, constant_values, preq_values, tcav_readbacks
from twobunch_deploy import pvs, screen

path, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with h5py.File(path) as f:
    step_frames = f["entry/data/data"][:].astype("int16")              # (frames, SizeY, SizeX)
    attrs = f["entry/instrument/NDAttributes"]
    step_stamps = list(zip(attrs["NDArrayEpicsTSSec"][:].tolist(), attrs["NDArrayEpicsTSnSec"][:].tolist()))
daq_frames, stamps = step_frames[first:last], step_stamps[first:last]
tcav, preq = tcav_readbacks(), preq_values()
phases = {**{name: ChannelDouble(value=0.0) for name in PREQ_STAND_IN},
          **{name: ChannelDouble(value=value) for name, value in constant_values().items()}}

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
    amplitude = pvproperty(name=pvs.TCAV_AMPLITUDE, dtype=float, value=0.0, read_only=True)
    phase = pvproperty(name=pvs.TCAV_PHASE, dtype=float, value=0.0, read_only=True)

    async def publish(self, frame, stamp):
        """The TCAV and L1 readbacks of the frame's shot, if it has one, then the frame, all with the frame's timestamp
        ((EPICS seconds, nanoseconds): exact)."""
        if (pulse_id := stamp[1] & 0x1ffff) in tcav:
            await self.amplitude.write(tcav[pulse_id][0], timestamp=stamp)
            await self.phase.write(tcav[pulse_id][1], timestamp=stamp)
            for name, value in preq[pulse_id].items():
                await phases[name].write(value, timestamp=stamp)
        await self.image.write(frame, timestamp=stamp)

    @image.startup
    async def image(self, instance, async_lib):
        while mode := (await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)).strip():
            camera = MODES[mode]
            for name, channel in settings.items():
                await channel.write(camera[name.rsplit(":", 1)[1]])
            data, times = ((step_frames.reshape(len(step_frames), -1), step_stamps) if mode == "step"
                           else (frames(camera), stamps))
            await self.publish(data[0], times[0])
            print("ready", flush=True)
            await async_lib.library.sleep(2)
            for frame, stamp in zip(data[1:], times[1:]):
                await self.publish(frame, stamp)
                await async_lib.library.sleep(0.1)


run({**Camera(prefix="").pvdb, **settings, **phases}, interfaces=["127.0.0.1"])
