"""Channel Access server for the offline tests: serves DAQ frames as the DTOTR2 camera PV.

    python tests/soft_ioc.py <DAQ h5 file> <first> <last>

For each line on stdin: publishes frame `first` with its own EPICS timestamp, prints "ready", waits 2 s for
the test to subscribe, then publishes frames first+1 .. last-1 at 10 Hz and keeps the last one. Frames are
served as the raw flat array the camera publishes, as 16-bit integers (the live CA type is recorded by the
S1 machine run). The test sets the addresses and ports through the environment, all on 127.0.0.1.
"""
import asyncio
import sys

import h5py
from caproto import ChannelType
from caproto.server import PVGroup, pvproperty, run

from twobunch_deploy import pvs

path, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with h5py.File(path) as f:
    frames = f["entry/data/data"][first:last].reshape(last - first, -1).astype("int16")
    attrs = f["entry/instrument/NDAttributes"]
    stamps = list(zip(attrs["NDArrayEpicsTSSec"][first:last].tolist(),
                      attrs["NDArrayEpicsTSnSec"][first:last].tolist()))


class Camera(PVGroup):
    # max_length: room for the whole 1292 x 964 sensor; frames are the 184 x 894 ROI, as on TEST_15668
    image = pvproperty(name=pvs.CAMERA, dtype=ChannelType.INT, value=frames[0], max_length=1292 * 964,
                       read_only=True)

    @image.startup
    async def image(self, instance, async_lib):
        while await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline):
            await instance.write(frames[0], timestamp=stamps[0])     # (EPICS seconds, nanoseconds): exact
            print("ready", flush=True)
            await async_lib.library.sleep(2)
            for frame, stamp in zip(frames[1:], stamps[1:]):
                await instance.write(frame, timestamp=stamp)
                await async_lib.library.sleep(0.1)


run(Camera(prefix="").pvdb, interfaces=["127.0.0.1"])
