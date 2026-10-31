"""The soft IOC the Channel Access tests share (S1, S2, S3), and the code and data the comparisons use.

The soft IOC serves real DAQ frames, which are not in this public repo, so these tests run only where the
TEST_15668 DTOTR2 files are (DTOTR2_DAQ, default next to this repo) and are skipped in CI. So are the comparisons
with Yiheng Ye's DAQ classes and with twobunch-s2e-rl's online/features.py, which need those next to this repo.
"""
import os
import socket
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image

from twobunch_deploy import screen

REPO = Path(__file__).resolve().parents[1]
DAQ = Path(os.environ.get("DTOTR2_DAQ", REPO.parent / "XTCAV_analysis/TEST_15668/images/DTOTR2"))
# 30 frames of step 8; the pulse ID wraps after the 11th
FILE, FIRST, LAST = "DTOTR2_data_step08.h5", 20, 50
needs_daq = pytest.mark.skipif(not (DAQ / FILE).exists(), reason=f"needs the DAQ file {DAQ / FILE}")
XTCAV = REPO.parent / "XTCAV_analysis"
YIHENG = XTCAV / "yiheng/DAQ_analysis_class"
needs_yiheng = pytest.mark.skipif(not (YIHENG / "image.py").exists(), reason=f"needs Yiheng's code in {YIHENG}")
FEATURES = REPO.parent / "twobunch-s2e-rl/src/twobunch_s2e_rl/online/features.py"
needs_features = pytest.mark.skipif(not FEATURES.exists(), reason=f"needs {FEATURES}")


def camera(min_x, min_y, size_x, size_y, **changes):
    """The camera readbacks (machine.camera_settings()) at an ROI: the training values, the driver's own
    unbinned output, and any `changes`."""
    return {**screen.TRAINED, "MinX_RBV": min_x, "MinY_RBV": min_y, "SizeX_RBV": size_x, "SizeY_RBV": size_y,
            "ReverseX_RBV": "No", "ReverseY_RBV": "No", "ArraySizeX_RBV": size_x, "ArraySizeY_RBV": size_y,
            "NDArrayPort_RBV": "PS107", "ArraySize0_RBV": size_x, "ArraySize1_RBV": size_y, **changes}


# the soft IOC's modes: the camera at the trained ROI (as in the DAQ runs), at the full sensor (as for S1 on
# 2026-10-30) and at the ROI read on srv01 on 2026-10-30 15:26, which misses the trained ROI's bottom rows
MODES = {"roi": camera(396, 70, 184, 894), "full": camera(0, 0, 1292, 964), "srv01": camera(346, 38, 324, 860)}


def pil_crop(frame):
    """A DAQ frame at the trained ROI, oriented (TEST_15668's: a transpose) and hand-cropped by PIL, as the
    DAQ classes do."""
    return np.array(Image.fromarray(np.ascontiguousarray(frame.T)).crop(screen.HAND_CROP))


def daq():
    """The frames the soft IOC serves, as (SizeY, SizeX) arrays, and their camera frame IDs."""
    with h5py.File(DAQ / FILE) as f:
        data = f["entry/data/data"][FIRST:LAST]
        frame_ids = f["entry/instrument/NDAttributes/NDArrayUniqueId"][FIRST:LAST].tolist()
    return data, frame_ids


@pytest.fixture(scope="session")
def replay():
    """One soft IOC for the whole test run; replay(mode) sets the camera readbacks to MODES[mode] and
    starts publishing the frames again (soft_ioc.py).

    The soft IOC, and Channel Access for this process, use 127.0.0.1 and a free port only: the tests can't
    reach a real IOC, and no other host can see the soft IOC. pyepics reads these settings once, when it
    starts Channel Access, so all tests in a process share one IOC.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        port = str(s.getsockname()[1])
    env = {"EPICS_CA_ADDR_LIST": "127.0.0.1", "EPICS_CA_AUTO_ADDR_LIST": "NO", "EPICS_CA_SERVER_PORT": port,
           "EPICS_CAS_INTF_ADDR_LIST": "127.0.0.1", "EPICS_CAS_SERVER_PORT": port,
           "EPICS_CAS_BEACON_ADDR_LIST": "127.0.0.1", "EPICS_CAS_AUTO_BEACON_ADDR_LIST": "NO"}
    with pytest.MonkeyPatch.context() as mp:
        for k, v in env.items():
            mp.setenv(k, v)
        ioc = subprocess.Popen([sys.executable, REPO / "tests/soft_ioc.py", DAQ / FILE, str(FIRST), str(LAST)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                               env={**os.environ, "PYTHONPATH": str(REPO)})

        def start(mode="roi"):
            ioc.stdin.write(mode + "\n")
            ioc.stdin.flush()
            assert ioc.stdout.readline().strip() == "ready"

        try:
            yield start
        finally:
            ioc.kill()
            ioc.wait()
