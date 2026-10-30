"""S1: camera frames and pulse IDs.

The soft-IOC tests serve real DAQ frames, which are not in this public repo, so they run only where the
TEST_15668 DTOTR2 files are (DTOTR2_DAQ, default next to this repo) and are skipped in CI.
"""
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from twobunch_deploy import machine, session

REPO = Path(__file__).resolve().parents[1]
DAQ = Path(os.environ.get("DTOTR2_DAQ", REPO.parent / "XTCAV_analysis/TEST_15668/images/DTOTR2"))
# 30 frames of step 8; the pulse ID wraps after the 11th
FILE, FIRST, LAST = "DTOTR2_data_step08.h5", 20, 50
needs_daq = pytest.mark.skipif(not (DAQ / FILE).exists(), reason=f"needs the DAQ file {DAQ / FILE}")


def test_pulse_id_steps():
    # the wrap from step 8, then a skip and a repeat as in step 27
    assert session.pulse_id_steps([130978, 131014, 10, 46, 118, 118, 154]) == {0: 1, 36: 4, 72: 1}


@pytest.fixture(scope="module")
def replay():
    """One soft IOC for this module; replay() starts publishing the frames again.

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

        def start():
            ioc.stdin.write("replay\n")
            ioc.stdin.flush()
            assert ioc.stdout.readline().strip() == "ready"

        try:
            yield start
        finally:
            ioc.kill()
            ioc.wait()


def daq():
    with h5py.File(DAQ / FILE) as f:
        data = f["entry/data/data"][FIRST:LAST].reshape(LAST - FIRST, -1)
        frame_ids = f["entry/instrument/NDAttributes/NDArrayUniqueId"][FIRST:LAST].tolist()
    return data, frame_ids


@needs_daq
def test_frames(replay):
    data, frame_ids = daq()
    replay()
    got = machine.frames(6)
    # every frame published, in order, each with its own pulse ID (the camera's frame ID)
    assert [f.pulse_id for f in got] == frame_ids
    for f, expected in zip(got, data):
        assert np.array_equal(f.data, expected)
    assert all(f.received >= f.timestamp for f in got)


@needs_daq
def test_capture_test(replay, tmp_path, monkeypatch):
    monkeypatch.setattr(session, "DATA", tmp_path)
    data, frame_ids = daq()
    replay()
    summary = session.capture_test(seconds=6, n_saved=10)
    assert summary["frames"] == 30 and summary["capture_fraction"] == 1.0
    assert summary["pulse_id_steps"] == {36: 29}
    assert summary["lengths"] == [894 * 184] and summary["dtype"] == "int16"
    assert (summary["min"], summary["max"]) == (data.min(), data.max())

    [record] = tmp_path.glob("*_S1/capture_test_*.json")
    assert json.loads(record.read_text())["frames"] == 30
    saved = np.load(record.with_suffix(".npz"))
    assert saved["pulse_id"].tolist() == frame_ids
    for pulse_id, frame in zip(saved["saved_pulse_id"], saved["saved_data"]):
        assert np.array_equal(frame, data[frame_ids.index(pulse_id)])
    assert len(saved["saved_data"]) == 10
