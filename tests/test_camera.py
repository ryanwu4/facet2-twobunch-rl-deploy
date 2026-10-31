"""S1: camera frames and pulse IDs. The soft IOC and the DAQ frames are in conftest.py."""
import json

import numpy as np

from conftest import daq, needs_daq
from twobunch_deploy import machine, session


def test_pulse_id_steps():
    # the wrap from TEST_15668 step 8, a frame stamped with the next frame's ID, a frame one pulse early
    assert session.pulse_id_steps([130978, 131014, 10, 46, 118, 118, 154, 189, 226]) == \
        {0: 1, 35: 1, 36: 4, 37: 1, 72: 1}


@needs_daq
def test_frames(replay):
    data, frame_ids = daq()
    data = data.reshape(len(data), -1)
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
    data = data.reshape(len(data), -1)
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
