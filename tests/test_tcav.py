"""S4: TCAV readbacks matched to frames by pulse ID, the off / + / - tags and the frames dropped after each flip.

The synthetic tests run everywhere. The comparison with Yiheng Ye's classify_tcav_shots on the DAQ runs, and the
session test on the soft IOC, run only where those are (conftest.py).
"""
import json
import sys
from collections import Counter

import numpy as np
import pytest

from conftest import XTCAV, YIHENG, daq, needs_daq, needs_yiheng, tcav_readbacks
from twobunch_deploy import acquire, machine, pvs, session


def test_tag():
    # the mean of the amplitudes above 10 MV is 22.0
    amplitude = np.array([22.0, 22.0, 22.2, 21.8, 22.3, 21.7, 0.99, 1.0, np.nan, 22.0])
    phase = np.array([90.0, -90.9, 90.99, -89.0, 90.0, -90.0, 90.0, 0.0, 90.0, np.nan])
    assert acquire.tag(amplitude, phase).tolist() == [
        "+", "-", "+",
        "neither",                  # the phase 1.0 deg from -90
        "neither", "neither",       # the amplitude 0.3 MV from the mean
        "off",                      # below 1 MV, whatever the phase
        "neither",                  # 1 MV is not off
        "unmatched", "unmatched"]


def test_tag_without_the_tcav_on():
    """No amplitude above 10 MV: no frame is + or -, whatever its phase."""
    assert acquire.tag(np.array([0.01, 0.02, 5.0]), np.array([90.0, -90.0, 90.0])).tolist() == \
        ["off", "off", "neither"]


def test_dropped():
    # frames every 36 pulse IDs (10 Hz) across the wrap; a 0.25 s drop is 90 pulse IDs
    pulse_ids = [(131040 - 72 + 36 * i) % 131040 for i in range(16)]
    tags = ["+", "+", "+", "+", "neither", "+", "off", "+", "+", "+", "+", "-", "unmatched", "-", "-", "-"]
    expected = [True, True, True, False,    # the capture's first tagged frame begins a window
                False, False,               # a phase excursion doesn't begin a new one
                False,                      # off frames are never dropped
                True, True, True, False,    # + after off begins a new window
                True, False, True, False, False]    # unmatched frames are skipped
    assert acquire.dropped(pulse_ids, np.array(tags, dtype=object), 0.25).tolist() == expected


def test_readback():
    frames = [machine.Frame(p, 100.0 + i, 0.0, None) for i, p in enumerate([10, 46, 82, 118])]
    a, p = pvs.TCAV_AMPLITUDE, pvs.TCAV_PHASE
    updates = [machine.Update(a, 10, 100.0, 21.0),
               machine.Update(a, 46, 101.0, 22.0), machine.Update(a, 46, 101.0, 22.5),    # the later counts
               machine.Update(a, 82, 102.0 - 400, 23.0),     # the same pulse ID, 400 s earlier: another pulse
               machine.Update(p, 118, 103.0, 90.0)]          # another PV
    np.testing.assert_array_equal(acquire.readback(frames, updates, a), [21.0, 22.5, np.nan, np.nan])
    np.testing.assert_array_equal(acquire.readback(frames, updates, p), [np.nan, np.nan, np.nan, 90.0])


@needs_yiheng
@pytest.mark.parametrize("run", ["TEST_15668", "E300_15671", "E300_15673"])
def test_tags_equal_classify_tcav_shots(run):
    """On each DAQ run's readbacks: + and - are classify_tcav_shots', frame for frame. Off is a part of its off,
    and the rest of its off has the TCAV on (above 10 MV), in the middle of a flip."""
    if not (XTCAV / run / f"{run}.mat").exists():
        pytest.skip(f"needs the DAQ run {XTCAV / run}")
    sys.path[:0] = [str(YIHENG), str(YIHENG.parent / "XTCAV_analysis_script")]
    import dataset
    import xtcav
    classified = xtcav.classify_tcav_shots(dataset.DATASET(str(XTCAV), *run.split("_")))
    amplitude = np.asarray(classified["amp"], float)
    tags = acquire.tag(amplitude, np.asarray(classified["phase"], float))
    assert np.array_equal(np.flatnonzero(tags == "+"), classified["ix_plus"])
    assert np.array_equal(np.flatnonzero(tags == "-"), classified["ix_neg"])
    off = np.flatnonzero(tags == "off")
    assert set(off) <= set(classified["ix_off"]) and len(off) > 0
    assert (amplitude[np.setdiff1d(classified["ix_off"], off)] > acquire.ON_AMPLITUDE).all()


@needs_daq
def test_tcav_test(replay, tmp_path, monkeypatch):
    monkeypatch.setattr(session, "DATA", tmp_path)
    _, frame_ids = daq()
    readbacks = tcav_readbacks()
    amplitude = np.array([readbacks[p][0] for p in frame_ids])      # every frame served has a shot
    phase = np.array([readbacks[p][1] for p in frame_ids])
    tags = acquire.tag(amplitude, phase)
    replay("roi")
    summary = session.tcav_test(seconds=6)
    assert summary["frames"] == 30 and summary["matched_fraction"] == 1
    for name in (pvs.TCAV_AMPLITUDE, pvs.TCAV_PHASE):
        assert summary["updates"][name]["count"] == 30           # one per frame, the first on subscribing
    assert summary["tags"] == dict(Counter(tags))
    assert summary["dropped"] == acquire.dropped(frame_ids, tags, 1.0).sum()
    for t, ranges in summary["range"].items():
        assert ranges["amplitude"] == [amplitude[tags == t].min(), np.median(amplitude[tags == t]),
                                       amplitude[tags == t].max()]
    [record] = tmp_path.glob("*_S4/tcav_test_*.json")
    assert json.loads(record.read_text()) == json.loads(json.dumps(summary))     # JSON keys are strings
    saved = np.load(record.with_suffix(".npz"))
    assert saved["pulse_id"].tolist() == frame_ids and saved["tag"].tolist() == tags.tolist()
    assert np.array_equal(saved["amplitude"], amplitude) and np.array_equal(saved["phase"], phase)
    assert saved["update_value"].size == 60
