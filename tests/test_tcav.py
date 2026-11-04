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
    # the amplitudes above 10 MV average 22.0; the + and - frames' 22.067, so off is below 21.567
    amplitude = np.array([22.0, 22.0, 22.2, 21.8, 22.3, 21.7, 21.5, 22.5, 21.6, 22.4, 22.0, np.nan, 0.01])
    phase = np.array([90.0, -90.9, 90.99, -89.0, 90.0, -90.0, 90.0, 90.0, 90.0, 90.0, np.nan, 90.0, 90.0])
    assert acquire.tag(amplitude, phase).tolist() == [
        "+", "-", "+",
        "neither",                  # the phase 1.0 deg from -90
        "neither", "neither",       # the amplitude 0.3 MV from the first mean
        "off",                      # 0.567 MV below the + and - frames' mean, whatever the phase
        "neither",
        "neither",                  # 0.467 MV below: not off
        "neither",
        "unmatched", "unmatched",
        "off"]


def test_tag_without_plus_or_minus():
    """No + or - frame: off is 0.5 MV below the mean of the amplitudes above 10 MV; none above 10 MV (the TCAV off
    throughout): below 9.5 MV, and no frame is + or -, whatever its phase."""
    assert acquire.tag(np.array([22.0, 22.0, 21.0, 0.01]), np.array([0.0, 45.0, 0.0, 90.0])).tolist() == \
        ["neither", "neither", "off", "off"]
    assert acquire.tag(np.array([0.01, 0.02, 9.4, 9.6]), np.array([90.0, -90.0, 90.0, -90.0])).tolist() == \
        ["off", "off", "off", "neither"]


def test_dropped():
    # frames every 36 pulse IDs (10 Hz) across the wrap; a 0.25 s drop is 90 pulse IDs
    tags = (["+", "+", "+", "+", "neither", "+", "off", "+", "+", "+", "+", "-", "unmatched", "-", "-", "-"]
            + ["neither"] * 5 + ["-"] * 4 + ["neither"] * 4 + ["-"]
            + ["neither"] * 3 + ["unmatched"] * 2 + ["neither"] * 2 + ["-"])
    pulse_ids = [(131040 - 72 + 36 * i) % 131040 for i in range(len(tags))]
    expected = ([True, True, True, False,     # the capture's first tagged frame begins a window
                 False, False,                # one frame of neither doesn't begin a new one
                 False,                       # off frames are never dropped
                 True, True, True, False,     # + after off begins a new window
                 True, False, True, False, False]    # unmatched frames are skipped
                + [False] * 5 + [True, True, True, False]    # 5 frames of neither end the window
                + [False] * 4 + [False]                      # 4 don't
                + [False] * 7 + [True])                      # unmatched frames don't break a run of neither
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
    """On each DAQ run's readbacks: off, + and - are classify_tcav_shots', frame for frame."""
    if not (XTCAV / run / f"{run}.mat").exists():
        pytest.skip(f"needs the DAQ run {XTCAV / run}")
    sys.path[:0] = [str(YIHENG), str(YIHENG.parent / "XTCAV_analysis_script")]
    import dataset
    import xtcav
    classified = xtcav.classify_tcav_shots(dataset.DATASET(str(XTCAV), *run.split("_")))
    tags = acquire.tag(np.asarray(classified["amp"], float), np.asarray(classified["phase"], float))
    for t, key in (("+", "ix_plus"), ("-", "ix_neg"), ("off", "ix_off")):
        assert np.array_equal(np.flatnonzero(tags == t), classified[key])
    assert len(classified["ix_off"]) > 300


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
    assert np.isin(saved["update_name"], [pvs.TCAV_AMPLITUDE, pvs.TCAV_PHASE]).sum() == 60     # and the L1, L2 PVs'
