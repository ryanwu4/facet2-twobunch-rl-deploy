"""S6, S7: the L1 and L2 PVs, monitored during each step's capture and by phases_test().

The synthetic tests run everywhere; the session test on the soft IOC only where its DAQ data are (conftest.py).
step()'s record of them is tested in test_step.py.
"""
import json

import numpy as np

from conftest import PREQ, constant_values, daq, l1_readbacks, needs_daq
from twobunch_deploy import acquire, machine, pvs, session


def test_pvs():
    assert set(pvs.L1_SETPOINTS) | set(pvs.L1_SETPOINTS.values()) <= set(pvs.L1)
    assert all(name in pvs.READ for name in [*pvs.L1, *pvs.L2])
    for readback, setpoint in pvs.L1_SETPOINTS.items():          # each FAST_PACT with its own klystron's PREQ
        assert readback.endswith(":FAST_PACT") and setpoint.endswith(":PREQ")
        assert readback.split(":")[:3] == setpoint.split(":")[:3]


def test_summarise():
    a, b = pvs.L1_SETPOINTS
    updates = [machine.Update(a, 0, 1.0, -23.0), machine.Update(b, 0, 1.0, -20.0), machine.Update(a, 36, 1.1, 0.0),
               machine.Update(a, 72, 1.2, -23.5), machine.Update(a, 108, 1.3, -23.25)]
    assert acquire.summarise(updates, a) == {"updates": 4, "median": -23.125, "first": -23.0, "last": -23.25,
                                             "zeros": 1}
    assert acquire.summarise(updates, b) == {"updates": 1, "median": -20.0, "first": -20.0, "last": -20.0, "zeros": 0}
    assert acquire.summarise(updates, pvs.TCAV_PHASE) == {"updates": 0}


@needs_daq
def test_phases_test(replay, tmp_path, monkeypatch):
    """phases_test() on the soft IOC: the L1 readbacks of each frame's shot, the other PVs constant."""
    monkeypatch.setattr(session, "DATA", tmp_path)
    _, frame_ids = daq()
    l1, constants = l1_readbacks(), constant_values()
    replay("roi")
    summary = session.phases_test(seconds=6)
    assert list(summary["pvs"]) == [*pvs.L1, *pvs.L2]
    for name in next(iter(l1.values())):                  # one per frame, the first on subscribing
        values = [l1[p][name] for p in frame_ids]
        s = summary["pvs"][name]
        assert {k: s[k] for k in ("updates", "median", "first", "last", "zeros")} == {
            "updates": 30, "median": np.median(values), "first": values[0], "last": values[-1], "zeros": 0}
        assert s["std"] == np.std(values) and [s["min"], s["max"]] == [min(values), max(values)]
        assert s["pulse_id_steps"] == session.pulse_id_steps(frame_ids) and s["first_age_s"] > 0
    for name, value in constants.items():                  # one update, on subscribing
        s = summary["pvs"][name]
        assert s["updates"] == 1 and s["median"] == s["min"] == s["max"] == value and s["std"] == 0
        assert s["pulse_id_steps"] == {} and s["first_age_s"] > 0
    for readback, setpoint in pvs.L1_SETPOINTS.items():
        median = np.median([l1[p][readback] for p in frame_ids])
        assert summary["readback - setpoint"][readback] == median - PREQ[setpoint]
    [record] = tmp_path.glob("*_S6/phases_test_*.json")
    assert json.loads(record.read_text()) == json.loads(json.dumps(summary))     # JSON keys are strings
    saved = np.load(record.with_suffix(".npz"))
    assert saved["value"].size == 30 * len(next(iter(l1.values()))) + len(constants)
    assert record.with_suffix(".png").stat().st_size > 0
