"""S8: the bunch-length and drift PVs, monitored during each step's capture and by drift_test(), and the magnets, read
at each step and by optics_test().

The synthetic tests run everywhere; the session tests on the soft IOC only where its DAQ data are (conftest.py).
step()'s record of them is tested in test_step.py.
"""
import json
import time

import numpy as np
import pytest

from conftest import DRIFT_STAND_IN, XTCAV, daq, magnet_values, needs_daq, per_shot_values
from twobunch_deploy import acquire, machine, pvs, session


def test_pvs():
    assert all(name in pvs.READ for name in [*pvs.BLEN, *pvs.DRIFT, *pvs.MAGNETS])
    assert [name.split(":")[1] for name in pvs.BLEN] == ["IN10", "LI11", "LI14", "LI20"]
    assert all(name.endswith(":BRAWBR") for name in pvs.BLEN)                   # read at beam rate
    assert list(pvs.MAGNETS_IN_DAQ) == list(pvs.MAGNETS)


@pytest.mark.parametrize("run", ["TEST_15668", "E300_15671", "E300_15673"])
def test_magnets_in_daq(run):
    """pvs.MAGNETS_IN_DAQ is each supply's median in TEST_15668, and the other two runs agree within 0.01%, the
    correctors (near 0) within 2e-5."""
    mat = XTCAV / run / f"{run}.mat"
    if not mat.exists():
        pytest.skip(f"needs {mat}")
    for name, value in magnet_values(mat).items():
        assert abs(pvs.MAGNETS_IN_DAQ[name] - value) <= max(1e-4 * abs(value), 2e-5), name


@needs_daq
def test_drift_test(replay, tmp_path, monkeypatch):
    """drift_test() on the soft IOC: the bunch-length signals of each frame's shot (the injector's NaN for about half
    of them, as in the DAQ), the drift stand-ins constant."""
    monkeypatch.setattr(session, "DATA", tmp_path)
    _, frame_ids = daq()
    per_shot = per_shot_values()
    replay("roi")
    summary = session.drift_test(seconds=6)
    assert list(summary["pvs"]) == [*pvs.BLEN, *pvs.DRIFT]
    for name in pvs.BLEN:                                    # one per frame, the first on subscribing
        values = [per_shot[p][name] for p in frame_ids]
        finite = [v for v in values if np.isfinite(v)]
        s = summary["pvs"][name]
        assert {k: s[k] for k in ("updates", "median", "first", "last", "zeros", "nans")} == {
            "updates": 30, "median": np.median(finite), "first": acquire.number(values[0]),
            "last": acquire.number(values[-1]), "zeros": values.count(0), "nans": 30 - len(finite)}
        assert [s["std"], s["min"], s["max"]] == [np.std(finite), min(finite), max(finite)]
        assert s["pulse_id_steps"] == session.pulse_id_steps(frame_ids)
    assert summary["pvs"]["BLEN:IN10:596:BRAWBR"]["nans"] > 0
    for name, value in DRIFT_STAND_IN.items():               # one update, on subscribing
        s = summary["pvs"][name]
        assert s["updates"] == 1 and s["median"] == s["min"] == s["max"] == value and s["std"] == 0
    [record] = tmp_path.glob("*_S8/drift_test_*.json")
    assert json.loads(record.read_text()) == json.loads(json.dumps(summary))
    assert np.load(record.with_suffix(".npz"))["value"].size == 30 * len(pvs.BLEN) + len(DRIFT_STAND_IN)
    assert record.with_suffix(".png").stat().st_size > 0


@needs_daq
def test_optics_test(replay, tmp_path, monkeypatch):
    """optics_test() on the soft IOC, which serves TEST_15668's medians: each supply's value and its difference from
    pvs.MAGNETS_IN_DAQ (rounded to 4 decimals)."""
    monkeypatch.setattr(session, "DATA", tmp_path)
    served = magnet_values()
    replay("roi")
    summary = session.optics_test()
    assert list(summary) == list(pvs.MAGNETS)
    for name, s in summary.items():
        assert s["value"] == served[name] and s["daq"] == pvs.MAGNETS_IN_DAQ[name]
        assert s["difference"] == served[name] - s["daq"]
        assert s["difference_pct"] == 100 * s["difference"] / abs(s["daq"])
        assert abs(s["difference"]) <= 5e-5                       # MAGNETS_IN_DAQ's rounding
    [record] = tmp_path.glob("*_S8/optics_test_*.json")
    assert json.loads(record.read_text()) == json.loads(json.dumps(summary))


@needs_daq
def test_pv_log_beside_a_monitor(replay, tmp_path, monkeypatch):
    """The optional PV log (R16) runs beside another monitor of the same PVs, as during a step: that monitor gets every
    shot as without the log, the log has them too, and it starts a new file every file_s."""
    monkeypatch.setattr(session, "DATA", tmp_path)
    name = list(pvs.BLEN)[1]                         # BC11's
    replay("roi")                                   # 2 s, then 29 more frames at 10 Hz
    session.start_pv_log([name, pvs.TCAV_AMPLITUDE], file_s=2.0)
    with machine.updates([name]) as during:
        time.sleep(6)
    session.stop_pv_log()
    files = sorted(tmp_path.glob("*_V2/pvlog_*.npz"))
    logged = set()
    for f in files:
        z = np.load(f)
        logged |= {int(p) for n, p in zip(z["name"], z["pulse_id"]) if n == name}
    assert len(files) >= 2 and len(during) == 30                    # one per frame, as test_drift_test
    assert {u.pulse_id for u in during[1:]} <= logged
