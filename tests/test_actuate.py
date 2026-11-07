"""W1: the write path (actuate.py), on the soft IOC and the stand-in multiknob service (mkb_server.py). They run only
where the soft IOC's DAQ data are (conftest.py).

The soft IOC serves the PREQs as stand-ins that change with every frame it publishes, like the feedback that writes the
real ones, and the subbooster setpoints as constants that the stand-in multiknob moves. Each test restores them.
"""
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import epics
import pytest

from conftest import REPO, needs_daq
from twobunch_deploy import actuate, machine, pvs, session

SETPOINTS = [*pvs.L1_SETPOINTS, *pvs.MULTIKNOB_MOVES]


@pytest.fixture
def fresh(replay, knob, tmp_path, monkeypatch):
    """The soft IOC once its replay has ended, so the PREQs are still; records under tmp_path; a 1 s quiet check. Puts
    every setpoint back afterwards."""
    monkeypatch.setattr(session, "DATA", tmp_path)
    monkeypatch.setattr(actuate, "QUIET_S", 1.0)
    replay("roi")
    time.sleep(6)                               # 2 s wait, then 30 frames at 10 Hz, a PREQ update with each
    start = {n: machine.get(n) for n in SETPOINTS}
    yield start
    actuate.disarm()
    for name, value in start.items():
        epics.caput(name, value, wait=True)


@pytest.fixture
def armed(fresh):
    first = actuate.arm("test")
    yield fresh, Path(actuate.state()["ledger"]), first


def lines(ledger):
    return [json.loads(line) for line in ledger.read_text().splitlines()]


def subprocess_python(code, data):
    """Run `code` in a fresh process on the same soft IOC and multiknob stand-in, with records under `data`."""
    setup = (f"from pathlib import Path; from twobunch_deploy import actuate, session; "
             f"session.DATA = Path({str(data)!r}); actuate.QUIET_S = 1.0; ")
    return subprocess.run([sys.executable, "-c", setup + code], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(REPO)})


@needs_daq
def test_arm_refused_while_preq_moves(replay, knob, tmp_path, monkeypatch):
    monkeypatch.setattr(session, "DATA", tmp_path)
    monkeypatch.setattr(actuate, "QUIET_S", 1.0)
    replay("roi")                               # publishing: the stand-in PREQs change at 10 Hz
    time.sleep(2.5)
    with pytest.raises(RuntimeError, match="feedback is on"):
        actuate.arm("test")
    assert not actuate.lock_path().exists() and actuate.state() == {"armed": False, "lock": None}


@needs_daq
def test_moves(armed):
    start, ledger, first = armed
    assert first["start"] == start and first["armed"] == "test" and first["write"] == pvs.WRITE
    assert json.loads(actuate.lock_path().read_text())["ledger"] == str(ledger)
    l1 = actuate.move_l1(0.1)
    l2 = actuate.move_l2(0.3)
    now = {n: machine.get(n) for n in SETPOINTS}
    for name in pvs.L1_SETPOINTS:
        assert abs(now[name] - (start[name] + 0.1)) < 1e-6
    for name in pvs.MULTIKNOB_MOVES:
        assert abs(now[name] - (start[name] + 0.3)) < 1e-6
    written = lines(ledger)[1:]
    assert [(e["write"], e["kind"], e["state"]) for e in written] == [
        (1, "L1", "pending"), (1, "L1", "done"), (2, "L2", "pending"), (2, "L2", "done")]
    assert l1 == written[1] and l2 == written[3] and l2["reply"].endswith("0.3")
    assert l1["before"] == {n: start[n] for n in pvs.L1_SETPOINTS} and l1["undo"] is None
    assert actuate.state()["net"] == {"L1": 0.1, "L2": 0.3} and actuate.state()["pending"] == []


@needs_daq
def test_refusals(armed, monkeypatch):
    start, ledger, _ = armed
    with pytest.raises(ValueError, match="one step"):
        actuate.move_l1(0.26)
    with pytest.raises(ValueError, match="one step"):
        actuate.move_l2(-0.31)
    for _ in range(4):
        actuate.move_l1(-0.25)                  # the budget: 1.0 deg
    with pytest.raises(ValueError, match="budget"):
        actuate.move_l1(-0.01)
    actuate.move_l1(0.25)                       # back towards the start is within it
    assert actuate.state()["net"]["L1"] == -0.75
    monkeypatch.setattr(actuate, "IDLE_S", 0.0)
    with pytest.raises(RuntimeError, match="idle|no write"):
        actuate.move_l2(0.1)
    assert actuate.state()["armed"] is False and not actuate.lock_path().exists()
    with pytest.raises(RuntimeError, match="not armed"):
        actuate.move_l1(0.1)
    assert sum(e["state"] == "done" for e in lines(ledger)[1:]) == 5           # nothing written by the refusals


@needs_daq
def test_moved_elsewhere(armed):
    """A setpoint changed since this session's last write (a feedback back on, or ops): the next move refuses and
    disarms, and undo_all() won't undo that write."""
    start, ledger, _ = armed
    actuate.move_l1(0.1)
    epics.caput(pvs.L1_SETPOINTS[0], start[pvs.L1_SETPOINTS[0]] + 0.15, wait=True)
    with pytest.raises(RuntimeError, match="something else moved"):
        actuate.move_l1(0.1)
    assert actuate.state()["armed"] is False
    actuate.arm("test")
    with pytest.raises(RuntimeError, match="can't be undone"):
        actuate.undo_all(ledger)


@needs_daq
@pytest.mark.parametrize("fault", ["unknown knob", "no service"])
def test_knob_fails(armed, monkeypatch, fault):
    """The multiknob call fails (a refusal, or a timeout): the setpoints didn't move, so the write is "not applied", and
    it disarms."""
    start, ledger, _ = armed
    if fault == "unknown knob":
        monkeypatch.setattr(pvs, "MULTIKNOB", "nothing")
    else:
        monkeypatch.setattr(pvs, "MULTIKNOB_SERVICE", "NO:SERVICE")
        monkeypatch.setattr(actuate, "KNOB_TIMEOUT_S", 0.5)
    monkeypatch.setattr(actuate, "KNOB_SETTLE_S", 0.5)
    with pytest.raises(RuntimeError, match="not applied"):
        actuate.move_l2(0.3)
    assert lines(ledger)[-1]["state"] == "not applied" and actuate.state()["armed"] is False
    assert {n: machine.get(n) for n in pvs.MULTIKNOB_MOVES} == {n: start[n] for n in pvs.MULTIKNOB_MOVES}


@needs_daq
def test_lock(armed, tmp_path):
    """While this process is armed, another can't arm; once it is gone, its lock can be taken over."""
    result = subprocess_python("actuate.arm('other')", tmp_path)
    assert result.returncode != 0 and "armed by test" in result.stderr


@needs_daq
def test_undo_all_from_a_fresh_process(armed, tmp_path):
    """The session dies after three writes: a fresh process takes over its lock and undoes them, newest first; a second
    undo_all() finds nothing left to do."""
    start, ledger, _ = armed
    actuate.move_l1(0.25)
    actuate.move_l1(0.25)
    actuate.move_l2(-0.3)
    actuate._session = None                     # this process forgets it was armed, as if it had died
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    held = json.loads(actuate.lock_path().read_text())
    actuate.lock_path().write_text(json.dumps({**held, "pid": dead.pid}))
    result = subprocess_python(f"actuate.arm('undo'); print(len(actuate.undo_all({str(ledger)!r}))); "
                               f"print(len(actuate.undo_all({str(ledger)!r}))); actuate.disarm()", tmp_path)
    assert result.returncode == 0, result.stderr
    assert "taking over the lock test left" in result.stdout and result.stdout.split()[-2:] == ["3", "0"]
    now = {n: machine.get(n) for n in SETPOINTS}
    assert all(abs(now[n] - start[n]) < 1e-6 for n in SETPOINTS)
    [undo_ledger] = [p for p in ledger.parent.glob("ledger_*.jsonl") if p != ledger]
    undone = [e for e in lines(undo_ledger)[1:] if e["state"] == "done"]
    assert [(e["kind"], e["delta"], e["undo"]["write"]) for e in undone] == [("L2", 0.3, 3), ("L1", -0.25, 2),
                                                                            ("L1", -0.25, 1)]
    sent = [datetime.datetime.fromisoformat(e["time"]) for e in lines(undo_ledger)[1:] if e["state"] == "pending"]
    assert all((b - a).total_seconds() >= actuate.UNDO_PAUSE_S for a, b in zip(sent, sent[1:]))      # paced


@needs_daq
def test_one_multiknob_connection_per_session(armed, monkeypatch):
    """The session's multiknob turns share one PVAccess connection, opened on the first and closed by disarm()."""
    assert actuate._session["knob"] is None
    actuate.move_l2(0.3)
    knob = actuate._session["knob"]
    actuate.move_l2(-0.3)
    assert actuate._session["knob"] is knob
    closed, close = [], knob.close
    monkeypatch.setattr(knob, "close", lambda: (closed.append(True), close()))
    actuate.disarm()
    assert closed == [True]


@needs_daq
def test_staircase(armed, monkeypatch):
    """W2's staircase on L1 with a 0.5 deg budget: two steps up, the third refused, two back down."""
    start, ledger, _ = armed
    monkeypatch.setitem(pvs.WRITE, "L1", {"step": 0.25, "budget": 0.5})
    summary = session.staircase("L1", 0.25, 2, hold_s=0.5)
    assert "budget" in summary["refused"]
    assert [m["delta"] for m in summary["moves"]] == [0.25, 0.25, -0.25, -0.25]
    for m in summary["moves"]:
        name = pvs.L1_SETPOINTS[0]
        assert abs(m["hold_after"][name] - m["hold_before"][name] - m["delta"]) < 1e-6
    now = {n: machine.get(n) for n in pvs.L1_SETPOINTS}
    assert all(abs(now[n] - start[n]) < 1e-6 for n in now)
    [record] = session.DATA.glob("*_V2/staircase_L1_*.json")
    assert json.loads(record.read_text()) == json.loads(json.dumps(summary))
    assert record.with_suffix(".npz").exists()
