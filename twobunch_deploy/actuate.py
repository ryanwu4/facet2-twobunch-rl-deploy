"""Writes to the machine (v2). The only module that writes: tests/test_read_only.py forbids write calls everywhere
else, and here allows them only in put_preq() and turn_knob().

W1  arm(), disarm(), move_l1(), move_l2(), undo_all(), state(); pvaccess_version() for the session's record

What can be written is in pvs: the two L1 phase setpoints (PREQ, absolute) and the L2 phase multiknob (a delta), each
write at most one policy action, and the net move per armed session within a budget. Nothing is written before arm(),
which takes a lock (one armed session at a time, across processes) and checks that PREQ's feedback is off. Every write
goes in the session's ledger (data/<date>_V2/ledger_<HHMMSS>.jsonl), once before it is sent and once after its
readback check. undo_all() reverses a ledger's writes from any process, so a session that died can be undone.
"""
from __future__ import annotations

import datetime
import importlib.metadata
import json
import os
import socket
import time
from pathlib import Path

from . import machine, pvs, session

QUIET_S = 10.0          # arm() refuses if PREQ updated in this long: its feedback is still on
IDLE_S = 600.0          # an armed session disarms after this long without a write
L1_TOL = 1e-3           # deg: a PREQ must read back the value written to it within this
L2_TOL = 0.01           # deg: each subbooster setpoint must move by the multiknob's delta within this
KNOB_TIMEOUT_S = 10.0   # the multiknob call
KNOB_SETTLE_S = 5.0     # the subbooster setpoints reach the gateway about once a second: wait at most this long

_session: dict | None = None    # this process's armed session: name, ledger, time of the last write


def lock_path() -> Path:
    return session.DATA / "armed.lock"


def arm(name: str) -> dict:
    """Arm this process to write, as `name`: take the lock, check that PREQ's feedback is off (no PREQ update in
    QUIET_S), read the starting setpoints and open a ledger with them as its first line, which is returned.

    Refuses if this process is armed already, if a running process holds the lock, or if PREQ updated. A lock left by
    a process that no longer runs on this host is taken over, saying so: then undo_all() can reverse its ledger.
    """
    global _session
    if _session is not None:
        raise RuntimeError(f"already armed, as {_session['name']}")
    if lock_path().exists():
        held = json.loads(lock_path().read_text())
        if held["host"] != socket.gethostname() or _running(held["pid"]):
            raise RuntimeError(f"armed by {held['name']} (pid {held['pid']} on {held['host']}) since {held['time']}")
        print(f"taking over the lock {held['name']} left (pid {held['pid']} no longer runs); "
              f"its ledger: {held['ledger']}")
    with machine.updates(pvs.L1_SETPOINTS) as updates:
        time.sleep(QUIET_S)
    moved = len(updates) - len(pvs.L1_SETPOINTS)            # a monitor's first update is the current value
    if moved:
        raise RuntimeError(f"PREQ updated {moved} times in {QUIET_S:.0f} s: its feedback is on")
    now = datetime.datetime.now()
    ledger = (session.DATA / f"{now:%Y-%m-%d}_V2" / f"ledger_{now:%H%M%S}.jsonl").resolve()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    first = {"armed": name, "time": now.isoformat(timespec="seconds"), "pid": os.getpid(),
             "host": socket.gethostname(), "write": pvs.WRITE, "multiknob": pvs.MULTIKNOB, "p4p": pvaccess_version(),
             "start": {n: machine.get(n) for n in [*pvs.L1_SETPOINTS, *pvs.MULTIKNOB_MOVES]}}
    with open(ledger, "x") as f:                            # never over an existing ledger
        print(json.dumps(first), file=f)
    lock_path().write_text(json.dumps({"name": name, "pid": os.getpid(), "host": socket.gethostname(),
                                       "time": first["time"], "ledger": str(ledger)}))
    _session = {"name": name, "ledger": ledger, "last": time.time()}
    return first


def disarm() -> None:
    """Stop writing from this process and release the lock."""
    global _session
    if _session is not None and lock_path().exists() and json.loads(lock_path().read_text())["pid"] == os.getpid():
        lock_path().unlink()
    _session = None


def move_l1(delta: float) -> dict:
    """Move L1 by `delta` deg: both PREQs to their current values plus `delta`. Refuses unless armed, within one action
    and the budget, and unless both PREQs still read what this session last wrote (or found at arm()): otherwise a
    feedback or someone else moved them, and it disarms. Returns the ledger entry."""
    _check("L1", delta)
    before = _read(pvs.L1_SETPOINTS)
    expected = _last("L1")
    if any(abs(before[n] - expected[n]) > L1_TOL for n in before):
        disarm()
        raise RuntimeError(f"PREQ is {before}, not {expected} as last written: something else moved it; disarmed")
    return _write_l1(delta, before, {n: v + delta for n, v in before.items()})


def move_l2(delta: float) -> dict:
    """Move L2 by `delta` deg through the multiknob. Refuses unless armed, within one action and the budget, and unless
    the LI12-14 subbooster setpoints still read what this session last left (or found at arm()); otherwise it
    disarms. Returns the ledger entry."""
    _check("L2", delta)
    before = _read(pvs.MULTIKNOB_MOVES)
    expected = _last("L2")
    if any(abs(before[n] - expected[n]) > L2_TOL for n in before):
        disarm()
        raise RuntimeError(f"the subbooster setpoints are {before}, not {expected}: something else moved them; "
                           "disarmed")
    return _write_l2(delta, before)


def undo_all(ledger: str | Path | None = None) -> list[dict]:
    """Reverse the writes in `ledger` (this session's if None), newest first, through the same checks as a move; the
    reversals go in this session's ledger, each naming the write it reverses. Each L1 write is undone only if both
    PREQs still read its after values (they are set back to its before values), each L2 write only if the subbooster
    setpoints still read its after values (the knob turns by minus its delta). Writes that a ledger already records as
    reversed are skipped, so undo_all() can be run again after stopping part way. Only completed writes are reversed:
    one that ended as "mismatch" or "not applied" disarmed the session, and its ledger lines show what moved, for ops
    to restore by hand. Needs this process armed. Returns the reversals' ledger entries."""
    if _session is None:
        raise RuntimeError("not armed")
    path = (Path(ledger) if ledger is not None else _session["ledger"]).resolve()
    reversed_already = _reversed(path)
    done = [e for e in _entries(path) if e["state"] == "done" and e["undo"] is None]
    undone = []
    for e in reversed(done):
        if e["write"] in reversed_already:
            continue
        names = pvs.L1_SETPOINTS if e["kind"] == "L1" else pvs.MULTIKNOB_MOVES
        tol = L1_TOL if e["kind"] == "L1" else L2_TOL
        now = _read(names)
        if not all(abs(now[n] - e["after"][n]) <= tol for n in names):
            disarm()
            raise RuntimeError(f"write {e['write']} of {path} can't be undone: its setpoints are {now}, not "
                               f"{e['after']}; disarmed")
        _check(e["kind"], -e["delta"])
        undo = {"ledger": str(path), "write": e["write"]}
        if e["kind"] == "L1":
            undone.append(_write_l1(-e["delta"], now, e["before"], undo))
        else:
            undone.append(_write_l2(-e["delta"], now, undo))
    return undone


def state() -> dict:
    """This process's session: armed or not, as whom, its ledger, the net move per kind, and any write logged as pending
    with no result (the process died while sending it, or the call is still running)."""
    if _session is None:
        return {"armed": False, "lock": json.loads(lock_path().read_text()) if lock_path().exists() else None}
    entries = _entries(_session["ledger"])
    ended = {e["write"] for e in entries if e["state"] != "pending"}
    return {"armed": True, "name": _session["name"], "ledger": str(_session["ledger"]),
            "net": {kind: _net(kind) for kind in pvs.WRITE},
            "pending": [e for e in entries if e["state"] == "pending" and e["write"] not in ended]}


def pvaccess_version() -> str:
    """The version of p4p, the PVAccess client the multiknob call uses, or why it can't be imported."""
    try:
        importlib.import_module("p4p")
        return importlib.metadata.version("p4p")
    except Exception as e:
        return repr(e)


def put_preq(target: dict) -> None:
    """Write each PREQ in `target` its value over Channel Access, waiting for each write to complete."""
    import epics

    for name, value in target.items():
        if name not in pvs.L1_SETPOINTS:
            raise ValueError(f"{name} is not an L1 setpoint")
        pv = epics.PV(name, auto_monitor=False)
        pv.put(value, wait=True, timeout=5)
        pv.disconnect()


def turn_knob(delta: float) -> str:
    """Turn the L2 phase multiknob by `delta` deg: the same AIDA-PVA call as ops' multiknob wrapper in F2_pytools (an
    NTURI to the service with the knob's name and the delta), with a timeout. Returns the reply as text; raises on an
    error or a timeout."""
    from p4p.client.thread import Context
    from p4p.nt import NTURI

    request = NTURI([("VALUE", "s"), ("MKB", "s")]).wrap(
        scheme="pva", path=pvs.MULTIKNOB_SERVICE, kws={"VALUE": delta, "MKB": pvs.MULTIKNOB})
    ctxt = Context("pva")
    try:
        return str(ctxt.rpc(pvs.MULTIKNOB_SERVICE, request, timeout=KNOB_TIMEOUT_S))
    finally:
        ctxt.close()


def _check(kind: str, delta: float) -> None:
    """Refuse a write unless armed and active in the last IDLE_S (else disarm), `delta` is at most one step, and the
    session's net move stays within the budget."""
    if _session is None:
        raise RuntimeError("not armed")
    if time.time() - _session["last"] > IDLE_S:
        disarm()
        raise RuntimeError(f"disarmed: no write in {IDLE_S:.0f} s")
    limit = pvs.WRITE[kind]
    if abs(delta) > limit["step"]:
        raise ValueError(f"{kind} {delta:+.3f} deg is more than one step ({limit['step']} deg)")
    if abs(_net(kind) + delta) > limit["budget"] + 1e-9:
        raise ValueError(f"{kind} {delta:+.3f} deg would take the net move past the budget (+-{limit['budget']} deg)")


def _write_l1(delta: float, before: dict, target: dict, undo: dict | None = None) -> dict:
    entry = _log({"kind": "L1", "delta": delta, "before": before, "target": target, "undo": undo, "state": "pending"})
    put_preq(target)
    after = _read(pvs.L1_SETPOINTS)
    ok = all(abs(after[n] - target[n]) <= L1_TOL for n in target)
    entry = _log({**entry, "after": after, "state": "done" if ok else "mismatch"})
    if not ok:
        disarm()
        raise RuntimeError(f"PREQ reads {after} after writing {target}; disarmed")
    return entry


def _write_l2(delta: float, before: dict, undo: dict | None = None) -> dict:
    entry = _log({"kind": "L2", "delta": delta, "before": before, "undo": undo, "state": "pending"})
    try:
        reply = turn_knob(delta)
    except Exception as e:              # a timeout or a refusal: the setpoints say whether the knob turned
        reply = repr(e)
    end = time.time() + KNOB_SETTLE_S
    while True:
        after = _read(pvs.MULTIKNOB_MOVES)
        moved = {n: after[n] - before[n] for n in before}
        if all(abs(m - delta) <= L2_TOL for m in moved.values()) or time.time() > end:
            break
        time.sleep(0.5)
    if all(abs(m - delta) <= L2_TOL for m in moved.values()):
        state = "done"
    else:
        state = "not applied" if all(abs(m) <= L2_TOL for m in moved.values()) else "mismatch"
    entry = _log({**entry, "reply": reply, "after": after, "state": state})
    if state != "done":
        disarm()
        raise RuntimeError(f"multiknob {delta:+.3f} deg: {state} (reply {reply}; setpoints moved {moved}); disarmed")
    return entry


def _log(entry: dict) -> dict:
    """Append `entry` to this session's ledger, numbered and timed; returns it as written. A new write gets the next
    number; its result line repeats it with the same number."""
    if "write" not in entry:
        entry = {"write": sum(1 for e in _entries(_session["ledger"]) if e["state"] == "pending") + 1, **entry}
    entry = {**entry, "time": datetime.datetime.now().isoformat(timespec="milliseconds")}
    with open(_session["ledger"], "a") as f:
        print(json.dumps(entry), file=f)
    _session["last"] = time.time()
    return entry


def _reversed(path: Path) -> set[int]:
    """The writes of the ledger at `path` that a ledger under data/ records as reversed (undo_all)."""
    return {e["undo"]["write"] for other in session.DATA.glob("*_V2/ledger_*.jsonl") for e in _entries(other)
            if e["state"] == "done" and e["undo"] is not None and e["undo"]["ledger"] == str(path)}


def _entries(path: Path) -> list[dict]:
    """A ledger's write lines (all but the first, which holds the session and its starting setpoints)."""
    return [json.loads(line) for line in Path(path).read_text().splitlines()[1:]]


def _last(kind: str) -> dict:
    """The setpoints of `kind` as this session's last write left them, or as arm() found them."""
    done = [e for e in _entries(_session["ledger"]) if e["kind"] == kind and e["state"] == "done"]
    if done:
        return done[-1]["after"]
    start = json.loads(Path(_session["ledger"]).read_text().splitlines()[0])["start"]
    return {n: start[n] for n in (pvs.L1_SETPOINTS if kind == "L1" else pvs.MULTIKNOB_MOVES)}


def _net(kind: str) -> float:
    return sum(e["delta"] for e in _entries(_session["ledger"]) if e["kind"] == kind and e["state"] == "done")


def _read(names: list[str]) -> dict:
    return {n: machine.get(n) for n in names}


def _running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
