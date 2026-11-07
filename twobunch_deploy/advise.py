"""P1b (v2): the policy's goals, episodes and proposals, and the approved cycle. Writes only through actuate's moves, so
arm(), the limits, the budget and the ledger apply as in W1. Each episode's proposals and decisions go in its log,
data/<date>_V2/episode_<HHMMSS_mmm>.jsonl; goals in data/<date>_V2/goals.json.

A session (v2 plan, 11-06):
  step(background, streak)      readings, as session.step(), remembered here with the writes sent around them
  record_goal("A")              the mean of the last GOAL_READINGS valid readings, with no write between them
  actuate.arm("name"), actuate.move_l1() / move_l2()   detune, inside the budget
  use_goal("A")                 an episode on the latest reading, with no history, as training's reset
  cycle(background, streak)     step(), propose(), and when armed a y/N prompt: y applies, anything else declines
  end_episode(); actuate.undo_all() or keep; actuate.disarm()

Unarmed, cycle() is shadow mode: proposals are logged and nothing is written. The policy's inputs follow training's
bookkeeping (controller.py): a declined, expired, unanswered or shadow proposal is "asked for nothing, got nothing"
(action and observed change 0); an applied one is the policy's action and the change written (L1: the PREQ change;
L2: the multiknob delta, not written below MIN_L2_MOVE and then 0). A proposal is clipped to the session's budget, as
training's knob window clipped actions. An invalid reading gets no proposal and the episode doesn't advance. After the
policy's episode length the episode restarts on the same goal.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import actuate, controller, pvs, session

POLICY = "nc_zs_screen_noblen_jit_s0"
GOAL_READINGS = 3       # record_goal() averages this many valid readings
EXPIRE_S = 60.0         # a proposal can be applied for this long after it is made
MIN_L2_MOVE = 0.02      # deg: a smaller L2 change isn't written (under the multiknob's readback tolerance it would
                        # look "not applied" and disarm)
KINDS = {"L1PhaseSet": "L1", "L2PhaseSet": "L2"}

_policy: controller.Policy | None = None
_readings: list[dict] = []          # this kernel's step()s: the Reading, its features (None if invalid), the writes
                                    # sent before and after it (actuate.writes)
_episode: dict | None = None        # goal, the policy's inputs, the reading they are at, the last decision's outcome
_proposal: dict | None = None       # the latest proposal


def policy() -> controller.Policy:
    """The policy (POLICY), loaded once; its controls and action steps must be the writable ones and within limits."""
    global _policy
    if _policy is None:
        p = controller.Policy(POLICY)
        m = p.manifest
        if [KINDS.get(c) for c in m["controls"]] != list(pvs.WRITE) or any(
                s > pvs.WRITE[KINDS[c]]["step"] + 1e-9 for c, s in zip(m["controls"], m["action_step"])):
            raise ValueError(f"{POLICY}: controls {m['controls']} with steps {m['action_step']} don't fit pvs.WRITE")
        _policy = p
    return _policy


def step(background, streak_um_per_deg: float, seconds: float = 15) -> session.Reading:
    """session.step(), remembered for record_goal() and propose() with the writes sent before and after it."""
    before = actuate.writes
    reading = session.step(background, streak_um_per_deg, seconds)
    _readings.append({"reading": reading, "values": values(reading), "writes": (before, actuate.writes)})
    return reading


def values(reading: session.Reading) -> dict | None:
    """A valid reading's features as the policy takes them (m); None if the reading is invalid."""
    r = reading.record
    if not r["valid"]:
        return None
    read = r["reading"]
    return {"bunch_spacing": 1e-6 * read["spacing_um"], "drive_sigma_z": 1e-6 * read["drive_sigma_z_um"],
            "drive_charge_fraction": read["drive_charge_fraction"], "screen_valid_deployed": 1.0}


def record_goal(name: str) -> dict:
    """Goal `name`: the mean of each goal feature over the last GOAL_READINGS valid readings, saved to today's
    goals.json. Refused if there are fewer, or if a write was sent between the first and the last."""
    valid = [r for r in _readings if r["values"] is not None][-GOAL_READINGS:]
    if len(valid) < GOAL_READINGS:
        raise RuntimeError(f"{len(valid)} valid readings so far; a goal needs {GOAL_READINGS}")
    if valid[0]["writes"][0] != valid[-1]["writes"][1]:
        raise RuntimeError(f"a write was sent between the last {GOAL_READINGS} valid readings: take more steps")
    goal = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
            "readings": [r["reading"].path for r in valid],
            **{g["key"]: float(np.mean([r["values"][g["key"]] for r in valid])) for g in policy().manifest["goals"]}}
    path = _day() / "goals.json"
    goals = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**goals, name: goal}, indent=1) + "\n")
    return goal


def use_goal(name: str) -> dict:
    """Start an episode on goal `name` (today's goals.json) from the latest reading, which must be valid and taken with
    no write since: the policy's inputs with no history, as training's reset. Manual moves are refused until
    end_episode(); a running episode ends first. Returns the episode log's first line."""
    global _episode, _proposal
    path = _day() / "goals.json"
    goals = json.loads(path.read_text()) if path.exists() else {}
    if name not in goals:
        raise RuntimeError(f"no goal {name!r} today (goals: {sorted(goals)}): record_goal() first")
    goal = goals[name]
    latest = _latest()
    if latest["values"] is None:
        raise RuntimeError("the latest reading is invalid: an episode starts on a valid one")
    end_episode()
    now = datetime.datetime.now()
    _episode = {"name": name, "goal": goal, "inputs": controller.Episode(policy(), goal, latest["values"]),
                "reading": latest, "proposed": False, "outcome": None, "n": 0,
                "log": _day() / f"episode_{now:%H%M%S}_{now.microsecond // 1000:03d}.jsonl"}
    _proposal = None
    actuate.episode = name
    files = [controller.POLICIES / POLICY / f for f in ("policy.json", "weights.npz")]
    return _log({"goal": name, "goal_record": goal, "policy": POLICY,
                 "policy_sha256": hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest(),
                 "code_sha256": session.code_sha256(), "reading": latest["reading"].path})


def end_episode() -> None:
    """End the running episode: manual moves are allowed again."""
    global _episode, _proposal
    if _episode is not None:
        _log({"ended": True})
    _episode, _proposal = None, None
    actuate.episode = None


def propose() -> dict | None:
    """The policy's proposal from the latest reading, logged and printed; None, with the reason, if the reading is
    invalid. A newer reading first advances the episode with the last decision's outcome (none, or not applied: action
    and change 0). The change is clipped to the armed session's budget (unarmed: nothing has been written)."""
    global _proposal
    e = _running()
    latest = _latest()
    if latest["values"] is None:
        failed = [check for check, ok in latest["reading"].record["checks"].items() if not ok]
        print(f"hold: invalid reading ({'; '.join(failed)}): no proposal, the episode doesn't advance")
        return None
    if latest is e["reading"]:
        if e["proposed"]:
            raise RuntimeError("already proposed on this reading: step() first")
    else:
        action, change = e["outcome"] or (np.zeros(2), np.zeros(2))
        e["inputs"].advance(action, change, latest["values"])
        e.update(reading=latest, outcome=None)
        if e["inputs"].over:
            e["inputs"] = controller.Episode(policy(), e["goal"], latest["values"])
            print(f"episode at {policy().manifest['episode_length']} steps: restarted on goal {e['name']}")
            _log({"restarted": True, "reading": latest["reading"].path})
    p, armed = policy(), actuate.state()["armed"]
    inputs = e["inputs"].inputs()
    action = p.act(inputs)
    net = np.array([actuate.state()["net"][k] for k in pvs.WRITE]) if armed else np.zeros(2)
    budget = np.array([pvs.WRITE[k]["budget"] for k in pvs.WRITE])
    change = controller.clip(action, p.manifest["action_step"], net, budget)
    e["n"], e["proposed"] = e["n"] + 1, True
    _proposal = _log({"proposal": e["n"], "made": time.time(), "reading": latest["reading"].path,
                      "episode_step": e["inputs"].steps, "armed": armed, "values": latest["values"],
                      "action": action.tolist(), "asked": (action * p.manifest["action_step"]).tolist(),
                      "change": change.tolist(), "net": net.tolist(), "inputs": inputs.tolist(), "decision": None})
    print(_line(_proposal, e))
    return _proposal


def apply() -> dict:
    """Write the latest proposal, once: L1 (unless its change is 0), then L2 (unless under MIN_L2_MOVE). Refused if it
    has been decided, expired (EXPIRE_S) or a newer reading taken, or if not armed. A failed write ends the episode
    (actuate disarms)."""
    p = _undecided()
    if not actuate.state()["armed"]:
        raise RuntimeError("not armed: the proposal stays undecided (shadow)")
    if time.time() - p["made"] > EXPIRE_S:
        _decide(p, "expired", np.zeros(2))
        raise RuntimeError(f"proposal {p['proposal']} expired ({EXPIRE_S:.0f} s): not applied")
    got = np.zeros(2)
    try:
        if p["change"][0] != 0:
            actuate.move_l1(p["change"][0], proposal=p["proposal"])
            got[0] = p["change"][0]
        if abs(p["change"][1]) >= MIN_L2_MOVE:
            actuate.move_l2(p["change"][1], proposal=p["proposal"])
            got[1] = p["change"][1]
    except Exception:
        _decide(p, "failed", got)
        end_episode()
        raise
    return _decide(p, "applied", got)


def decline() -> dict:
    """Decline the latest proposal: the next step's inputs say it asked for nothing and got nothing."""
    return _decide(_undecided(), "declined", np.zeros(2))


def cycle(background, streak_um_per_deg: float, seconds: float = 15) -> dict | None:
    """One cycle: step(), propose(), then, if armed, `apply? [y/N]`: y applies the proposal, anything else declines
    it; an answer after EXPIRE_S declines it as expired. Unarmed (shadow mode) nothing is asked or written. Returns
    the proposal, or None if the reading was invalid."""
    step(background, streak_um_per_deg, seconds)
    p = propose()
    if p is None or not p["armed"]:
        return p
    answer = input(f"apply proposal {p['proposal']}? [y/N] ").strip().lower()
    if time.time() - p["made"] > EXPIRE_S:
        print(f"answered after {EXPIRE_S:.0f} s: proposal {p['proposal']} expired, not applied")
        return _decide(p, "expired", np.zeros(2))
    return apply() if answer == "y" else decline()


def state() -> dict:
    """The running episode: goal, step, proposals made, the latest proposal's decision."""
    if _episode is None:
        return {"episode": None}
    return {"episode": _episode["name"], "log": str(_episode["log"]), "step": _episode["inputs"].steps,
            "proposals": _episode["n"], "latest": None if _proposal is None else
            {k: _proposal[k] for k in ("proposal", "change", "decision")}}


def _decide(p: dict, decision: str, got: np.ndarray) -> dict:
    """Record a decision on proposal `p` and its outcome for the episode's next step: the action and the change the
    beam got if applied, else 0 and 0. An answer in time (applied or declined) restarts the idle timer."""
    p["decision"], p["applied"] = decision, got.tolist()
    if _episode is not None:
        _episode["outcome"] = (np.array(p["action"]) if decision == "applied" else np.zeros(2), got)
        _log({"proposal": p["proposal"], "decision": decision, "applied": got.tolist()})
    if decision in ("applied", "declined"):
        actuate.attended()
    return p


def _undecided() -> dict:
    _running()
    if _proposal is None or _proposal["decision"] is not None:
        raise RuntimeError("no proposal to decide: propose() first")
    if _readings[-1]["reading"].path != _proposal["reading"]:
        raise RuntimeError(f"proposal {_proposal['proposal']} is from an older reading: propose() again")
    return _proposal


def _running() -> dict:
    if _episode is None or actuate.episode != _episode["name"]:
        raise RuntimeError("no episode running: use_goal() first")
    return _episode


def _latest() -> dict:
    """The latest reading, which must have had no write sent since it began."""
    if not _readings:
        raise RuntimeError("no reading yet: step() first")
    if _readings[-1]["writes"][0] != actuate.writes:
        raise RuntimeError("a write was sent during or after the latest reading: step() again")
    return _readings[-1]


def _day() -> Path:
    path = session.DATA / f"{datetime.date.today():%Y-%m-%d}_V2"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _log(entry: dict) -> dict:
    """Append `entry` to the episode's log, with the time; returns it."""
    entry = {**entry, "time": datetime.datetime.now().isoformat(timespec="milliseconds")}
    with open(_episode["log"], "a") as f:
        print(json.dumps(entry), file=f)
    return entry


def _line(p: dict, e: dict) -> str:
    v, g = p["values"], e["goal"]
    change = ", ".join(f"{k} {c:+.3f} deg" + (f" (clipped from {a:+.3f})" if abs(c - a) > 1e-9 else "")
                       for k, c, a in zip(pvs.WRITE, p["change"], p["asked"]))
    small = " (L2 under the minimum: not written)" if 0 < abs(p["change"][1]) < MIN_L2_MOVE else ""
    net = ", ".join(f"{k} {n:+.2f}/{pvs.WRITE[k]['budget']}" for k, n in zip(pvs.WRITE, p["net"]))
    return (f"proposal {p['proposal']} (episode step {p['episode_step']}, goal {e['name']}): "
            f"spacing {1e6 * v['bunch_spacing']:.1f} um (goal {1e6 * g['bunch_spacing']:.1f}), "
            f"drive sigma_z {1e6 * v['drive_sigma_z']:.1f} um (goal {1e6 * g['drive_sigma_z']:.1f}) | "
            f"{change}{small} | net {net}" + ("" if p["armed"] else " | shadow: not written"))
