"""P1b (v2): the policy's goals, episodes and proposals, and the approved cycle. Writes only through actuate's moves, so
arm(), the limits, the budget and the ledger apply as in W1. Each episode's proposals and decisions go in its log,
data/<date>_V2/episode_<HHMMSS_mmm>.jsonl. Goals are kept in data/goals.json until removed or replaced (R15).

A session (v2 plan, 11-06):
  pre_shift(entered)            the policy's envelope beside the live values and what the operator enters (R10, R16)
  step(background, streak)      readings, as session.step(), remembered here with the writes sent around them
  record_goal("A")              the mean of the last GOAL_READINGS valid readings, with no write between them
  actuate.arm("name"), actuate.move_l1() / move_l2()   detune, inside the budget
  use_goal("A")                 an episode on the latest reading, with no history, as training's reset
  remove_goal("A")              a goal that no longer applies
  cycle(background, streak)     step(), propose(), and when armed a y/N prompt: y applies, anything else declines
  run(background, streak, schedule)   closed loop (R8): cycles applied without asking until a stop
  end_episode(); actuate.undo_all() or keep; actuate.disarm()

Unarmed, cycle() is shadow mode: proposals are logged and nothing is written. The policy's inputs follow training's
bookkeeping (controller.py): a declined, expired, unanswered or shadow proposal is "asked for nothing, got nothing"
(action and observed change 0); an applied one is the policy's action and the change written (L1: the PREQ change;
L2: the multiknob delta, not written below MIN_L2_MOVE and then 0). A proposal is clipped to the session's budget, as
training's knob window clipped actions. An invalid reading gets no proposal and the episode doesn't advance (a hold).
A reading with the witness ahead of the drive (negative spacing) is held too: the policy was trained only with it
trailing (R4). After ALERT_HOLDS holds in a row with a beam-state fault (the beam itself unreadable, not the
acquisition) it alerts, and the operator decides; nothing is undone (R3). Goals outside the policy's trained ranges are
recorded with a warning (R5). Each proposal logs the time since the reading the episode last advanced on (R9). After
the policy's episode length the episode restarts on the same goal.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import actuate, controller, machine, pvs, session

POLICY = "nf_screen_full_s0"
GOAL_READINGS = 3       # record_goal() averages this many valid readings
EXPIRE_S = 60.0         # a proposal can be applied for this long after it is made
MIN_L2_MOVE = 0.02      # deg: a smaller L2 change isn't written (under the multiknob's readback tolerance it would
                        # look "not applied" and disarm)
KINDS = {"L1PhaseSet": "L1", "L2PhaseSet": "L2"}
ALERT_HOLDS = 3         # holds in a row with a beam-state fault before the alert (R3)
BEAM_STATE = ("step cut", "|spacing|", "edge fraction", "witness trailing")   # checks that say the beam itself is
                        # unreadable; the others (frames, row-correction reference, intensity) are acquisition faults

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


def pre_shift(entered: dict) -> dict:
    """R10, R16: the policy's envelope (the ranges it was trained on) beside the live values, for the before-shift
    checklist, saved as the session's record (data/<date>_V2/session_<HHMMSS>.json) and printed one line per item.
    Reads once: the TCAV amplitude, each magnet against TEST_15668's value (the 04-15 state the optics draws are about),
    2 s of camera frames for the beam rate, and the free disk under data/. `entered` is what the operator enters, saved
    as given; these keys are checked against the envelope when present: charge_nC, delay_ps, budget and step_scale
    ({"L1": .., "L2": ..}). Other keys (the TCAV calibration and its sign, the cut polarity, ops' pulse lengths, the
    feedback states) are recorded only. Nothing here refuses: the operator ticks the checklist."""
    import shutil

    envelope = policy().manifest.get("envelope", {})
    tcav = machine.describe([pvs.TCAV_AMPLITUDE])[pvs.TCAV_AMPLITUDE]
    magnets = machine.describe(list(pvs.MAGNETS))
    steps = session.pulse_id_steps([f.pulse_id for f in machine.frames(2)])
    rate = session.PULSES_PER_SECOND / max(steps, key=steps.get) if steps else None
    session.DATA.mkdir(parents=True, exist_ok=True)
    rows = [("TCAV amplitude, MV", envelope.get("tcav_MV"), None if tcav is None else tcav["value"]),
            ("beam rate, Hz (capture 15 s at 10 Hz, 30 s at 5 Hz)", None, rate),
            ("free disk under data/, GB (5-10 MB a step)", None, shutil.disk_usage(session.DATA).free / 1e9)]
    for name, m in magnets.items():
        label = pvs.MAGNETS[name].split(" supply")[0]
        frac = envelope.get("ff_quad_fraction", {}).get(label, envelope.get("sextupole_fraction")
                                                         if label.startswith("sextupole") else None)
        ratio = None if m is None else m["value"] / pvs.MAGNETS_IN_DAQ[name]
        rows.append((f"{label}, x TEST_15668", None if frac is None else [1 - frac, 1 + frac], ratio))
    for key, span in (("charge_nC", "charge_nC"), ("delay_ps", "delay_ps")):
        rows.append((f"{key} (entered)", envelope.get(span), entered.get(key)))
    for key, span in (("budget", "budget_deg"), ("step_scale", "step_scale")):
        for kind, control in KINDS.items():
            rows.append((f"{key} {control} (entered)", envelope.get(span, {}).get(kind),
                         entered.get(key, {}).get(control)))
    table = [{"item": item, "trained": span, "value": value,
              "inside": None if span is None or value is None else span[0] <= value <= span[1]}
             for item, span, value in rows]
    for row in table:
        flag = {True: "ok", False: "OUTSIDE", None: ""}[row["inside"]]
        print(f"{row['item']:<52} {str(row['trained']):<26} {row['value']!s:<24} {flag}")
    record = {"policy": POLICY, "envelope": envelope, "entered": entered, "table": table,
              "tcav": tcav, "magnets": magnets, "pulse_id_steps": steps}
    now = datetime.datetime.now()
    (_day() / f"session_{now:%H%M%S}.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    return record


def step(background, streak_um_per_deg: float, seconds: float = 15, cut_polarity: str = "+") -> session.Reading:
    """session.step(), remembered for record_goal() and propose() with when it began and the writes sent before and
    after it."""
    before, began = actuate.writes, time.time()
    reading = session.step(background, streak_um_per_deg, seconds, cut_polarity=cut_polarity)
    _readings.append({"reading": reading, "values": values(reading), "writes": (before, actuate.writes),
                      "began": began})
    return reading


def checks(reading: session.Reading) -> dict:
    """The reading's checks (session.step's), and the witness trailing (spacing > 0) when there is a reading (R4)."""
    r = reading.record
    return {**r["checks"], **({"witness trailing": r["reading"]["spacing_um"] > 0} if r["reading"] else {})}


def values(reading: session.Reading) -> dict | None:
    """A valid reading's features as the policy takes them (m); None if any check fails (checks())."""
    if not all(checks(reading).values()):
        return None
    read = reading.record["reading"]
    return {"bunch_spacing": 1e-6 * read["spacing_um"], "drive_sigma_z": 1e-6 * read["drive_sigma_z_um"],
            "drive_charge_fraction": read["drive_charge_fraction"], "screen_valid_deployed": 1.0}


def record_goal(name: str) -> dict:
    """Goal `name`: the mean of each goal feature over the last GOAL_READINGS valid readings, saved in the goal store
    (replacing a goal of that name). Refused if there are fewer, or if a write was sent between the first and the
    last."""
    valid = [r for r in _readings if r["values"] is not None][-GOAL_READINGS:]
    if len(valid) < GOAL_READINGS:
        raise RuntimeError(f"{len(valid)} valid readings so far; a goal needs {GOAL_READINGS}")
    if valid[0]["writes"][0] != valid[-1]["writes"][1]:
        raise RuntimeError(f"a write was sent between the last {GOAL_READINGS} valid readings: take more steps")
    goal = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
            "readings": [r["reading"].path for r in valid],
            **{g["key"]: float(np.mean([r["values"][g["key"]] for r in valid])) for g in policy().manifest["goals"]}}
    goal["outside_training"] = _outside(goal)
    for line in goal["outside_training"]:
        print(f"warning: goal {name}: {line}; recorded, the operator decides")
    _save_goals({**_goals(), name: goal})
    return goal


def remove_goal(name: str) -> None:
    """Remove goal `name` from the store. A running episode keeps its own copy."""
    goals = _goals()
    if name not in goals:
        raise RuntimeError(f"no goal {name!r} (goals: {sorted(goals)})")
    del goals[name]
    _save_goals(goals)


def use_goal(name: str) -> dict:
    """Start an episode on goal `name` (from the goal store) from the latest reading, which must be valid and taken
    with no write since: the policy's inputs with no history, as training's reset. Prints when the goal was recorded:
    whether an old goal still applies is the operator's call. Manual moves are refused until end_episode(); a running
    episode ends first. Returns the episode log's first line."""
    global _episode, _proposal
    goals = _goals()
    if name not in goals:
        raise RuntimeError(f"no goal {name!r} (goals: {sorted(goals)}): record_goal() first")
    goal = goals[name]
    latest = _latest()
    if latest["values"] is None:
        raise RuntimeError("the latest reading is invalid: an episode starts on a valid one")
    end_episode()
    now = datetime.datetime.now()
    scale = _limits()[3]
    _episode = {"name": name, "goal": goal, "inputs": controller.Episode(policy(), goal, latest["values"], scale),
                "reading": latest, "proposed": False, "outcome": None, "n": 0, "beam_holds": 0,
                "log": _day() / f"episode_{now:%H%M%S}_{now.microsecond // 1000:03d}.jsonl"}
    _proposal = None
    actuate.episode = name
    print(f"goal {name}, recorded {goal['time']}: spacing {1e6 * goal['bunch_spacing']:.1f} um, "
          f"drive sigma_z {1e6 * goal['drive_sigma_z']:.1f} um")
    files = [controller.POLICIES / POLICY / f for f in ("policy.json", "weights.npz")]
    return _log({"goal": name, "goal_record": goal, "step_scale": scale, "policy": POLICY,
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
    and change 0). The change is the action times the session's step, clipped to its budget (unarmed: nothing has been
    written, pvs.WRITE's budget). Refused if the session is armed at another step setting than the episode began
    with."""
    global _proposal
    e = _running()
    latest = _latest()
    if latest["values"] is None:
        return _hold(e, latest)
    e["beam_holds"] = 0
    period = None                       # since the reading the episode last advanced on (R9)
    if latest is e["reading"]:
        if e["proposed"]:
            raise RuntimeError("already proposed on this reading: step() first")
    else:
        period = latest["began"] - e["reading"]["began"]
        action, change = e["outcome"] or (np.zeros(2), np.zeros(2))
        e["inputs"].advance(action, change, latest["values"])
        e.update(reading=latest, outcome=None)
        if e["inputs"].over:
            scale = e["inputs"].step_scale if e["inputs"].scaled else None
            e["inputs"] = controller.Episode(policy(), e["goal"], latest["values"], scale)
            print(f"episode at {policy().manifest['episode_length']} steps: restarted on goal {e['name']}")
            _log({"restarted": True, "reading": latest["reading"].path})
    if e.get("next_goal"):                  # a schedule's next goal (run()), after the step's advance as in training
        name = e.pop("next_goal")
        e["name"], e["goal"] = name, _goals()[name]
        e["inputs"].set_goal(e["goal"], latest["values"])
        actuate.episode = name
        _log({"goal_change": name, "goal_record": e["goal"]})
        print(f"goal now {name}")
    armed, net, budget, scale = _limits()
    if armed and not np.allclose(scale or [1.0, 1.0], e["inputs"].step_scale):
        raise RuntimeError(f"the episode runs at step setting {e['inputs'].step_scale.tolist()}, the session is "
                           f"armed at {scale or [1.0, 1.0]}: use_goal() again")
    inputs = e["inputs"].inputs()
    action = policy().act(inputs)
    step = e["inputs"].step
    change = controller.clip(action, step, net, budget)
    e["n"], e["proposed"] = e["n"] + 1, True
    _proposal = _log({"proposal": e["n"], "made": time.time(), "reading": latest["reading"].path,
                      "episode_step": e["inputs"].steps, "armed": armed, "values": latest["values"],
                      "action": action.tolist(), "asked": (action * step).tolist(), "change": change.tolist(),
                      "net": net.tolist(), "budget": budget.tolist(), "step": step.tolist(), "period_s": period,
                      "inputs": inputs.tolist(), "decision": None})
    print(_line(_proposal, e))
    return _proposal


def apply(unattended: bool = False) -> dict:
    """Write the latest proposal, once: L1 (unless its change is 0), then L2 (unless under MIN_L2_MOVE). Refused if it
    has been decided, expired (EXPIRE_S) or a newer reading taken, or if not armed. A failed write ends the episode
    (actuate disarms). Ctrl-C meanwhile waits until the writes are made, checked and the decision recorded (R8)."""
    p = _undecided()
    if not actuate.state()["armed"]:
        raise RuntimeError("not armed: the proposal stays undecided (shadow)")
    if time.time() - p["made"] > EXPIRE_S:
        _decide(p, "expired", np.zeros(2))
        raise RuntimeError(f"proposal {p['proposal']} expired ({EXPIRE_S:.0f} s): not applied")
    got = np.zeros(2)
    with actuate.held_interrupt():
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
        _decide(p, "applied (unattended)" if unattended else "applied", got)
    return p


def decline() -> dict:
    """Decline the latest proposal: the next step's inputs say it asked for nothing and got nothing."""
    return _decide(_undecided(), "declined", np.zeros(2))


def cycle(background, streak_um_per_deg: float, seconds: float = 15, cut_polarity: str = "+") -> dict | None:
    """One cycle: step(), propose(), then, if armed, `apply? [y/N]`: y applies the proposal, anything else declines
    it; an answer after EXPIRE_S declines it as expired. Unarmed (shadow mode) nothing is asked or written. Returns
    the proposal, or None if the reading was invalid."""
    step(background, streak_um_per_deg, seconds, cut_polarity)
    p = propose()
    if p is None or not p["armed"]:
        return p
    answer = input(f"apply proposal {p['proposal']}? [y/N] ").strip().lower()
    if time.time() - p["made"] > EXPIRE_S:
        print(f"answered after {EXPIRE_S:.0f} s: proposal {p['proposal']} expired, not applied")
        return _decide(p, "expired", np.zeros(2))
    return apply() if answer == "y" else decline()


def run(background, streak_um_per_deg: float, schedule: list | None = None, seconds: float = 15,
        cut_polarity: str = "+") -> str:
    """Closed loop (R8): cycles (step(), propose(), apply()) without asking, until a stop: a hold (the R3 alert among
    them), a failed write, a proposal clipped to 0 on both controls (the budget used up), the end of the schedule, or
    Ctrl-C (a write in flight finishes first). Needs the running episode and an armed session. `schedule`: [(goal name,
    n), ...]: each goal for n applied proposals, then the next takes over mid-episode as training's goal changes did
    (history kept, the goal error restarting at the new goal); without one the episode's goal is kept. A stop ends the
    schedule; to go on, pass the rest. Prints and returns why it stopped, and logs it; the episode stays, approved."""
    e = _running()
    if not actuate.state()["armed"]:
        raise RuntimeError("not armed: run() writes (cycle() unarmed is the shadow mode)")
    plan = [(name, int(n)) for name, n in (schedule or [])]
    unknown = [name for name, _ in plan if name not in _goals()]
    if unknown or any(n < 1 for _, n in plan):
        raise ValueError(f"schedule {plan}: goals {unknown} unknown (goals: {sorted(_goals())}), or n < 1")
    _log({"run": "started", "schedule": plan})
    stop, current, left = None, None, None
    try:
        while stop is None:
            if plan and not left:
                current, left = plan.pop(0)
                if current != e["name"]:
                    e["next_goal"] = current            # switched by propose(), after the step's advance
            step(background, streak_um_per_deg, seconds, cut_polarity)
            p = propose()
            if p is None:
                stop = "the R3 alert" if e["beam_holds"] >= ALERT_HOLDS else "a hold"
            elif not any(p["change"]) and any(p["asked"]):
                stop = "the budget used up on both controls"
            else:
                left = None if left is None else left - 1      # counted first: an interrupt waits for the decision
                try:
                    apply(unattended=True)
                except Exception as err:
                    stop = f"a failed write ({err}); the episode ended"
                    continue
                if left == 0 and not plan:
                    stop = "the end of the schedule"
    except KeyboardInterrupt:
        stop = "Ctrl-C"
    rest = ([(current, left)] if left else []) + plan
    if _episode is not None:
        _episode.pop("next_goal", None)
        _log({"run": "stopped", "reason": stop, "schedule_left": rest})
    print(f"run stopped: {stop}" + (f"; schedule left: {rest}" if rest else ""))
    return stop


def state() -> dict:
    """The running episode: goal, step, proposals made, the latest proposal's decision."""
    if _episode is None:
        return {"episode": None}
    return {"episode": _episode["name"], "log": str(_episode["log"]), "step": _episode["inputs"].steps,
            "proposals": _episode["n"], "beam_holds": _episode["beam_holds"], "latest": None if _proposal is None else
            {k: _proposal[k] for k in ("proposal", "change", "decision")}}


def _hold(e: dict, latest: dict) -> None:
    """A held reading: no proposal, and the episode doesn't advance. Logged with its failed checks; one that says the
    beam itself is unreadable counts towards the alert (a valid reading resets the count), at ALERT_HOLDS (R3)."""
    failed = [check for check, ok in checks(latest["reading"]).items() if not ok]
    beam = [check for check in failed if check.startswith(BEAM_STATE)]
    e["beam_holds"] += bool(beam)
    _log({"held": latest["reading"].path, "failed": failed, "beam_state": bool(beam), "beam_holds": e["beam_holds"]})
    print(f"hold, {'beam state' if beam else 'acquisition'} ({'; '.join(failed)}): no proposal, the episode doesn't "
          "advance" + ("; take another step" if not beam else f" ({e['beam_holds']} in a row)"))
    if beam and e["beam_holds"] == ALERT_HOLDS:
        last, state = actuate.last_write(), actuate.state()
        written = "none" if last is None else f"{last['kind']} {last['delta']:+.3f} deg at {last['time']}"
        print(f"ALERT: {ALERT_HOLDS} readings in a row with the beam unreadable: operator decision needed (wait, "
              f"end_episode() and move by hand, or undo_all()). Last write: {written}; net move: {state.get('net')}. "
              "Nothing is proposed or written.")
        _log({"alert": failed, "last_write": last, "net": state.get("net")})
    return None


def _outside(goal: dict) -> list[str]:
    """Where the goal lies outside the policy's trained goal ranges (its envelope), and by how much (R5)."""
    envelope = policy().manifest.get("envelope", {})
    out = []
    for key, span in (("bunch_spacing", "spacing_goal_m"), ("drive_sigma_z", "drive_sigma_z_goal_m")):
        if span in envelope:
            lo, hi = envelope[span]
            if not lo <= goal[key] <= hi:
                off = lo - goal[key] if goal[key] < lo else goal[key] - hi
                out.append(f"{key} {1e6 * goal[key]:.1f} um is outside the trained {1e6 * lo:.1f}-{1e6 * hi:.1f} um, "
                           f"by {1e6 * off:.1f} um")
    return out


def _decide(p: dict, decision: str, got: np.ndarray) -> dict:
    """Record a decision on proposal `p` and its outcome for the episode's next step: the action and the change the
    beam got if applied, else 0 and 0. An answer in time (applied or declined) restarts the idle timer."""
    p["decision"], p["applied"] = decision, got.tolist()
    if _episode is not None:
        _episode["outcome"] = (np.array(p["action"]) if decision.startswith("applied") else np.zeros(2), got)
        _log({"proposal": p["proposal"], "decision": decision, "applied": got.tolist()})
    if decision.startswith("applied") or decision == "declined":
        actuate.attended()
    return p


def _undecided() -> dict:
    _running()
    if _proposal is None or _proposal["decision"] is not None:
        raise RuntimeError("no proposal to decide: propose() first")
    if _readings[-1]["reading"].path != _proposal["reading"]:
        raise RuntimeError(f"proposal {_proposal['proposal']} is from an older reading: propose() again")
    return _proposal


def _limits() -> tuple[bool, np.ndarray, np.ndarray, list | None]:
    """Armed or not, and per control the net move, the budget and the step setting (None when 1 for both). Unarmed:
    nothing written, pvs.WRITE's budget, setting 1."""
    st = actuate.state()
    if not st["armed"]:
        return False, np.zeros(2), np.array([pvs.WRITE[k]["budget"] for k in pvs.WRITE]), None
    scale = [st["step_scale"][k] for k in pvs.WRITE]
    return (True, np.array([st["net"][k] for k in pvs.WRITE]), np.array([st["write"][k]["budget"] for k in pvs.WRITE]),
            None if scale == [1.0, 1.0] else scale)


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


def _goals() -> dict:
    path = session.DATA / "goals.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _save_goals(goals: dict) -> None:
    session.DATA.mkdir(parents=True, exist_ok=True)
    (session.DATA / "goals.json").write_text(json.dumps(goals, indent=1) + "\n")


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
    net = ", ".join(f"{k} {n:+.2f}/{b}" for k, n, b in zip(pvs.WRITE, p["net"], p["budget"]))
    longest = policy().manifest.get("envelope", {}).get("step_period_s", [None, float("inf")])[1]
    period = "" if p["period_s"] is None else f" | period {p['period_s']:.0f} s" + (
        f" (over the trained {longest:.0f} s)" if p["period_s"] > longest else "")
    return (f"proposal {p['proposal']} (episode step {p['episode_step']}, goal {e['name']}): "
            f"spacing {1e6 * v['bunch_spacing']:.1f} um (goal {1e6 * g['bunch_spacing']:.1f}), "
            f"drive sigma_z {1e6 * v['drive_sigma_z']:.1f} um (goal {1e6 * g['drive_sigma_z']:.1f}) | "
            f"{change}{small} | net {net}{period}" + ("" if p["armed"] else " | shadow: not written"))
