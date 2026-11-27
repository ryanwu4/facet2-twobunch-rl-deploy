"""P1b: advise.py. Goals, episodes and proposals with stand-in readings (session.step replaced), checked against
controller.py's bookkeeping; apply() and cycle() through actuate on the soft IOC and the stand-in multiknob service;
and a proposal from a real step() on the soft IOC replaying a two-bunch DAQ step."""
import itertools
import json
import os
import signal
import time

import numpy as np
import pytest

from conftest import daq, needs_daq, pil_crop
from test_actuate import armed, fresh, lines  # noqa: F401 (fixtures)
from twobunch_deploy import actuate, advise, controller, machine, pvs, screen, session

COUNT = itertools.count()
ZERO = np.zeros(2)


@pytest.fixture(autouse=True)
def clean(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "DATA", tmp_path)
    for name, value in (("_readings", []), ("_episode", None), ("_proposal", None)):
        monkeypatch.setattr(advise, name, value)
    monkeypatch.setattr(actuate, "episode", None)
    monkeypatch.setattr(actuate, "writes", 0)
    yield
    advise.end_episode()


def fake(spacing_um=150.0, sigma_z_um=40.0, valid=True, failed="|spacing| > 60 um"):
    """A stand-in step() reading; if not valid, the check `failed` fails (a beam-state one by default)."""
    checks = {"|spacing| > 60 um": True, "+ frames >= 15": True}
    checks[failed] = valid
    record = {"valid": valid, "checks": checks,
              "reading": {"spacing_um": spacing_um, "drive_sigma_z_um": sigma_z_um, "drive_charge_fraction": 0.7}}
    return session.Reading(record, f"step_{next(COUNT)}.json", {}, [])


def feed(monkeypatch, *readings):
    """session.step returns these, in order."""
    queue = list(readings)
    monkeypatch.setattr(session, "step", lambda *args, **kw: queue.pop(0))


def start(monkeypatch, *later):
    """Three valid readings, goal A from them, an episode on the third; then `later` for the following step()s."""
    feed(monkeypatch, fake(150.0), fake(152.0), fake(154.0), *later)
    for _ in range(3):
        advise.step(None, 1100.0)
    goal = advise.record_goal("A")
    advise.use_goal("A")
    return goal


def test_goal_is_the_mean_of_three_valid_readings(monkeypatch):
    feed(monkeypatch, fake(150.0, 40.0), fake(10.0, valid=False), fake(152.0, 42.0), fake(154.0, 44.0))
    for _ in range(3):
        advise.step(None, 1100.0)
    with pytest.raises(RuntimeError, match="needs 3"):
        advise.record_goal("A")
    advise.step(None, 1100.0)
    goal = advise.record_goal("A")
    assert goal["bunch_spacing"] == pytest.approx(152e-6) and goal["drive_sigma_z"] == pytest.approx(42e-6)
    assert goal["readings"] == [r["reading"].path for r in advise._readings if r["values"]]
    assert json.loads((session.DATA / "goals.json").read_text()) == {"A": goal}


def test_goals_kept_until_removed_or_replaced(monkeypatch, tmp_path):
    """The goal store outlives the day: a goal recorded today is used tomorrow (its episode log in tomorrow's folder);
    recording under its name replaces it; once removed, use_goal() refuses with the goals left."""
    goal = start(monkeypatch, fake(160.0))
    monkeypatch.setattr(advise, "_day", lambda: tmp_path)                   # the next day's folder
    advise.step(None, 1100.0)
    advise.use_goal("A")
    assert advise._episode["goal"] == goal and advise._episode["log"].parent == tmp_path
    replaced = advise.record_goal("A")                                       # the last three: 152, 154, 160 um
    assert replaced["bunch_spacing"] == pytest.approx(155.333e-6, abs=1e-9) and list(advise._goals()) == ["A"]
    advise.remove_goal("A")
    assert advise._episode["goal"] == goal                                   # the running episode keeps its copy
    with pytest.raises(RuntimeError, match="no goal 'A' \\(goals: \\[\\]\\)"):
        advise.use_goal("A")


def test_goal_refused_with_a_write_between(monkeypatch):
    feed(monkeypatch, fake(), fake(), fake(), fake(), fake(), fake())
    advise.step(None, 1100.0)
    monkeypatch.setattr(actuate, "writes", 1)            # a write between the first and the second
    advise.step(None, 1100.0)
    advise.step(None, 1100.0)
    with pytest.raises(RuntimeError, match="write was sent"):
        advise.record_goal("A")
    advise.step(None, 1100.0)                            # three valid readings after it
    advise.record_goal("A")


def test_shadow_episode_follows_training_bookkeeping(monkeypatch):
    """Unarmed: proposals are made and logged, never decided, so each next step is "asked for nothing, got nothing"; an
    invalid reading holds; one proposal per reading."""
    goal = start(monkeypatch, fake(10.0, valid=False), fake(149.0, 41.0), fake(147.0, 43.0))
    policy = advise.policy()
    reference = controller.Episode(policy, goal, advise._readings[2]["values"])
    first = advise.propose()
    np.testing.assert_array_equal(first["inputs"], reference.inputs())
    np.testing.assert_array_equal(first["action"], policy.act(reference.inputs()))
    assert not first["armed"] and first["net"] == [0.0, 0.0]
    with pytest.raises(RuntimeError, match="already proposed"):
        advise.propose()
    advise.step(None, 1100.0)
    assert advise.propose() is None and advise.state()["step"] == 0              # held
    for _ in range(2):
        advise.step(None, 1100.0)
        p = advise.propose()
        reference.advance(ZERO, ZERO, advise._readings[-1]["values"])
        np.testing.assert_array_equal(p["inputs"], reference.inputs())
    assert p["episode_step"] == 2 and p["proposal"] == 3
    log = [json.loads(line) for line in advise._episode["log"].read_text().splitlines()]
    assert log[0]["goal"] == "A" and log[0]["policy"] == advise.POLICY and len(log) == 5        # 3 proposals, 1 hold
    assert log[2]["failed"] == ["|spacing| > 60 um"] and log[2]["beam_state"]


def test_reversed_bunches_are_held(monkeypatch, capsys):
    """A valid reading with negative spacing (the witness ahead, or a wrong streak sign) is held as a beam-state fault
    and doesn't count for a goal (R4)."""
    start(monkeypatch, fake(-150.0))
    advise.step(None, 1100.0)
    assert advise.checks(advise._readings[-1]["reading"])["witness trailing"] is False
    assert advise.values(advise._readings[-1]["reading"]) is None
    assert advise.propose() is None and "beam state (witness trailing)" in capsys.readouterr().out
    assert advise.record_goal("B")["bunch_spacing"] == pytest.approx(152e-6)            # the three before it


def test_alert_after_beam_state_holds(monkeypatch, capsys):
    """Holds with a beam-state fault count towards the alert; acquisition faults neither count nor reset it; a valid
    reading resets it. The alert prints once, at ALERT_HOLDS, and is logged with the net move (R3)."""
    start(monkeypatch, fake(valid=False), fake(valid=False, failed="+ frames >= 15"), fake(valid=False),
          fake(valid=False), fake(valid=False), fake(), fake(valid=False))
    advise.propose()
    counts = []
    for _ in range(5):
        advise.step(None, 1100.0)
        assert advise.propose() is None
        counts.append(advise.state()["beam_holds"])
    assert counts == [1, 1, 2, 3, 4] and capsys.readouterr().out.count("ALERT") == 1
    advise.step(None, 1100.0)
    assert advise.propose() is not None and advise.state()["beam_holds"] == 0
    advise.step(None, 1100.0)
    advise.propose()
    assert advise.state()["beam_holds"] == 1
    log = [json.loads(line) for line in advise._episode["log"].read_text().splitlines()]
    assert sum("alert" in e for e in log) == 1 and sum("held" in e for e in log) == 6


def test_goal_outside_training_is_recorded_with_a_warning(monkeypatch, capsys):
    feed(monkeypatch, fake(80.0, 100.0), fake(80.0, 100.0), fake(80.0, 100.0))
    for _ in range(3):
        advise.step(None, 1100.0)
    goal = advise.record_goal("A")
    assert goal["outside_training"] == ["bunch_spacing 80.0 um is outside the trained 100.0-400.0 um, by 20.0 um"]
    assert "warning: goal A" in capsys.readouterr().out and advise._goals()["A"] == goal


def test_period_is_logged_and_flagged(monkeypatch, capsys):
    """Each proposal logs the time since the reading the episode last advanced on, flagged over the trained 45 s (R9)."""
    start(monkeypatch, fake())
    first = advise.propose()
    advise.step(None, 1100.0)
    advise._readings[-1]["began"] = advise._readings[2]["began"] + 50.0
    p = advise.propose()
    assert first["period_s"] is None and p["period_s"] == pytest.approx(50.0)
    assert "period 50 s (over the trained 45 s)" in capsys.readouterr().out


def test_episode_restarts_after_its_length(monkeypatch):
    start(monkeypatch, fake(149.0), fake(148.0))
    monkeypatch.setitem(advise.policy().manifest, "episode_length", 1)
    advise.propose()
    advise.step(None, 1100.0)
    restarted = advise.propose()                          # advanced to step 1 = the length: a new episode
    reference = controller.Episode(advise.policy(), advise._episode["goal"], advise._readings[-1]["values"])
    np.testing.assert_array_equal(restarted["inputs"], reference.inputs())
    assert restarted["episode_step"] == 0
    assert any(json.loads(line).get("restarted") for line in advise._episode["log"].read_text().splitlines())


def test_manual_moves_refused_during_an_episode(monkeypatch):
    start(monkeypatch)
    for move in (actuate.move_l1, actuate.move_l2):
        with pytest.raises(RuntimeError, match="episode on goal A"):
            move(0.1)
    advise.end_episode()
    with pytest.raises(RuntimeError, match="not armed"):
        actuate.move_l1(0.1)


def test_reading_must_follow_the_last_write(monkeypatch):
    start(monkeypatch, fake())
    advise.propose()
    monkeypatch.setattr(actuate, "writes", 1)
    with pytest.raises(RuntimeError, match="step\\(\\) again"):
        advise.propose()
    advise.step(None, 1100.0)
    assert advise.propose()["proposal"] == 2


def test_cycle_unarmed_asks_nothing(monkeypatch):
    start(monkeypatch, fake())
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("asked while unarmed"))
    p = advise.cycle(None, 1100.0)
    assert p["proposal"] == 1 and p["decision"] is None
    with pytest.raises(RuntimeError, match="not armed"):
        advise.apply()
    assert advise.state()["episode"] == "A" and p["decision"] is None
    with pytest.raises(RuntimeError, match="no goal 'B'"):
        advise.use_goal("B")
    first = advise._episode["log"]
    advise.use_goal("A")                                  # a new episode ends the running one
    assert json.loads(first.read_text().splitlines()[-1])["ended"]


@needs_daq
def test_apply_writes_through_actuate(armed, monkeypatch):     # noqa: F811
    initial, ledger, _ = armed
    start(monkeypatch, fake(149.0), fake(148.0))
    monkeypatch.setattr(advise.policy(), "act", lambda inputs: np.array([0.4, -0.5]))
    p = advise.propose()
    assert p["armed"] and p["change"] == pytest.approx([0.1, -0.15])
    applied = advise.apply()
    assert applied["decision"] == "applied" and applied["applied"] == pytest.approx([0.1, -0.15])
    done = [e for e in lines(ledger)[1:] if e["state"] == "done"]
    assert [(e["kind"], e["delta"], e["proposal"]) for e in done] == [("L1", 0.1, 1), ("L2", pytest.approx(-0.15), 1)]
    assert machine.get(pvs.L1_SETPOINTS[0]) == pytest.approx(initial[pvs.L1_SETPOINTS[0]] + 0.1)
    with pytest.raises(RuntimeError, match="no proposal to decide"):
        advise.apply()
    advise.step(None, 1100.0)
    second = advise.propose()                             # the episode's next step: the applied action and change
    reference = controller.Episode(advise.policy(), advise._episode["goal"], advise._readings[2]["values"])
    reference.advance(np.array([0.4, -0.5]), np.array([0.1, -0.15]), advise._readings[-1]["values"])
    np.testing.assert_allclose(second["inputs"], reference.inputs(), atol=1e-12)
    with pytest.raises(RuntimeError, match="episode on goal A"):
        actuate.move_l1(0.1)
    actuate.undo_all()                                    # ends the episode
    with pytest.raises(RuntimeError, match="no episode running"):
        advise.propose()


@needs_daq
def test_proposals_use_the_session_settings(fresh, monkeypatch):     # noqa: F811
    """Armed with a step setting and a budget: the episode's inputs end with the setting, the change is the action times
    the session's step, clipped to its budget. Armed again at another setting, the episode refuses to propose."""
    actuate.arm("test", budget={"L1": 0.2}, step_scale={"L1": 0.5, "L2": 0.5})
    start(monkeypatch, fake(), fake(), fake())
    monkeypatch.setattr(advise.policy(), "act", lambda inputs: np.array([1.0, -1.0]))
    p = advise.propose()
    assert p["step"] == pytest.approx([0.125, 0.15]) and p["change"] == pytest.approx([0.125, -0.15])
    assert p["budget"] == [0.2, 1.5] and p["inputs"][-2:] == [0.5, 0.5]
    advise.apply()
    advise.step(None, 1100.0)
    assert advise.propose()["change"] == pytest.approx([0.075, -0.15])         # 0.2 - 0.125 left on L1
    actuate.disarm()
    actuate.arm("test")
    advise.step(None, 1100.0)
    with pytest.raises(RuntimeError, match="use_goal"):
        advise.propose()


@needs_daq
def test_small_l2_changes_and_the_budget(armed, monkeypatch):     # noqa: F811
    """L1 at full scale reaches the budget after 4 writes, then is clipped to 0 and not written; an L2 change of
    0.015 deg is never written. Both count as what the beam got."""
    _, ledger, _ = armed
    start(monkeypatch, *[fake() for _ in range(4)])
    monkeypatch.setattr(advise.policy(), "act", lambda inputs: np.array([1.0, 0.05]))
    applied = []
    for i in range(5):
        if i:
            advise.step(None, 1100.0)
        advise.propose()
        applied.append(advise.apply()["applied"])
    assert applied == [[0.25, 0.0]] * 4 + [[0.0, 0.0]]
    assert [e["kind"] for e in lines(ledger)[1:] if e["state"] == "done"] == ["L1"] * 4
    assert advise._episode["outcome"][0].tolist() == [1.0, 0.05]          # the action as asked


@needs_daq
def test_cycle_asks_and_decides(armed, monkeypatch):     # noqa: F811
    _, ledger, _ = armed
    start(monkeypatch, fake(), fake(), fake())
    monkeypatch.setattr(advise.policy(), "act", lambda inputs: np.array([0.4, 0.0]))
    answers = iter(["y", "", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert advise.cycle(None, 1100.0)["decision"] == "applied"
    before = time.time()
    assert advise.cycle(None, 1100.0)["decision"] == "declined"
    assert actuate._session["last"] >= before            # an answer restarts the idle timer, as a write does
    last = actuate._session["last"]
    monkeypatch.setattr(advise, "EXPIRE_S", 0.0)
    assert advise.cycle(None, 1100.0)["decision"] == "expired"
    assert actuate._session["last"] == last              # a late answer doesn't
    reference = controller.Episode(advise.policy(), advise._episode["goal"], advise._readings[2]["values"])
    for action, change, r in ((ZERO, ZERO, 3), (np.array([0.4, 0.0]), np.array([0.1, 0.0]), 4), (ZERO, ZERO, 5)):
        reference.advance(action, change, advise._readings[r]["values"])     # applied, then declined: 0 and 0
    np.testing.assert_allclose(advise._proposal["inputs"], reference.inputs(), atol=1e-12)
    assert [e["kind"] for e in lines(ledger)[1:] if e["state"] == "done"] == ["L1"]
    decisions = [json.loads(line) for line in advise._episode["log"].read_text().splitlines() if "decision" in line]
    assert [d["decision"] for d in decisions if d["decision"]] == ["applied", "declined", "expired"]


@needs_daq
def test_failed_write_ends_the_episode(armed, monkeypatch):     # noqa: F811
    start(monkeypatch)
    monkeypatch.setattr(advise.policy(), "act", lambda inputs: np.array([0.0, 1.0]))
    monkeypatch.setattr(pvs, "MULTIKNOB", "nothing")
    monkeypatch.setattr(actuate, "KNOB_SETTLE_S", 0.5)
    advise.propose()
    path = advise._episode["log"]
    with pytest.raises(RuntimeError, match="not applied"):
        advise.apply()
    assert actuate.episode is None and not actuate.state()["armed"]
    log = [json.loads(line) for line in path.read_text().splitlines()]
    assert log[-2]["decision"] == "failed" and log[-1]["ended"]


@needs_daq
def test_propose_from_a_real_step(replay, monkeypatch):
    """advise.step() on the soft IOC serving a whole two-bunch DAQ step: a valid reading, and a proposal the policy
    makes from it. The DAQ step has fewer + frames than a live step's MIN_FRAMES, so that check is lowered."""
    monkeypatch.setattr(screen, "MIN_FRAMES", 5)
    data, _ = daq(0, None)
    background = session.Background(screen.pedestal(np.stack([pil_crop(frame) for frame in data])), "pedestal", {})
    replay("step")
    reading = advise.step(background, 1100.0, seconds=19)
    assert reading.record["valid"], reading.record["checks"]
    values = advise._readings[-1]["values"]
    assert values["bunch_spacing"] == pytest.approx(1e-6 * reading.record["reading"]["spacing_um"])
    goal = {"bunch_spacing": values["bunch_spacing"] + 10e-6, "drive_sigma_z": values["drive_sigma_z"]}
    (session.DATA / "goals.json").write_text(json.dumps({"A": {"time": "now", **goal}}))
    advise.use_goal("A")
    p = advise.propose()
    expected = advise.policy().act(controller.Episode(advise.policy(), goal, values).inputs())
    np.testing.assert_array_equal(p["action"], expected)


def test_set_goal_keeps_the_history():
    """controller.Episode.set_goal(): a goal change mid-episode as training's, history and cumulative change kept, the
    goal error against the new goal with trend 0."""
    p = advise.policy()
    r = {"bunch_spacing": 150e-6, "drive_sigma_z": 90e-6}
    e = controller.Episode(p, {"bunch_spacing": 160e-6, "drive_sigma_z": 90e-6}, r)
    e.advance(np.array([0.5, -0.5]), np.array([0.1, -0.15]), {"bunch_spacing": 155e-6, "drive_sigma_z": 95e-6})
    history, change = e.history.copy(), e.change.copy()
    e.set_goal({"bunch_spacing": 200e-6, "drive_sigma_z": 100e-6}, {"bunch_spacing": 155e-6, "drive_sigma_z": 95e-6})
    assert np.array_equal(e.history, history) and np.array_equal(e.change, change)
    assert e.error[0] == pytest.approx((155e-6 - 200e-6) / 1e-5) and np.array_equal(e.error, e.error_before)


def test_run_needs_an_armed_session(monkeypatch):
    start(monkeypatch)
    with pytest.raises(RuntimeError, match="not armed"):
        advise.run(None, 1100.0)


@needs_daq
def test_run_until_the_budget_is_used_up(armed, monkeypatch):     # noqa: F811
    """Unattended at full scale: L1 reaches its budget after 4 writes and L2 after 5; the next proposal is clipped to 0
    on both and the run stops there, leaving it undecided."""
    _, ledger, _ = armed
    start(monkeypatch, *[fake() for _ in range(6)])
    monkeypatch.setattr(advise.policy(), "act", lambda inputs: np.array([1.0, 1.0]))
    assert advise.run(None, 1100.0) == "the budget used up on both controls"
    done = [e for e in lines(ledger)[1:] if e["state"] == "done"]
    assert [e["kind"] for e in done] == ["L1", "L2"] * 4 + ["L2"]
    log = [json.loads(line) for line in advise._episode["log"].read_text().splitlines()]
    assert [d["decision"] for d in log if "applied" in d] == ["applied (unattended)"] * 5          # decision lines
    assert advise._proposal["decision"] is None and log[-1]["reason"] == "the budget used up on both controls"


@needs_daq
def test_run_with_a_goal_schedule(armed, monkeypatch):     # noqa: F811
    """A schedule [(A, 2), (B, 2)]: two applied proposals on A, then B takes over mid-episode (after that step's
    advance, as training's goal changes), two more, and the run stops at the end of the schedule."""
    start(monkeypatch, fake(180.0), fake(182.0), fake(184.0), *[fake(170.0) for _ in range(4)])
    for _ in range(3):
        advise.step(None, 1100.0)
    advise.record_goal("B")
    monkeypatch.setattr(advise.policy(), "act", lambda inputs: np.array([0.2, 0.2]))
    reference = controller.Episode(advise.policy(), advise._goals()["A"], advise._readings[2]["values"])
    assert advise.run(None, 1100.0, schedule=[("A", 2), ("B", 2)]) == "the end of the schedule"
    action, change = np.array([0.2, 0.2]), np.array([0.05, 0.06])
    reference.advance(ZERO, ZERO, advise._readings[6]["values"])
    reference.advance(action, change, advise._readings[7]["values"])
    reference.advance(action, change, advise._readings[8]["values"])
    reference.set_goal(advise._goals()["B"], advise._readings[8]["values"])
    third = [json.loads(line) for line in advise._episode["log"].read_text().splitlines() if '"proposal": 3,' in line][0]
    np.testing.assert_allclose(third["inputs"], reference.inputs(), atol=1e-12)
    assert advise.state()["episode"] == "B" == actuate.episode and advise._episode["n"] == 4


@needs_daq
def test_run_stops_on_a_hold_and_reports_the_schedule_left(armed, monkeypatch, capsys):     # noqa: F811
    start(monkeypatch, fake(), fake(valid=False, failed="+ frames >= 15"))
    assert advise.run(None, 1100.0, schedule=[("A", 3)]) == "a hold"
    assert "schedule left: [('A', 2)]" in capsys.readouterr().out
    assert "next_goal" not in advise._episode


@needs_daq
def test_ctrl_c_during_a_write_waits_for_it(armed, monkeypatch):     # noqa: F811
    """Ctrl-C (SIGINT) while a PREQ write is in flight: the write finishes, is checked and logged, the proposal is
    decided as applied, and then the run stops; no write is left pending."""
    _, ledger, _ = armed
    start(monkeypatch, fake(), fake())
    monkeypatch.setattr(advise.policy(), "act", lambda inputs: np.array([0.4, 0.4]))
    put = actuate.put_preq

    def interrupted(target):
        os.kill(os.getpid(), signal.SIGINT)
        put(target)
    monkeypatch.setattr(actuate, "put_preq", interrupted)
    assert advise.run(None, 1100.0) == "Ctrl-C"
    entries = lines(ledger)[1:]
    assert [(e["kind"], e["state"]) for e in entries] == [("L1", "pending"), ("L1", "done"), ("L2", "pending"),
                                                          ("L2", "done")]
    assert advise._proposal["decision"] == "applied (unattended)" and actuate.state()["pending"] == []
