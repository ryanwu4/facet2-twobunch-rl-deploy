"""P1b: advise.py. Goals, episodes and proposals with stand-in readings (session.step replaced), checked against
controller.py's bookkeeping; apply() and cycle() through actuate on the soft IOC and the stand-in multiknob service;
and a proposal from a real step() on the soft IOC replaying a two-bunch DAQ step."""
import itertools
import json
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


def fake(spacing_um=150.0, sigma_z_um=40.0, valid=True):
    """A stand-in step() reading."""
    record = {"valid": valid, "checks": {"|spacing| > 60 um": valid, "+ frames >= 15": True},
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
    assert json.loads((advise._day() / "goals.json").read_text()) == {"A": goal}


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
    assert log[0]["goal"] == "A" and log[0]["policy"] == advise.POLICY and len(log) == 4


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
    (advise._day() / "goals.json").write_text(json.dumps({"A": goal}))
    advise.use_goal("A")
    p = advise.propose()
    expected = advise.policy().act(controller.Episode(advise.policy(), goal, values).inputs())
    np.testing.assert_array_equal(p["action"], expected)
