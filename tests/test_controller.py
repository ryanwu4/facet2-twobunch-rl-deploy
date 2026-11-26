"""P1: controller.py against the reference data twobunch-s2e-rl's deploy_export wrote with each policy: the numpy runner
against the torch policy, and the inputs against the training env's own, step by step."""
import json
import shutil

import numpy as np
import pytest

from twobunch_deploy import controller

NAME = "nc_zs_screen_noblen_jit_s0"


@pytest.fixture(scope="module")
def policy():
    return controller.Policy(NAME)


def test_runner_reproduces_the_torch_policy(policy):
    with np.load(controller.POLICIES / NAME / "reference.npz") as ref:
        actions = np.array([policy.act(x.astype(float)) for x in ref["inputs"]])
        np.testing.assert_allclose(actions, ref["actions"], atol=1e-5)
    assert np.abs(actions).max() > 0.9                  # the reference reaches the saturated range


def test_inputs_reproduce_the_training_env(policy):
    """Episodes of the training env with the policy's actions, declined ones (0) and a run at full scale that the budget
    clips: the change of each action and the inputs after it."""
    m = policy.manifest
    with np.load(controller.POLICIES / NAME / "episodes.npz") as e:
        episodes = [{k: e[f"{k}_{i}"] for k in ("goals", "readings", "actions", "changes", "inputs", "budget",
                                                 "step_scale") if f"{k}_{i}" in e.files}
                    for i in range(sum(f.startswith("goals_") for f in e.files))]
    for e in episodes:
        readings = [dict(zip(controller.READING, r)) for r in e["readings"]]
        goal = {g["key"]: v for g, v in zip(m["goals"], e["goals"])}
        episode = controller.Episode(policy, goal, readings[0], step_scale=e.get("step_scale"))
        # the env works in float32: a sigma_z goal error (log10, / 0.0414) is resolved to about 1e-5
        np.testing.assert_allclose(episode.inputs(), e["inputs"][0], atol=1e-4)
        for t, action in enumerate(e["actions"]):
            change = controller.clip(action, episode.step, episode.change, e["budget"])
            # the env clips against float32 knob positions of 20-40 deg, resolved to about 4e-6 deg
            np.testing.assert_allclose(change, e["changes"][t], atol=1e-5)
            episode.advance(action, change, readings[t + 1])
            np.testing.assert_allclose(episode.inputs(), e["inputs"][t + 1], atol=1e-4)
        assert (np.abs(e["changes"] - e["actions"] * episode.step) > 1e-6).any()         # some were clipped
        assert (np.abs(e["actions"]).sum(axis=1) == 0).any()                              # some were declined


def test_episode_ends_at_the_training_length(policy):
    reading = {"bunch_spacing": 150e-6, "drive_sigma_z": 40e-6, "drive_charge_fraction": 0.7,
               "screen_valid_deployed": 1.0}
    episode = controller.Episode(policy, {"bunch_spacing": 160e-6, "drive_sigma_z": 40e-6}, reading)
    for _ in range(policy.manifest["episode_length"]):
        assert not episode.over
        episode.advance(np.zeros(2), np.zeros(2), reading)
    assert episode.over


def test_refuses_inputs_it_doesnt_build(tmp_path, monkeypatch):
    shutil.copytree(controller.POLICIES / NAME, tmp_path / NAME)
    path = tmp_path / NAME / "policy.json"
    manifest = json.loads(path.read_text())
    path.write_text(json.dumps({**manifest, "actuators": ["blen_dlog", "direct"]}))
    monkeypatch.setattr(controller, "POLICIES", tmp_path)
    with pytest.raises(ValueError, match="doesn't build"):
        controller.Policy(NAME)


def test_a_step_setting_policy_takes_its_scale_as_the_last_input(policy, tmp_path, monkeypatch):
    """A policy trained with the session step setting (its last block step_scale): the scale ends its inputs (1 by
    default), and the session's step is action_step times it. A policy without the block refuses a scale."""
    m = json.loads(json.dumps(policy.manifest))
    m["blocks"].append({"name": "step_scale", "size": 2})
    m["inputs"] += 2
    (tmp_path / "scaled").mkdir()
    (tmp_path / "scaled" / "policy.json").write_text(json.dumps(m))
    shutil.copy(controller.POLICIES / NAME / "weights.npz", tmp_path / "scaled" / "weights.npz")
    monkeypatch.setattr(controller, "POLICIES", tmp_path)
    scaled = controller.Policy("scaled")
    reading = {"bunch_spacing": 150e-6, "drive_sigma_z": 40e-6, "drive_charge_fraction": 0.7,
               "screen_valid_deployed": 1.0}
    goal = {"bunch_spacing": 160e-6, "drive_sigma_z": 40e-6}
    plain = controller.Episode(policy, goal, reading).inputs()
    np.testing.assert_allclose(controller.Episode(scaled, goal, reading).inputs(), np.r_[plain, 1.0, 1.0])
    e = controller.Episode(scaled, goal, reading, step_scale=[0.5, 1.5])
    np.testing.assert_allclose(e.inputs()[-2:], [0.5, 1.5])
    np.testing.assert_allclose(e.step, np.asarray(m["action_step"]) * [0.5, 1.5])
    with pytest.raises(ValueError):
        controller.Episode(policy, goal, reading, step_scale=[0.5, 0.5])
