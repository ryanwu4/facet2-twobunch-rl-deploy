"""P1: the policy and its inputs, built as its training env built them (twobunch-s2e-rl, DeltaControlEnvV4). numpy
only; no PV access.

A policy is exported by twobunch-s2e-rl's deploy_export to policies/<name>/: policy.json (the manifest: the input
layout, its normalisers and the action steps), weights.npz, and the reference data the tests check this module
against.

The inputs, in the manifest's order (106 for the incumbent):
- cumulative change: each control's observed change since the episode began (deg);
- features: the reading's and the goals', each transformed and z-scored as in training;
- goal error: per goal, (t(reading) - t(goal)) / scale, t the goal's transform;
- error trend: the goal error's change over the last step (0 at the start);
- last action: the policy's last output (0 at the start, and after a declined proposal);
- gain estimate: 0 (only an adapter actuator, which this module refuses, fills it);
- history: the last (history - 1) steps, oldest first, zeros before the episode began. Each step adds (goal error before
  the step, the action before the one applied, the step's observed change): training's pairing, kept.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

POLICIES = Path(__file__).resolve().parents[1] / "policies"
BLOCKS = ["cumulative_change", "features", "goal_error", "error_trend", "last_action", "gain_estimate", "history"]
READING = ("bunch_spacing", "drive_sigma_z", "drive_charge_fraction", "screen_valid_deployed")


def transform(value: float, name: str) -> float:
    return float(np.log10(max(abs(value), 1e-30))) if name == "log10" else float(value)


class Policy:
    """An exported policy: its manifest and the network as a numpy forward pass."""

    def __init__(self, name: str):
        self.name = name
        self.manifest = json.loads((POLICIES / name / "policy.json").read_text())
        with np.load(POLICIES / name / "weights.npz") as w:
            self.arrays = {k: w[k].astype(float) for k in w.files}
        m = self.manifest
        if [b["name"] for b in m["blocks"]] != BLOCKS or set(m["actuators"]) - {"direct", "direct_rb"}:
            raise ValueError(f"{name}: inputs or actuators this module doesn't build")

    def act(self, inputs: np.ndarray) -> np.ndarray:
        """The policy's action, each in [-1, 1], for one vector of inputs."""
        h = (inputs - self.arrays["obs_mean"]) / np.sqrt(self.arrays["obs_var"] + 1e-5)
        for layer in self.manifest["network"]:
            if layer["type"] == "linear":
                h = self.arrays[layer["weight"]] @ h + self.arrays[layer["bias"]]
            elif layer["type"] == "elu":
                h = np.where(h > 0, h, np.expm1(np.minimum(h, 0)))
            elif layer["type"] == "layernorm":
                h = (h - h.mean()) / np.sqrt(h.var() + layer["eps"]) * self.arrays[layer["weight"]] \
                    + self.arrays[layer["bias"]]
            elif layer["type"] == "tanh":
                h = np.tanh(h)
            else:
                raise ValueError(f"unknown layer {layer['type']}")
        return h


def clip(action: np.ndarray, step: list, net: np.ndarray, budget: np.ndarray) -> np.ndarray:
    """The change (deg) an action asks for, clipped so each control's net change stays within +-budget, as training's
    knob window clipped it."""
    return np.clip(np.clip(action, -1, 1) * np.asarray(step), -budget - net, budget - net)


class Episode:
    """One episode's inputs: from `goal` and the reading it starts on, then one advance() per step."""

    def __init__(self, policy: Policy, goal: dict, reading: dict):
        m = policy.manifest
        self.policy, self.goal, self.steps = policy, dict(goal), 0
        n, frame = len(m["controls"]), len(m["goals"]) + 2 * len(m["controls"])
        self.change, self.action = np.zeros(n), np.zeros(n)
        self.error = self.error_before = self._error(reading)
        self.features = self._features(reading)
        self.history = np.zeros((m["history"] - 1, frame))

    def _error(self, reading: dict) -> np.ndarray:
        return np.array([(transform(reading[g["key"]], g["transform"]) - transform(self.goal[g["key"]], g["transform"]))
                         / g["scale"] for g in self.policy.manifest["goals"]])

    def _features(self, reading: dict) -> np.ndarray:
        values = {**reading, **{g["goal_key"]: self.goal[g["key"]] for g in self.policy.manifest["goals"]}}
        return np.array([(transform(values[f["key"]], f["transform"]) - f["mean"]) / f["std"]
                         for f in self.policy.manifest["features"]])

    def inputs(self) -> np.ndarray:
        return np.concatenate([self.change, self.features, self.error, self.error - self.error_before, self.action,
                               np.zeros_like(self.action), self.history.ravel()])

    def advance(self, action: np.ndarray, change: np.ndarray, reading: dict) -> None:
        """The step's action (the policy's output, or 0 if declined), the change it made (deg, after clipping) and the
        reading after it."""
        self.history = np.vstack([self.history[1:], np.concatenate([self.error, self.action, change])])
        self.change = self.change + change
        self.error_before, self.error = self.error, self._error(reading)
        self.features = self._features(reading)
        self.action = np.asarray(action, float)
        self.steps += 1

    @property
    def over(self) -> bool:
        return self.steps >= self.policy.manifest["episode_length"]
