"""Calls the notebooks make. S0: where and with what a session runs, and saving records under data/.
S1: capture_test() of the camera frames and pulse IDs."""
from __future__ import annotations

import datetime
import getpass
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

# (module the package imports, its distribution name for the version)
DEPENDENCIES = [("numpy", "numpy"), ("scipy", "scipy"), ("matplotlib", "matplotlib"), ("epics", "pyepics")]
# Channel Access settings. pyepics sets EPICS_CA_MAX_ARRAY_BYTES itself on its first CA call when it is
# unset, so check_env reads these before importing anything.
EPICS_VARS = ["EPICS_CA_ADDR_LIST", "EPICS_CA_AUTO_ADDR_LIST", "EPICS_CA_SERVER_PORT",
              "EPICS_CA_MAX_ARRAY_BYTES", "PYEPICS_LIBCA"]
# Pulse IDs count the 360 Hz pulses from 0 to 131039, then wrap. At 10 Hz, consecutive frames are 36 apart.
PULSE_ID_PERIOD = 131040
PULSE_IDS_PER_FRAME = 36


def check_env() -> dict:
    """Host, user, Python, package versions and EPICS settings of this session, and the code it runs.

    Imports the runtime dependencies (importing pyepics loads no Channel Access library) and reads no PV.
    """
    epics_env = {k: os.environ.get(k) for k in EPICS_VARS}
    env = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "host": socket.gethostname(),
           "user": getpass.getuser(), "python": sys.version.split()[0], "executable": sys.executable,
           "platform": platform.platform(), "git_commit": git_commit(), "code_sha256": code_sha256(),
           "epics_env": epics_env, "packages": {}, "import_errors": {}}
    for module, dist in DEPENDENCIES:
        try:
            importlib.import_module(module)
            env["packages"][dist] = importlib.metadata.version(dist)
        except Exception as e:                      # record every failure, not only the first
            env["import_errors"][dist] = repr(e)
    return env


def git_commit() -> str:
    """Commit checked out in REPO, read from .git without running git; "unknown" if it can't be read."""
    try:
        head = (REPO / ".git" / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head                             # detached HEAD
        ref = head.removeprefix("ref: ")
        if (REPO / ".git" / ref).exists():
            return (REPO / ".git" / ref).read_text().strip()
        for line in (REPO / ".git" / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return "unknown"


def code_sha256() -> str:
    """sha256 over the package's .py files (name and content, sorted by name).

    Identifies the code that runs whether or not it is committed; `pixi run code-hash` and CI print the
    same value for a checkout.
    """
    h = hashlib.sha256()
    for f in sorted((REPO / "twobunch_deploy").glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def save_record(stage: str, name: str, record: dict) -> Path:
    """Write record as JSON to data/<date>_<stage>/<name>_<HHMMSS>.json and return the path."""
    now = datetime.datetime.now()
    path = DATA / f"{now:%Y-%m-%d}_{stage}" / f"{name}_{now:%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1))
    return path


def pulse_id_steps(pulse_ids: list[int]) -> dict[int, int]:
    """How often each step between consecutive pulse IDs occurs, counted across the wrap.

    36 is the next frame and 72 one frame missed. 0 is a repeated pulse ID: it happened once in the 4,950
    frames of the TEST_15668 DAQ files, where the IOC gave a frame the next frame's timestamp (so a 72
    comes just before it).
    """
    steps = Counter((b - a) % PULSE_ID_PERIOD for a, b in zip(pulse_ids, pulse_ids[1:]))
    return dict(sorted(steps.items()))


def capture_test(seconds: float = 30, n_saved: int = 10) -> dict:
    """S1: capture the DTOTR2 camera for `seconds` and summarise what arrived.

    Saves the summary to data/<date>_S1/capture_test_<HHMMSS>.json and, next to it (.npz), every frame's
    pulse ID, timestamp and arrival time, plus `n_saved` raw frames spread over the capture.
    """
    import numpy as np                  # imported here, so that check_env() runs even without numpy or pyepics
    from . import machine

    frames = machine.frames(seconds)
    summary = {"seconds": seconds, "frames": len(frames)}
    if frames:
        steps = pulse_id_steps([f.pulse_id for f in frames])
        expected = sum(s * n for s, n in steps.items()) // PULSE_IDS_PER_FRAME + 1
        latency = [f.received - f.timestamp for f in frames]
        summary.update({
            "capture_fraction": len(frames) / expected,
            "pulse_id_steps": steps,
            "latency_s": {"first": latency[0], "median": float(np.median(latency)), "max": max(latency)},
            "dtype": str(frames[0].data.dtype),
            "lengths": sorted({f.data.size for f in frames}),
            "min": int(min(f.data.min() for f in frames)),
            "max": int(max(f.data.max() for f in frames)),
        })
    path = save_record("S1", "capture_test", summary)
    if frames:
        saved = frames[::max(1, len(frames) // n_saved)][:n_saved]
        np.savez_compressed(path.with_suffix(".npz"), pulse_id=[f.pulse_id for f in frames],
                            timestamp=[f.timestamp for f in frames], received=[f.received for f in frames],
                            saved_pulse_id=[f.pulse_id for f in saved],
                            saved_data=np.stack([f.data for f in saved]))
    return summary
