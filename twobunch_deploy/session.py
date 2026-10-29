"""Calls the notebooks make. S0: where and with what a session runs, and saving records under data/."""
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
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

# (module the package imports, its distribution name for the version)
DEPENDENCIES = [("numpy", "numpy"), ("scipy", "scipy"), ("matplotlib", "matplotlib"), ("epics", "pyepics")]
# Channel Access settings. pyepics sets EPICS_CA_MAX_ARRAY_BYTES itself on its first CA call when it is
# unset, so check_env reads these before importing anything.
EPICS_VARS = ["EPICS_CA_ADDR_LIST", "EPICS_CA_AUTO_ADDR_LIST", "EPICS_CA_SERVER_PORT",
              "EPICS_CA_MAX_ARRAY_BYTES", "PYEPICS_LIBCA"]


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
