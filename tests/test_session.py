"""S0: check_env() reports the session's environment without touching Channel Access; records save as JSON."""
import json
import re
import subprocess
import sys

from twobunch_deploy import session


def test_check_env():
    env = session.check_env()
    assert env["import_errors"] == {}
    assert set(env["packages"]) == {"numpy", "scipy", "matplotlib", "pyepics"}
    assert set(env["epics_env"]) == set(session.EPICS_VARS)
    assert re.fullmatch(r"[0-9a-f]{64}", env["code_sha256"])
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", env["git_commit"])
    json.dumps(env)                                 # the record must be JSON-serialisable


def test_check_env_loads_no_channel_access():
    """In a fresh interpreter, so a test that uses Channel Access can't hide a CA call here."""
    code = "import epics; from twobunch_deploy import session; session.check_env(); print(epics.ca.libca is None)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=session.REPO)
    assert out.stdout.strip() == "True"


def test_save_record(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "DATA", tmp_path)
    path = session.save_record("S0", "check_env", {"a": 1})
    assert path.parent.parent == tmp_path and path.parent.name.endswith("_S0")
    assert json.loads(path.read_text()) == {"a": 1}
