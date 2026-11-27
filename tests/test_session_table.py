"""R16: tools/session_table.py on a day's records made by advise and actuate (a closed-loop run on the soft IOC), and
on a step record copied back beside them."""
import csv
import importlib.util
import json

import numpy as np
import pytest

from conftest import REPO, needs_daq
from test_actuate import armed, fresh  # noqa: F401 (fixtures)
from test_advise import clean, fake, start  # noqa: F401 (clean: autouse)
from twobunch_deploy import advise, session

spec = importlib.util.spec_from_file_location("session_table", REPO / "tools" / "session_table.py")
session_table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(session_table)


def test_local_paths_and_step_fields(tmp_path):
    """A host path maps under the copied-back data folder; a step record gives its reading, frames and PV medians."""
    host = "/home/fphysics/rwu4/facet2-twobunch-rl-deploy/data/2026-12-03_S5/step_101500.json"
    assert session_table.local(host, tmp_path) == tmp_path / "2026-12-03_S5" / "step_101500.json"
    path = session_table.local(host, tmp_path)
    path.parent.mkdir()
    path.write_text(json.dumps({"valid": True, "cut_polarity": "-", "streak_um_per_deg": -1035.0,
                                "polarity": {"+": {"kept": 18}, "-": {"kept": 17}},
                                "reading": {"spacing_um": 150.0, "drive_sigma_z_um": 90.0},
                                "phases": {"KLYS:LI11:11:PREQ": {"median": -20.0}},
                                "drift": {"BLEN:LI11:359:BRAWBR": {"median": 4000.0}},
                                "per_frame": {"timestamp": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                                              "tag": ["+", "+", "-", "+", "-", "off"],
                                              "drop": [False, False, False, True, False, False],
                                              "band_stats": [{"x_d": 2e-4, "x_w": 1e-4}, {"x_d": 2.2e-4, "x_w": 1e-4},
                                                             {"x_d": 1e-4, "x_w": 1e-4}, {"x_d": 9e-4, "x_w": 0.0},
                                                             {"x_d": 1.5e-4, "x_w": 1e-4}, None]}}))
    fields = session_table.step_fields(path)
    assert fields["read_spacing_um"] == 150.0 and fields["frames_minus"] == 17 and fields["cut_polarity"] == "-"
    assert fields["median KLYS:LI11:11:PREQ"] == -20.0 and fields["median BLEN:LI11:359:BRAWBR"] == 4000.0
    # R7: the halves split at 2.5 s; the dropped + frame at 3 s is left out
    assert fields["offset_plus_first_um"] == pytest.approx(110.0) and fields["offset_plus_second_um"] is None
    assert fields["offset_minus_first_um"] == pytest.approx(0.0)
    assert fields["offset_minus_second_um"] == pytest.approx(50.0)
    assert session_table.step_fields(tmp_path / "missing.json") == {}


@needs_daq
def test_table_of_a_closed_loop_run(armed, monkeypatch):     # noqa: F811
    """A run that stops with the budget used up: the episode's start, the run's start, five applied proposals each with
    its ledger writes, the undecided sixth, and the stop with its reason, in time order; written as CSV."""
    start(monkeypatch, *[fake() for _ in range(6)])
    monkeypatch.setattr(advise.policy(), "act", lambda inputs: np.array([1.0, 1.0]))
    advise.run(None, 1100.0)
    day = advise._day()
    rows = session_table.table(day)
    assert [r["event"] for r in rows] == ["goal", "run started"] + ["proposal"] * 6 + ["run stopped"]
    proposals = rows[2:8]
    assert [p["decision"] for p in proposals] == ["applied (unattended)"] * 5 + [None]
    assert [p["writes"].count("done") for p in proposals] == [2, 2, 2, 2, 1, 0]
    assert proposals[4]["applied_L1"] == 0.0 and proposals[4]["change_L2"] == proposals[4]["applied_L2"]
    assert rows[-1]["detail"] == "the budget used up on both controls"
    monkeypatch.setattr("sys.argv", ["session_table.py", str(day)])          # the command line writes the same rows
    session_table.main()
    with open(day / "table.csv") as f:
        assert len(list(csv.DictReader(f))) == len(rows)
    assert session.DATA in day.parents
