"""S9: the console's calls. check_pvs(), watch() and peek_strip() on the soft IOC run only where its DAQ data are
(conftest.py); status() runs everywhere on records written here.
"""
import datetime
import json

import numpy as np

from conftest import DRIFT_STAND_IN, MODES, daq, magnet_values, needs_daq, pil_crop
from twobunch_deploy import pvs, screen, session


@needs_daq
def test_check_pvs(replay, tmp_path, monkeypatch):
    """Every PV in pvs.READ answers on the soft IOC, with the value it serves."""
    monkeypatch.setattr(session, "DATA", tmp_path)
    replay("roi")
    summary = session.check_pvs()
    assert summary["answered"] == summary["pvs"] == len(pvs.READ) and summary["missing"] == []
    read = summary["read"]
    assert read[pvs.CAMERA]["elements"] >= MODES["roi"]["SizeX_RBV"] * MODES["roi"]["SizeY_RBV"]   # its maximum
    assert {name: read[name]["value"] for name in pvs.MAGNETS} == magnet_values()
    assert {name: read[name]["value"] for name in pvs.DRIFT} == DRIFT_STAND_IN
    assert all(r["age_s"] >= 0 for r in read.values())
    [record] = tmp_path.glob("*_S9/check_pvs_*.json")
    assert json.loads(record.read_text()) == json.loads(json.dumps(summary))


def test_status(tmp_path, monkeypatch):
    """status() from today's records: two steps, the second valid, and a record of another call; yesterday's step
    doesn't count."""
    monkeypatch.setattr(session, "DATA", tmp_path)
    today = f"{datetime.date.today():%Y-%m-%d}"
    yesterday = f"{datetime.date.today() - datetime.timedelta(days=1):%Y-%m-%d}"
    step = {"valid": False, "reading": None, "streak_um_per_deg": 1100.0, "background": "b.npz"}
    for date, name, record in [(yesterday, "step_090000", {**step, "valid": True}), (today, "step_100000", step),
                               (today, "step_100030", {**step, "valid": True, "reading": {"spacing_um": 150.0}})]:
        (tmp_path / f"{date}_S5").mkdir(exist_ok=True)
        (tmp_path / f"{date}_S5" / f"{name}.json").write_text(json.dumps(record))
    (tmp_path / f"{today}_S9").mkdir()
    (tmp_path / f"{today}_S9" / "check_pvs_095900.json").write_text("{}")
    summary = session.status()
    assert {k: summary[k] for k in ("date", "records", "steps", "valid_steps")} == {
        "date": today, "records": 3, "steps": 2, "valid_steps": 1}
    assert summary["last_step"] == {"path": str(tmp_path / f"{today}_S5" / "step_100030.json"), "valid": True,
                                    "reading": {"spacing_um": 150.0}, "streak_um_per_deg": 1100.0,
                                    "background": "b.npz"}
    monkeypatch.setattr(session, "DATA", tmp_path / "none")
    assert session.status() == {"date": today, "records": 0, "first": None, "latest": None, "steps": 0,
                                "valid_steps": 0}


@needs_daq
def test_watch(replay, tmp_path, monkeypatch):
    """watch() on the soft IOC: each look is peek()'s newest frame (the same one again if none came since: the IOC
    keeps its last); only a summary is saved."""
    monkeypatch.setattr(session, "DATA", tmp_path)
    data, frame_ids = daq()
    background = session.Background(screen.pedestal(np.stack([pil_crop(frame) for frame in data])), "pedestal", {})
    replay("roi")
    looks = session.watch(background, seconds=3, every_s=1.0)
    assert len(looks) >= 2 and all(look["pulse_id"] in frame_ids for look in looks)
    order = [frame_ids.index(look["pulse_id"]) for look in looks]
    assert order == sorted(order)                                     # newest frames, in time
    for look in looks:
        processed = screen.process(pil_crop(data[frame_ids.index(look["pulse_id"])]), background.image)
        cut = screen.valley(processed.sum(axis=1)).cut
        stats = None if cut is None else screen.band_stats(processed, screen.x_positions(30.5), cut)
        assert look["x_d - x_w"] == (None if stats is None else stats["x_d"] - stats["x_w"])
    [record] = tmp_path.glob("*_S3/watch_*.json")
    assert json.loads(record.read_text())["looks"] == looks
    assert not list(tmp_path.glob("*_S3/*.npz"))                  # no frames saved


@needs_daq
def test_peek_strip(replay, tmp_path, monkeypatch):
    """peek_strip() on the soft IOC's whole step (its + frames are in the middle): each frame's own cut and band
    statistics, and the one pair's spacing, redone from the hand crops it saved."""
    monkeypatch.setattr(session, "DATA", tmp_path)
    data, frame_ids = daq(0, None)
    background = session.Background(screen.pedestal(np.stack([pil_crop(frame) for frame in data])), "pedestal", {})
    replay("step")
    summary = session.peek_strip(background, seconds=19, streak_um_per_deg=1100.0)    # the IOC takes about 17 s
    [record] = tmp_path.glob("*_S3/peek_strip_*.json")
    saved = np.load(record.with_suffix(".npz"))
    frames = summary["per_frame"]
    assert summary["frames"] == len(saved["crops"]) == len(frames["pulse_id"]) > 140
    assert all(np.array_equal(crop, pil_crop(data[frame_ids.index(p)]))
               for crop, p in zip(saved["crops"], frames["pulse_id"]))
    x = screen.x_positions(MODES["step"]["RESOLUTION"])
    processed = [screen.process(crop, background.image) for crop in saved["crops"]]
    cuts = [screen.valley(image.sum(axis=1)).cut for image in processed]
    assert frames["cut"] == cuts
    assert frames["band_stats"] == [None if c is None else screen.band_stats(image, x, c)
                                    for image, c in zip(processed, cuts)]
    pair = summary["one_pair"]
    p, m = frames["pulse_id"].index(pair["plus_pulse_id"]), frames["pulse_id"].index(pair["minus_pulse_id"])
    assert frames["tag"][p] == "+" and frames["tag"][m] == "-"
    assert all(t != "+" or s is None for t, s in zip(frames["tag"][p + 1:], frames["band_stats"][p + 1:]))
    assert "-" not in frames["tag"][m + 1:]                        # the newest of each
    expected = screen.reading([frames["band_stats"][p]], [screen.band_stats(processed[m], x, cuts[p])], 1100.0)
    assert pair["spacing_um"] == 1e6 * expected["spacing"]
    assert json.loads(record.read_text()) == json.loads(json.dumps(summary))
    assert record.with_suffix(".png").stat().st_size > 0
