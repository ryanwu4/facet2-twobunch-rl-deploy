"""S5: the row correction and the step reading.

The synthetic tests run everywhere. The comparisons with Yiheng Ye's row correction and with the v7 replay
(online_replay_v7_noauto_2026-10-19) on the DAQ runs, and the session test on the soft IOC, run only where those
are (conftest.py).
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from conftest import (MODES, V7, XTCAV, YIHENG, constant_values, daq, l1_readbacks, needs_daq, needs_v7, pil_crop,
                      tcav_readbacks)
from twobunch_deploy import acquire, machine, pvs, screen, session, viz

X = screen.x_positions(30.5)


def tilted(shift_per_row=0.1, rows=(40, 140), column=200.0):
    """A processed frame: a Gaussian beam 6 columns wide in each of `rows`, its centre moving by `shift_per_row`
    columns per row from `column` at the first."""
    image = np.zeros((300, 450))
    columns = np.arange(450)
    for row in range(*rows):
        centre = column + shift_per_row * (row - rows[0])
        image[row] = 100 * np.exp(-0.5 * ((columns - centre) / 6) ** 2)
    return image


def test_row_reference_and_shift():
    image = tilted()
    reference = screen.row_reference([image, image])
    fitted = np.flatnonzero(np.isfinite(reference.centroids))
    assert fitted.tolist() == list(range(40, 140))             # rows with signal
    assert reference.centroids[fitted] == pytest.approx(200 + 0.1 * (fitted - 40), abs=1e-6)
    assert reference.shifts[:40] == pytest.approx(224.5 - 200, abs=1e-6)   # the nearest fitted row's, past the ends
    corrected = screen.shift_rows(image, reference.shifts)
    columns = np.arange(450)
    centres = np.array([np.average(columns, weights=corrected[row]) for row in fitted])
    assert centres == pytest.approx(224.5, abs=1e-3)            # every row's centroid at the crop's centre
    assert screen.row_reference([np.zeros((300, 450))]) is None


def test_shift_rows():
    image = np.zeros((2, 6))
    image[:, 2] = 1.0
    shifted = screen.shift_rows(image, np.array([1.0, -2.5]))
    assert shifted.tolist() == [[0, 0, 0, 1, 0, 0], [0.5, 0, 0, 0, 0, 0]]     # half the second row's beam leaves


def two_bands(offset=0):
    """A processed frame with a witness band (rows 40-59) and a drive band (rows 100-139), the drive `offset`
    columns right of column 200."""
    image = np.zeros((300, 450))
    image[40:60, 100:120] = 1.0
    image[100:140, 200 + offset:240 + offset] = 2.0
    return image


def test_step_cut():
    cut = screen.valley(two_bands().sum(axis=1)).cut
    assert screen.step_cut([two_bands()] * 2 + [np.zeros((300, 450))] * 2) == cut     # half have a valley
    assert screen.step_cut([two_bands()] + [np.zeros((300, 450))] * 2) is None   # fewer than half
    assert screen.step_cut([]) is None
    shifted = np.roll(two_bands(), 1, axis=0)
    assert screen.step_cut([two_bands(), shifted]) == round((cut + cut + 1) / 2)    # the median, rounded to even


def test_reading():
    """+V moves the drive 10 columns right of its -V position: spacing = 10 columns / 2 S."""
    cut = screen.valley(two_bands().sum(axis=1)).cut
    plus = [screen.band_stats(two_bands(10), X, cut)] * 3
    minus = [screen.band_stats(two_bands(0), X, cut)] * 3
    r = screen.reading(plus, minus, 1100.0)
    streak = 1100.0 / screen.UM_PER_DEG_RF
    assert r["spacing"] == pytest.approx(10 * 30.5e-6 / (2 * streak))
    assert r["drive_sigma_z"] == pytest.approx(plus[0]["var_d"] ** 0.5 / streak)
    assert r["drive_charge_fraction"] == pytest.approx(3200 / 3600) and r["edge_frac"] == 0 and r["cut_sens"] == 0
    edge = two_bands(0)
    edge[100:105, 0:5] = 2.0                        # some of the -V drive within EDGE_BAND of the crop's edge
    r = screen.reading(plus, [screen.band_stats(edge, X, cut)] * 3, 1100.0)
    assert r["edge_frac"] == pytest.approx(50 / 3650)               # the larger of the two polarities'


def frames_for(read_tags, drop=None):
    """Hand crops for read_step: each + frame is two_bands(80) (a spacing of 81 um at 1100 um/deg), each -
    two_bands(0), each off frame the beam of tilted() without tilt, all with a dark level of 31 counts in the pixel
    rows; and the matching background."""
    background = np.zeros((300, 450))
    background[:screen.PIXEL_ROWS] = 31
    image = {"+": 50 * two_bands(80), "-": 50 * two_bands(0), "off": tilted(0)}
    crops = np.stack([np.rint(background + image.get(t, 0)).astype("int16") for t in read_tags])
    return crops, np.array(read_tags, dtype=object), np.zeros(len(read_tags), bool) if drop is None else drop, \
        background


def test_read_step():
    tags = ["off"] * 2 + ["+"] * 36 + ["neither"] + ["-"] * 16 + ["unmatched"]
    crops, tags, drop, background = frames_for(tags)
    # 20 dropped + frames with the bands 10 rows lower and no drive offset: in the cut or the reading, they'd move both
    drop[2:22] = True
    crops[2:22] = np.rint(background + 50 * np.roll(two_bands(0), 10, axis=0)).astype("int16")
    r = screen.read_step(crops, tags, drop, background, X, 1100.0)
    assert all(r.checks.values()) and r.reference is not None
    kept = screen.shift_rows(screen.process(crops[22], background), r.reference.shifts)
    assert r.cut == screen.valley(kept.sum(axis=1)).cut and np.array_equal(r.medians["+"], kept)
    assert r.reading["spacing"] == pytest.approx(80 * 30.5e-6 / (2 * 1100.0 / screen.UM_PER_DEG_RF))
    assert [s is not None for s in r.stats] == [t in ("+", "-") for t in tags]    # dropped frames too
    assert set(r.medians) == {"+", "-"}
    # beam widths and intensities: an off frame uncorrected (a narrow beam: the threshold cuts its tails), a kept +
    # frame corrected, none for neither
    off = screen.process(crops[0], background)
    assert r.x_rms[0] == screen.x_rms(off, X) < 0.15e-3 and r.intensity[0] == off.sum()
    assert r.x_rms[22] == screen.x_rms(kept, X) > 1e-3 and r.x_rms[38] is r.intensity[38] is None
    assert screen.x_rms(np.zeros((300, 450)), X) is None

    # 15 kept per polarity is enough, 14 not
    drop[22] = True
    assert all(screen.read_step(crops, tags, drop, background, X, 1100.0).checks.values())
    drop[23] = True
    assert not screen.read_step(crops, tags, drop, background, X, 1100.0).checks["+ frames >= 15"]

    # no off frame: no reference; read uncorrected, and invalid
    r = screen.read_step(crops[2:], tags[2:], drop[2:], background, X, 1100.0)
    assert r.reference is None and not r.checks["row-correction reference"] and r.reading is not None

    # faint - frames, 48,000 counts each: every check passes but the beam check
    faint = np.zeros((300, 450))
    faint[40:60, 100:120] = 40
    faint[100:110, 200:240] = 80
    faint_crops = np.stack([crops[0]] * 2 + [crops[22]] * 16 + [np.rint(background + faint).astype("int16")] * 16)
    faint_tags = np.array(["off"] * 2 + ["+"] * 16 + ["-"] * 16, dtype=object)
    r = screen.read_step(faint_crops, faint_tags, np.zeros(34, bool), background, X, 1100.0)
    assert r.intensity[18:] == [48000.0] * 16
    assert [k for k, ok in r.checks.items() if not ok] == ["beam: median intensity >= 1e+05"]

    # only off frames, as on parasitic beam: the reference is still built
    r = screen.read_step(crops[:2], tags[:2], drop[:2], background, X, 1100.0)
    assert r.reference is not None and r.cut is None and r.reading is None and r.medians == {}
    assert [k for k, ok in r.checks.items() if not ok] == list(r.checks)[:2] + ["step cut"] + list(r.checks)[4:]


def test_step_figure():
    tags = ["off"] * 2 + ["+"] * 16 + ["neither"] + ["-"] * 16 + ["unmatched"]
    crops, tags, drop, background = frames_for(tags)
    drop[2] = True
    r = screen.read_step(crops, tags, drop, background, X, 1100.0)
    record = {"camera": MODES["roi"], "cut_row": r.cut, "valid": True, "checks": r.checks, "tags": {},
              "reading": {"spacing_um": 1, "drive_sigma_z_um": 1, "drive_charge_fraction": 1, "edge_frac": 0,
                          "cut_sens_um": 0},
              "polarity": {"+": {"kept": 15, "dropped": 1, "x_d": 0.006, "x_w": 0.003, "x_d - x_w": 0.003,
                                 "witness_low": True}, "-": {"kept": 0, "dropped": 0}},
              "per_frame": {"pulse_id": [36 * i for i in range(len(tags))], "tag": tags.tolist(),
                            "drop": drop.tolist(), "band_stats": r.stats,
                            "amplitude": [None if t == "unmatched" else 22.0 for t in tags],
                            "phase": [None if t == "unmatched" else 90.0 for t in tags],
                            "timestamp": [100 + 0.1 * i for i in range(len(tags))]}}
    names = [*pvs.L1, *pvs.L2]
    updates = [machine.Update(name, 36 * i, 100 + 0.1 * i, -23.5 + 0.1 * i if i % 3 else 0.0)
               for i in range(4) for name in names[:2]] + [machine.Update(name, 0, 50.0, -20.0) for name in names[2:]]
    updates.append(machine.Update(names[-1], 72, 100.2, -19.0))                  # a change: drawn first -> last
    record["phases"] = {name: acquire.summarise(updates, name) for name in names}
    assert len(viz.step(record, {"+": r.medians["+"]}, updates).axes) == 6
    assert viz.changed(record["phases"][names[-1]]) == "-20.000 -> -19.000"
    assert viz.changed(record["phases"][names[-2]]) == "-20.000" and viz.changed({"updates": 0}) == "no update"


@pytest.fixture(scope="module")
def yiheng():
    sys.path[:0] = [str(YIHENG), str(YIHENG.parent / "XTCAV_analysis_script")]
    import dataset
    import xtcav
    return dataset, xtcav


def daq_steps(dataset, xtcav, run):
    """Each step of a DAQ run with v7's inputs: the steps, step(k) giving the hand crops and tags of step k's +, -
    and off shots in shot order, and the run's DAQ background. The tags are acquire.tag on the whole run's readbacks,
    as v7 classified them (test_tcav: equal to classify_tcav_shots)."""
    ds = dataset.DATASET(str(XTCAV), *run.split("_"))
    classified = xtcav.classify_tcav_shots(ds)
    tags = acquire.tag(np.asarray(classified["amp"], float), np.asarray(classified["phase"], float))
    steps = np.asarray(ds.common_stepIndex)
    images = ds.images["DTOTR2"]
    pulse_id = np.asarray(images["pid"])[np.asarray(images["common_index"]) - 1]
    background = screen.trained_frame(ds._data["backgrounds"]["DTOTR2"].T.ravel(), MODES["roi"]).astype(float)

    def step(k):
        shots = np.flatnonzero((steps == k) & (tags != "neither"))
        with h5py.File(XTCAV / run / "images/DTOTR2" / Path(images["loc"][k - 1]).name) as f:
            ids = f["entry/instrument/NDAttributes/NDArrayUniqueId"][:]
            rows = [int(np.flatnonzero(ids == pulse_id[i])[0]) for i in shots]     # the first, as the DAQ classes
            crops = np.stack([screen.trained_frame(f["entry/data/data"][r].ravel(), MODES["roi"]) for r in rows])
        return crops, tags[shots]

    return sorted(set(steps.tolist())), step, background


@needs_v7
def test_row_correction_equals_daq_classes(yiheng):
    """On the off frames of steps of each run (v7's off), processed with the DAQ background: the centroids equal
    xtcav.fit_row_distortion's, the shifts compute_distortion_shifts', and shifting a + frame equals
    apply_row_shift."""
    dataset, xtcav = yiheng
    for run in ("TEST_15668", "E300_15673"):
        steps, step, background = daq_steps(dataset, xtcav, run)
        for k in steps[::8]:
            crops, tags = step(k)
            off = [screen.process(c, background) for c, t in zip(crops, tags) if t == "off"]
            reference = screen.row_reference(off)
            np.testing.assert_array_equal(reference.centroids, xtcav.fit_row_distortion(reference.image))
            theirs = xtcav.compute_distortion_shifts(reference.image, reference.centroids)["shifts_px"]
            np.testing.assert_allclose(reference.shifts, theirs, rtol=0, atol=1e-9)
            plus = screen.process(crops[list(tags).index("+")], background)
            np.testing.assert_array_equal(screen.shift_rows(plus, theirs), xtcav.apply_row_shift(plus, theirs))


def v7_kept(ours, theirs):
    """Which of our frames v7 kept: its offsets are ours, in order, without the frames it dropped."""
    kept, j = [], 0
    for i, offset in enumerate(ours):
        if j < len(theirs) and offset == theirs[j]:
            kept.append(i)
            j += 1
    assert j == len(theirs)
    return kept


@needs_v7
@pytest.mark.parametrize("run", ["TEST_15668", "E300_15671", "E300_15673"])
def test_step_readings_equal_v7(yiheng, run):
    """Every step of the run through read_step with v7's inputs (DAQ background, the tags, the run's calibration at
    30.5 um/px, no drop): the same cut, frame offsets and reading as online_replay_v7_noauto_2026-10-19.

    v7 also dropped the frames whose energy-spectrum fit failed (xtcav.fit_dy_from_shot, which the reading doesn't
    use; v7's n_gate_fail): in those steps the frames v7 lacks are taken out before the reading is compared.
    """
    v7 = json.loads((V7 / f"{run}.json").read_text())
    steps, step, background = daq_steps(*yiheng, run)
    streak = v7["cal_um_per_deg"] / screen.UM_PER_DEG_RF
    taken_out = 0
    for k in steps:
        crops, tags = step(k)
        r = screen.read_step(crops, tags, np.zeros(len(tags), bool), background, X, v7["cal_um_per_deg"])
        expected = v7["steps"].get(str(k))
        if expected is None:                        # v7 has no reading without a cut or without both polarities
            assert r.reading is None
            continue
        assert r.cut == expected["cut_rows_plus"][0]
        stats = {}
        for sign, key in (("+", "offsets_plus_um"), ("-", "offsets_minus_um")):
            ours = [s for s, t in zip(r.stats, tags) if t == sign and s is not None]
            kept = v7_kept([round(1e6 * (s["x_d"] - s["x_w"]) / streak, 1) for s in ours], expected[key])
            stats[sign] = [ours[i] for i in kept]
            taken_out += len(ours) - len(kept)
        reading = screen.reading(stats["+"], stats["-"], v7["cal_um_per_deg"])
        for ours, theirs in (("spacing", "spacing_um"), ("drive_sigma_z", "drive_sigma_z_um"),
                             ("cut_sens", "cut_sens_um")):
            assert 1e6 * reading[ours] == pytest.approx(expected[theirs], rel=1e-9, nan_ok=True)
        for key in ("drive_charge_fraction", "edge_frac"):
            assert reading[key] == pytest.approx(expected[key], rel=1e-9, abs=1e-15)
        valid = abs(reading["spacing"]) > screen.SPACING_FLOOR and reading["edge_frac"] < screen.EDGE_MAX
        assert valid == bool(expected["valid"])
        if expected["n_gate_fail"] == 0:
            assert reading == r.reading
            assert valid == (r.checks["|spacing| > 60 um"] and r.checks["edge fraction < 0.01"])
    assert taken_out <= sum(s["n_gate_fail"] for s in v7["steps"].values())


@needs_daq
def test_step(replay, tmp_path, monkeypatch):
    """step() on the soft IOC serving the whole DAQ step with its TCAV readbacks: the record and the saved files
    equal read_step on the same frames, tagged and dropped offline."""
    monkeypatch.setattr(session, "DATA", tmp_path)
    data, frame_ids = daq(0, None)
    crops = np.stack([pil_crop(frame) for frame in data])
    readbacks = tcav_readbacks()
    amplitude = np.array([readbacks.get(p, (np.nan, np.nan))[0] for p in frame_ids])
    phase = np.array([readbacks.get(p, (np.nan, np.nan))[1] for p in frame_ids])
    tags = acquire.tag(amplitude, phase)
    drop = acquire.dropped(frame_ids, tags, 2.0)
    background = session.Background(screen.pedestal(crops), "pedestal", {})
    expected = screen.read_step(crops, tags, drop, background.image, screen.x_positions(31.22), 1100.0)

    replay("step")                                  # RESOLUTION 31.22, as on srv01
    r = session.step(background, 1100.0, seconds=19, drop_s=2.0)
    frames = r.record["per_frame"]
    assert frames["pulse_id"] == frame_ids and frames["tag"] == tags.tolist() and frames["drop"] == drop.tolist()
    assert "unmatched" in frames["tag"] and "off" in frames["tag"] and any(frames["drop"])
    np.testing.assert_equal(frames["band_stats"], expected.stats)
    assert frames["intensity"] == expected.intensity and frames["x_rms"] == expected.x_rms
    assert r.record["cut_row"] == expected.cut and r.record["checks"] == expected.checks
    assert r.record["reading"]["spacing_um"] == 1e6 * expected.reading["spacing"]
    assert r.record["reference"]["off_frames"] == list(tags).count("off")
    widths = [1e3 * w for w, t in zip(expected.x_rms, tags) if t == "off"]
    assert r.record["reference"]["off_x_rms_mm"] == {"median": np.median(widths), "max": max(widths)}
    for sign in ("+", "-"):
        np.testing.assert_array_equal(r.medians[sign], expected.medians[sign])
    l1 = l1_readbacks()
    constants = constant_values()
    assert 0.0 in constants.values()                              # LI11:SBST:1:PDES, 0 in the DAQ
    for name, value in constants.items():                         # S6, S7: one update each, on subscribing
        assert r.record["phases"][name] == {"updates": 1, "median": value, "first": value, "last": value,
                                            "zeros": int(value == 0)}
    for name in next(iter(l1.values())):                          # one per frame with a shot
        values = [l1[p][name] for p in frame_ids if p in l1]
        assert r.record["phases"][name] == {"updates": len(values), "median": np.median(values), "first": values[0],
                                            "last": values[-1], "zeros": 0}
    assert len(frames["timestamp"]) == len(frame_ids) and np.all(np.diff(frames["timestamp"]) > 0)
    np.testing.assert_equal(json.loads(Path(r.path).read_text()), json.loads(json.dumps(r.record)))
    saved = np.load(Path(r.path).with_suffix(".npz"))
    assert np.array_equal(saved["crops"], crops) and np.array_equal(saved["reference_shifts"], expected.reference.shifts)
    r.show()
    assert Path(r.path).with_suffix(".png").stat().st_size > 0
