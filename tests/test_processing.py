"""S3: background, noise processing, energy cut and band statistics.

The synthetic tests run everywhere. The comparisons on DAQ frames with Yiheng Ye's DAQ classes and with
twobunch-s2e-rl's online/features.py, and the session tests on the soft IOC, run only where those and the DAQ
files are (conftest.py).
"""
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from conftest import FEATURES, MODES, XTCAV, YIHENG, daq, needs_daq, needs_features, needs_yiheng, pil_crop
from twobunch_deploy import screen, session, viz


def two_bands():
    """A processed frame with a witness band (rows 40-59, columns 100-119, 400 counts in all) and a drive band
    (rows 100-139, columns 200-239, 3200)."""
    image = np.zeros((300, 450))
    image[40:60, 100:120] = 1.0
    image[100:140, 200:240] = 2.0
    return image


def test_process():
    background = np.zeros((300, 450))
    background[:184] = 31                       # the live dark level (S1); the padding rows stay 0
    crop = background.astype("int16")
    crop[50:55, 100:105] = 500                  # beam: 469 after the background
    crop[20, 20] = 4000                         # a hot pixel in the dark: the median around it is 0
    crop[60:63, 200:203] = 1500                 # a 3 x 3 spot above HOT_PIXEL: each corner has 4 bright and 5
                                                # dark neighbours, so its median is 0
    crop[100, 300], crop[150, 300] = 1031, 1032     # 1000 stays, 1001 is a hot pixel
    crop[170, 10], crop[170, 20] = 65, 66       # 34 is below the threshold, 35 isn't
    expected = np.zeros((300, 450))
    expected[50:55, 100:105] = 469
    expected[60:63, 200:203] = [[0, 1469, 0], [1469, 1469, 1469], [0, 1469, 0]]
    expected[100, 300] = 1000
    expected[170, 20] = 35
    assert np.array_equal(screen.process(crop, background), expected)


def test_pedestal():
    crops = np.zeros((3, 300, 450), "int16")
    crops[:, :184] = 31
    crops[:, 50:60, 100:200] = 900              # beam on 1000 of the 82,800 pixels: the median is the dark level
    background = screen.pedestal(crops)
    assert (background[:184] == 31).all() and (background[184:] == 0).all()


def test_x_positions():
    x = screen.x_positions(30.5)
    assert len(x) == 450 and x[:3] == pytest.approx([0, 30.5e-6, 61e-6])


def test_valley_and_band_stats():
    image, x = two_bands(), screen.x_positions(30.5)
    valley = screen.valley(image.sum(axis=1))
    low, high = sorted(valley.between)
    assert 40 <= low < 60 and 100 <= high < 140 and 60 <= valley.cut < 100
    stats = screen.band_stats(image, x, valley.cut)
    offset = x[200:240].mean() - x[100:120].mean()
    assert stats == pytest.approx({
        "cut_row": valley.cut, "witness_low": True, "x_d": x[200:240].mean(), "x_w": x[100:120].mean(),
        "var_d": x[200:240].var(), "frac_d": 3200 / 3600, "edge_frac": 0, "off_cut_hi": offset,
        "off_cut_lo": offset})
    # one band: no valley; a cut past every row: no witness band
    assert screen.valley(image[80:].sum(axis=1)).cut is None
    assert screen.band_stats(image, x, 299) is None


def test_figures():
    image, x = two_bands(), screen.x_positions(30.5)
    valley = screen.valley(image.sum(axis=1))
    assert len(viz.peek(image, image, valley, screen.band_stats(image, x, valley.cut), x).axes) == 3
    assert len(viz.peek(image, image, screen.valley(image[80:].sum(axis=1)), None, x).axes) == 3
    assert len(viz.background(image, image).axes) == 4          # two images, each with its colour bar
    assert len(viz.background(screen.pedestal(image[None]), image).axes) == 4      # a constant background


@needs_yiheng
@pytest.mark.parametrize("run", ["TEST_15668", "E300_15673"])
def test_process_equals_daq_classes(run):
    """The DAQ classes' preprocessing with the deployed settings (no auto crop) on 12 frames of each run, with
    the run's background: the same pixels. Also each column's x."""
    if not (XTCAV / run / f"{run}.mat").exists():
        pytest.skip(f"needs the DAQ run {XTCAV / run}")
    sys.path[:0] = [str(YIHENG), str(YIHENG.parent / "XTCAV_analysis_script")]
    import dataset
    import xtcav
    ds = dataset.DATASET(str(XTCAV), *run.split("_"))
    roi = MODES["roi"]
    # the DAQ stores the background transposed from the raw frames (dataset.getImageBackground)
    background = screen.trained_frame(ds._data["backgrounds"]["DTOTR2"].T.ravel(), roi).astype(float)
    cam = ds.images["DTOTR2"]
    hot = 0
    for ind in np.linspace(0, ds.N - 1, 12).astype(int):
        img = xtcav.preprocess_shot(ds, "DTOTR2", ind, hand_crop=screen.HAND_CROP, auto_crop=None,
                                    subtract_background=True, hot_pix_threshold=screen.HOT_PIXEL,
                                    noise_threshold=screen.THRESHOLD, median_filter=3)
        pid = cam["pid"][cam["common_index"][ind] - 1]
        with h5py.File(XTCAV / run / "images/DTOTR2" / img.filename) as f:
            [k] = np.flatnonzero(f["entry/instrument/NDAttributes/NDArrayUniqueId"][:] == pid)
            crop = screen.trained_frame(f["entry/data/data"][k].ravel(), roi)
        assert np.array_equal(screen.process(crop, background), img.data)
        hot += np.sum(crop - background > screen.HOT_PIXEL)
    assert hot > 0                                  # the hot-pixel step was tested
    assert np.allclose(screen.x_positions(ds.metadata["DTOTR2"]["RESOLUTION"]), 1e-3 * img.x, rtol=1e-12, atol=0)


@needs_features
@pytest.mark.parametrize("run", ["TEST_15668", "E300_15673"])
def test_band_stats_equal_features(run):
    """valley and band_stats against online/features.py on every 10th frame of each run, processed with the
    step's pedestal: the same cut and the same numbers, with the frame's own cut and with the cut 7 rows off."""
    files = sorted((XTCAV / run / "images/DTOTR2").glob("*.h5"))
    if not files:
        pytest.skip(f"needs the DAQ run {XTCAV / run}")
    spec = importlib.util.spec_from_file_location("features", FEATURES)
    features = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(features)
    x = screen.x_positions(30.5)
    found = 0
    for path in files:
        with h5py.File(path) as f:
            crops = np.stack([screen.trained_frame(r.ravel(), MODES["roi"]) for r in f["entry/data/data"][::10]])
        background = screen.pedestal(crops)
        for crop in crops:
            processed = screen.process(crop, background)
            cut, _ = features.valley_threshold(processed.sum(axis=1))
            assert screen.valley(processed.sum(axis=1)).cut == cut
            if cut is None:
                continue
            found += 1
            for c in (cut, cut - 7, cut + 7):
                ours, theirs = screen.band_stats(processed, x, c), features.band_stats(processed, x, cut_row=c)
                assert (ours is None) == (theirs is None)
                if ours is not None:
                    np.testing.assert_equal({k: ours[k] for k in theirs}, theirs)
    assert found > 100


@needs_daq
def test_backgrounds(replay, tmp_path, monkeypatch):
    monkeypatch.setattr(session, "DATA", tmp_path)
    data, frame_ids = daq()
    crops = np.stack([pil_crop(frame) for frame in data])
    replay("srv01")
    taken = session.take_background(seconds=6)
    assert np.array_equal(taken.image, crops.mean(axis=0))
    assert taken.record["camera"] == MODES["srv01"] and taken.record["frames"] == 30
    assert taken.record["frame_max"] == [int(c.max()) for c in crops]
    saved = np.load(taken.path)
    assert np.array_equal(saved["crops"], crops) and saved["pulse_id"].tolist() == frame_ids
    assert Path(taken.path).with_suffix(".png").stat().st_size > 0
    loaded = session.load_background(taken.path)
    assert np.array_equal(loaded.image, taken.image) and loaded.record == taken.record and loaded.path == taken.path

    replay("full")
    pedestal = session.pedestal(seconds=6)
    dark = np.median(crops[:, :184])
    assert (pedestal.image[:184] == dark).all() and (pedestal.image[184:] == 0).all()
    assert pedestal.record["fraction_above_threshold"] == np.mean(crops[:, :184] - dark >= 35)


@needs_daq
def test_peek(replay, tmp_path, monkeypatch):
    monkeypatch.setattr(session, "DATA", tmp_path)
    data, frame_ids = daq()
    background = session.Background(screen.pedestal(np.stack([pil_crop(frame) for frame in data])), "pedestal", {})
    replay("srv01")
    summary = session.peek(background, seconds=3)
    assert summary["frames"] > 1 and summary["pulse_id"] != frame_ids[0]      # the newest frame, not the first
    processed = screen.process(pil_crop(data[frame_ids.index(summary["pulse_id"])]), background.image)
    cut = screen.valley(processed.sum(axis=1)).cut
    assert summary["band_stats"] == (None if cut is None else screen.band_stats(processed, screen.x_positions(30.5), cut))
    [record] = tmp_path.glob("*_S3/peek_*.json")
    assert json.loads(record.read_text()) == summary
    assert record.with_suffix(".png").stat().st_size > 0


@needs_daq
def test_processing_test(replay, tmp_path, monkeypatch):
    monkeypatch.setattr(session, "DATA", tmp_path)
    data, frame_ids = daq()
    crops = np.stack([pil_crop(frame) for frame in data])
    backgrounds = [session.Background(screen.pedestal(crops), "pedestal", {}),
                   session.Background(np.zeros((300, 450)), "zeros", {}),
                   session.Background(np.full((300, 450), 5000.0), "above every pixel: no cut", {})]
    replay("full")
    summary = session.processing_test(backgrounds, seconds=6)
    assert summary["frames"] == 30 and summary["camera"] == MODES["full"]
    x = screen.x_positions(30.5)
    for background, got in zip(backgrounds, summary["backgrounds"]):
        processed = [screen.process(c, background.image) for c in crops]
        cuts = [screen.valley(p.sum(axis=1)).cut for p in processed]
        stats = [screen.band_stats(p, x, c) for p, c in zip(processed, cuts) if c is not None]
        assert got["background"] == background.path and got["cut_found"] == len(stats) / 30
        assert got["intensity_median"] == np.median([p.sum() for p in processed])
        if stats:
            assert got["median"]["x_d - x_w"] == np.median([s["x_d"] - s["x_w"] for s in stats])
            assert got["median"]["frac_d"] == np.median([s["frac_d"] for s in stats])
        else:
            assert got["median"] == {}
        assert 0 < got["processing_s"]["median"] <= got["processing_s"]["max"]
    assert [got["cut_found"] > 0 for got in summary["backgrounds"]] == [True, True, False]
    [record] = tmp_path.glob("*_S3/processing_test_*.json")
    assert json.loads(record.read_text()) == summary
    saved = np.load(record.with_suffix(".npz"))
    assert np.array_equal(saved["crops"], crops) and saved["pulse_id"].tolist() == frame_ids
