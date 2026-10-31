"""S2: the hand crop cut from any camera ROI that contains it, oriented and cropped like the DAQ images.

The comparisons with Yiheng's DAQ classes need his code and the DAQ files, so they run only where both are
(next to this repo) and are skipped in CI, like the soft-IOC tests (conftest.py).
"""
import json
import sys

import numpy as np
import pytest

from conftest import DAQ, MODES, XTCAV, YIHENG, camera, daq, needs_daq, needs_yiheng, pil_crop
from twobunch_deploy import machine, pvs, screen, session

# the camera ROIs of the E331 runs (metadata.DTOTR2)
E331 = [(372, 108, 256, 830), (382, 124, 238, 786), (394, 112, 208, 826)]


def test_orient():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    assert screen.orient(a, "Positive", "Positive", 0).tolist() == [[1, 2, 3], [4, 5, 6]]
    assert screen.orient(a, "Negative", "Positive", 0).tolist() == [[3, 2, 1], [6, 5, 4]]
    assert screen.orient(a, "Positive", "Negative", 0).tolist() == [[4, 5, 6], [1, 2, 3]]
    assert screen.orient(a, "Negative", "Negative", 0).tolist() == [[6, 5, 4], [3, 2, 1]]
    assert screen.orient(a, "Positive", "Positive", 1).tolist() == [[1, 4], [2, 5], [3, 6]]
    with pytest.raises(ValueError):
        screen.orient(a, "Positive", "Pos", 0)


def test_trained_frame():
    # each pixel of the sensor holds 10000 y + x, so the result shows which sensor pixels it took
    sensor = (10000 * np.arange(964)[:, None] + np.arange(1292)).astype("int32")     # int32: PIL takes it
    expected = np.zeros((300, 450), int)
    i, j = np.mgrid[0:184, 0:450]
    expected[:184] = 10000 * (270 + j) + 396 + i     # transposed: row i is sensor column 396 + i
    assert np.array_equal(pil_crop(sensor[70:964, 396:580]), expected)

    def cut(c, **changes):
        x, y, w, h = c
        return screen.trained_frame(sensor[y:y + h, x:x + w].ravel(), camera(x, y, w, h, **changes))

    # the camera at the full sensor, the trained ROI, the ROI on srv01 (10-30) and the E331 ROIs
    for c in [(0, 0, 1292, 964), (396, 70, 184, 894), (346, 38, 324, 860), *E331]:
        assert np.array_equal(cut(c), expected)
    # one pixel short of the crop: its first column, its last row
    for c in [(397, 0, 895, 964), (0, 0, 1292, 719)]:
        with pytest.raises(ValueError, match="does not contain the hand crop"):
            cut(c)
    # not rotated, the trained ROI is 894 x 184 and the crop doesn't fit
    with pytest.raises(ValueError, match="does not contain the hand crop"):
        cut((0, 0, 1292, 964), IS_ROTATED=0)
    with pytest.raises(ValueError, match="not the driver"):
        cut((0, 0, 1292, 964), NDArrayPort_RBV="ROI107")
    for reverse in ["ReverseX_RBV", "ReverseY_RBV"]:
        with pytest.raises(ValueError, match="reverses"):
            cut((0, 0, 1292, 964), **{reverse: "Yes"})
    # binned 2 x 2 by the driver: the array is half the ROI's size
    for sizes in [dict(ArraySizeX_RBV=646, ArraySizeY_RBV=482), dict(ArraySize0_RBV=646, ArraySize1_RBV=482)]:
        with pytest.raises(ValueError, match="not the size of the camera ROI"):
            screen.trained_frame(sensor[::2, ::2].ravel(), camera(0, 0, 1292, 964, **sizes))


@needs_yiheng
def test_orient_matches_daq_rule():
    sys.path.insert(0, str(YIHENG))
    import image
    a = np.arange(12).reshape(3, 4)
    for x in ("Positive", "Negative"):
        for y in ("Positive", "Negative"):
            for rotated in (0, 1):
                assert np.array_equal(screen.orient(a, x, y, rotated), image.orientImage(a, x, y, rotated))


@needs_yiheng
@needs_daq
def test_trained_frame_equals_daq_classes():
    """The DAQ classes' hand-cropped TEST_15668 images, from the camera at three ROIs."""
    import h5py
    sys.path.insert(0, str(YIHENG))
    import dataset
    ds = dataset.DATASET(str(XTCAV), "TEST", "15668")
    cam = ds.images["DTOTR2"]
    for ind in (0, 1234, ds.N - 1):
        img = ds.getImage("DTOTR2", ind)
        img.crop(screen.HAND_CROP)
        pid = cam["pid"][cam["common_index"][ind] - 1]
        with h5py.File(DAQ.parent / "DTOTR2" / img.filename) as f:
            [k] = np.flatnonzero(f["entry/instrument/NDAttributes/NDArrayUniqueId"][:] == pid)
            raw = f["entry/data/data"][k]
        sensor = np.full((964, 1292), 31, raw.dtype)
        sensor[70:964, 396:580] = raw
        for c in MODES.values():
            x, y, w, h = c["MinX_RBV"], c["MinY_RBV"], c["SizeX_RBV"], c["SizeY_RBV"]
            assert np.array_equal(screen.trained_frame(sensor[y:y + h, x:x + w].ravel(), c), img.data)


def test_get_unlisted_pv():
    with pytest.raises(ValueError, match="TEST:SOFT_IOC:UNLISTED is not in pvs.READ"):
        machine.get("TEST:SOFT_IOC:UNLISTED")


@needs_daq
def test_get_missing_pv(replay, monkeypatch):
    monkeypatch.setitem(pvs.READ, "TEST:SOFT_IOC:MISSING", "a PV no IOC serves")
    with pytest.raises(TimeoutError, match="no answer from TEST:SOFT_IOC:MISSING"):
        machine.get("TEST:SOFT_IOC:MISSING")


@needs_daq
@pytest.mark.parametrize("mode", MODES)
def test_geometry_test(replay, tmp_path, monkeypatch, mode):
    monkeypatch.setattr(session, "DATA", tmp_path)
    data, frame_ids = daq()
    replay(mode)
    summary = session.geometry_test(seconds=6)
    assert summary["camera"] == MODES[mode] and summary["trained"] == screen.TRAINED
    assert summary["differs"] == ([] if mode == "roi" else ["MinX_RBV", "MinY_RBV", "SizeX_RBV", "SizeY_RBV"])
    assert summary["frames"] == 30 and summary["trained_shape"] == [300, 450]

    [record] = tmp_path.glob("*_S2/geometry_test_*.json")
    assert json.loads(record.read_text()) == summary
    saved = np.load(record.with_suffix(".npz"))
    assert saved["pulse_id"].tolist() == frame_ids
    assert np.array_equal(saved["trained"], [pil_crop(frame) for frame in data])
    assert saved["raw"].size == MODES[mode]["SizeX_RBV"] * MODES[mode]["SizeY_RBV"]
    assert record.with_suffix(".png").stat().st_size > 0
