"""pipeline/anya2/pose_backend.py: export shapes and the batch-1 guard."""
import pytest

from pipeline.anya2 import pose_backend as PB


def test_rect_shapes_match_ultralytics_letterbox():
    assert PB.rect_shape((540, 960), 640) == (384, 640)       # near proxy
    assert PB.rect_shape((150, 532), 960) == (288, 960)       # a far band
    assert PB.rect_shape((1080, 1920), 640) == (384, 640)


def test_fits_is_exact_only():
    # Extra padding was measured to change detections -- not allowed.
    assert PB.fits((384, 640), (540, 960), 640)
    assert not PB.fits((416, 640), (540, 960), 640)
    assert not PB.fits((640, 640), (540, 960), 640)


class _FakeYOLO:
    def __init__(self):
        self.calls = []

    def predict(self, src, imgsz=None, **kw):
        n = len(src) if isinstance(src, list) else 1
        self.calls.append(n)
        return [f"r{len(self.calls)}"]        # NCNN-like: ONE result per call


def test_non_batch_backend_splits_the_batch():
    m = _FakeYOLO()
    out = PB.Pose(m, (384, 640), False, "ncnn", "x").predict(["a", "b", "c"])
    assert len(out) == 3 and m.calls == [1, 1, 1]


def test_missing_export_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ANYA_POSE_MODELS", str(tmp_path))
    monkeypatch.setenv("ANYA_POSE_AUTO_EXPORT", "0")
    with pytest.raises(PB.MissingExport, match="384x640"):
        PB.load((540, 960), 640, kind="ncnn")


def test_export_dirs_are_found_by_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("ANYA_POSE_MODELS", str(tmp_path))
    d = tmp_path / PB.export_dir_name("ncnn", (384, 640))
    d.mkdir()
    (d / "model.ncnn.param").write_text("")
    hit = PB.find_ncnn((540, 960), 640)
    assert hit and hit[0] == (384, 640)
    assert PB.find_ncnn((150, 532), 960) is None
