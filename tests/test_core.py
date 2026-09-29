import numpy as np
import torch

from sahel.adapt import class_thresholds, fda
from sahel.model import confusion, scores
from sahel.predict import tiles


def test_iou_by_hand():
    s = scores(confusion(np.array([0, 0, 1, 1, 0, 1]), np.array([0, 0, 0, 1, 1, 255]), 2), ["a", "b"])
    assert np.isclose(s["iou"]["a"], 0.5) and np.isclose(s["iou"]["b"], 1 / 3, atol=1e-4)
    assert np.isclose(s["oa"], 0.6)


def test_absent_class_convention():
    s = scores(confusion(np.r_[np.zeros(9, int), 1], np.zeros(10, int), 3), ["a", "b", "c"])
    assert s["iou"]["b"] == 0 and np.isnan(s["iou"]["c"]) and np.isclose(s["miou"], 0.45)


def test_fda_takes_target_mean():
    src, tgt = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64) + 5
    assert torch.allclose(fda(src, tgt, 0.05).mean((2, 3)), tgt.mean((2, 3)), atol=0.05)


def test_thresholds_survive_float16():
    probs = np.random.default_rng(0).dirichlet(np.ones(3), (4, 200, 200)).transpose(0, 3, 1, 2).astype(np.float16)
    assert np.isfinite(class_thresholds(probs, 3, 0.5)).all()


def test_tiles_cover_image():
    covered = np.zeros((600, 700), bool)
    for y, x in tiles(600, 700, 256, 128):
        covered[y:y + 256, x:x + 256] = True
    assert covered.all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
