import argparse
import json

import numpy as np
import rasterio

from . import load_config, load_json
from .adapt import adabn
from .analysis import DriftMonitor
from .model import load, predict


def tiles(h, w, size, stride):
    ys, xs = list(range(0, h - size + 1, stride)), list(range(0, w - size + 1, stride))
    return [(y, x) for y in ys + [h - size] * (ys[-1] != h - size) for x in xs + [w - size] * (xs[-1] != w - size)]


def sliding_window(model, image, stats, size=256):
    _, h, w = image.shape
    offsets = tiles(h, w, size, size // 2)
    chips = np.stack([image[:, y:y + size, x:x + size] for y, x in offsets])
    probs = predict(model, chips, stats)
    total, count = np.zeros((probs.shape[1], h, w), np.float32), np.zeros((h, w), np.float32)
    for p, (y, x) in zip(probs, offsets):
        total[:, y:y + size, x:x + size] += p
        count[y:y + size, x:x + size] += 1
    return total / count, chips, probs


def run(tif, ckpt, monitor, out, cfg, scene_adabn=False):
    model, stats = load(cfg, ckpt)
    with rasterio.open(tif) as src:
        image, profile = src.read(), src.profile
    s = cfg["chip"]
    if scene_adabn:
        adabn(model, np.stack([image[:, y:y + s, x:x + s] for y, x in tiles(*image.shape[1:], s, s)]), stats)
    probs, chips, chip_probs = sliding_window(model, image, stats, s)
    with rasterio.open(out, "w", driver="COG", width=profile["width"], height=profile["height"], count=1,
                       dtype="uint8", crs=profile["crs"], transform=profile["transform"], compress="deflate") as dst:
        dst.write(probs.argmax(0).astype(np.uint8), 1)
    report = DriftMonitor(load_json(monitor)).check(chips, chip_probs)
    share = np.bincount(probs.argmax(0).ravel(), minlength=len(cfg["classes"])) / probs[0].size
    report["class_share"] = dict(zip(cfg["classes"], np.round(share, 4).tolist()))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tif")
    ap.add_argument("--ckpt", default="models/adapted.pt")
    ap.add_argument("--monitor", default="results/monitor_adapted.json")
    ap.add_argument("--out", default="results/prediction.tif")
    ap.add_argument("--scene-adabn", action="store_true")
    a = ap.parse_args()
    print(json.dumps(run(a.tif, a.ckpt, a.monitor, a.out, load_config(), a.scene_adabn), indent=2))
