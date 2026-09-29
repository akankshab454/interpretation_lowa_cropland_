import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import planetary_computer as pc
import rasterio
import torch
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform as warp_points
from rasterio.windows import from_bounds
from shapely.geometry import Point, shape
from torch.utils.data import DataLoader, Dataset

os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="5", VSI_CACHE="TRUE")
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
BAD_SCL = [0, 1, 3, 8, 9, 10]
RES, SCALE = 10, 1e4


def best_scenes(cat, bbox, dates, max_cloud):
    items = cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=dates).item_collection()
    items = [i for i in items if i.properties["eo:cloud_cover"] < max_cloud
             and i.properties.get("s2:nodata_pixel_percentage", 0) < 10]
    best = {}
    for i in sorted(items, key=lambda i: i.properties["eo:cloud_cover"]):
        best.setdefault(i.properties["s2:mgrs_tile"], i)
    return list(best.values())


def epsg(item):
    return "EPSG:" + str(item.properties.get("proj:epsg") or item.properties["proj:code"]).split(":")[-1]


def read(href, bounds, size, resampling):
    with rasterio.open(href) as src:
        return src.read(1, window=from_bounds(*bounds, transform=src.transform), out_shape=(size, size),
                        resampling=resampling, boundless=True, fill_value=0)


def read_label(href, bounds, crs, size):
    with rasterio.open(href) as src, WarpedVRT(src, crs=crs, width=size, height=size, resampling=Resampling.nearest,
                                               transform=from_origin(bounds[0], bounds[3], RES, RES)) as vrt:
        return vrt.read(1)


def read_bands(scene, bounds, size, bands):
    x = np.stack([read(scene.assets[b].href, bounds, size, Resampling.bilinear) for b in bands]).astype(np.int32)
    if float(scene.properties.get("s2:processing_baseline", "0")) >= 4.0:
        x = np.clip(x - 1000, 0, None)
    return x.astype(np.uint16)


def read_chip(scene, wc_href, bounds, cfg, lut):
    scl = read(scene.assets["SCL"].href, bounds, cfg["chip"], Resampling.nearest)
    if np.isin(scl, BAD_SCL).mean() > cfg["max_chip_bad"]:
        return None
    y = lut[read_label(wc_href, bounds, epsg(scene), cfg["chip"])]
    return None if (y == 255).mean() > 0.5 else (read_bands(scene, bounds, cfg["chip"], cfg["bands"]), y)


def cluster_centers(bbox, n, n_blocks, rng, min_sep=0.12):
    lon0, lat0, lon1, lat1 = bbox
    strip, pts = (lon1 - lon0) / n_blocks, []
    for _ in range(n * 200):
        if len(pts) == n:
            break
        block = len(pts) % n_blocks
        lon = rng.uniform(lon0 + block * strip + 0.06, lon0 + (block + 1) * strip - 0.06)
        lat = rng.uniform(lat0 + 0.06, lat1 - 0.06)
        if all(np.hypot(lon - p[0], lat - p[1]) > min_sep for p in pts):
            pts.append((lon, lat, block))
    return pts


def fetch_region(name, region, cfg, seed=0):
    cat = Client.open(STAC, modifier=pc.sign_inplace)
    scenes = best_scenes(cat, region["bbox"], region["dates"], cfg["max_scene_cloud"])
    wc_items = cat.search(collections=["esa-worldcover"], bbox=region["bbox"], datetime="2021-01-01/2021-12-31").item_collection()
    lut = np.full(256, 255, np.uint8)
    for code, cls in cfg["worldcover_map"].items():
        lut[int(code)] = cls
    jobs, span, n = [], cfg["chip"] * RES, cfg["cluster"]
    for cid, (lon, lat, block) in enumerate(cluster_centers(region["bbox"], region["clusters"], cfg["n_blocks"],
                                                            np.random.default_rng(seed))):
        scene = next((s for s in scenes if shape(s.geometry).contains(Point(lon, lat))), None)
        wc = next((w for w in wc_items if shape(w.geometry).contains(Point(lon, lat))), None)
        if scene is None or wc is None:
            continue
        xs, ys = warp_points("EPSG:4326", epsg(scene), [lon], [lat])
        left, top = np.floor(xs[0] / RES) * RES - span * n / 2, np.floor(ys[0] / RES) * RES + span * n / 2
        for i in range(n):
            for j in range(n):
                b = (left + j * span, top - (i + 1) * span, left + (j + 1) * span, top - i * span)
                jobs.append((scene, wc.assets["map"].href, b, dict(region=name, cluster=cid, block=block,
                                                                     lon=lon, lat=lat, scene=scene.id)))

    def run(job):
        try:
            return read_chip(*job[:3], cfg, lut), job[3]
        except Exception as err:
            print("skip:", err)
            return None, job[3]

    with ThreadPoolExecutor(8) as pool:
        kept = [(r, m) for r, m in pool.map(run, jobs) if r is not None]
    np.savez_compressed(cfg["data_dir"] / f"{name}.npz", X=np.stack([r[0] for r, _ in kept]),
                        Y=np.stack([r[1] for r, _ in kept]))
    pd.DataFrame([m for _, m in kept]).to_csv(cfg["data_dir"] / f"{name}.csv", index=False)
    print(f"{name}: kept {len(kept)}/{len(jobs)} chips from {len(scenes)} scenes")


def fetch_scene(lon, lat, dates, cfg, path, size=1024):
    cat = Client.open(STAC, modifier=pc.sign_inplace)
    scene = best_scenes(cat, [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01], dates, cfg["max_scene_cloud"])[0]
    xs, ys = warp_points("EPSG:4326", epsg(scene), [lon], [lat])
    left, top = np.floor(xs[0] / RES) * RES - size * RES / 2, np.floor(ys[0] / RES) * RES + size * RES / 2
    x = read_bands(scene, (left, top - size * RES, left + size * RES, top), size, cfg["bands"])
    with rasterio.open(path, "w", driver="GTiff", width=size, height=size, count=len(x), dtype="uint16",
                       crs=epsg(scene), transform=from_origin(left, top, RES, RES), compress="deflate") as dst:
        dst.write(x)


def load_domain(cfg, names, blocks=None):
    parts = [(np.load(cfg["data_dir"] / f"{n}.npz"), pd.read_csv(cfg["data_dir"] / f"{n}.csv")) for n in names]
    X = np.concatenate([p[0]["X"] for p in parts])
    Y = np.concatenate([p[0]["Y"] for p in parts])
    meta = pd.concat([p[1] for p in parts], ignore_index=True)
    if blocks is not None:
        keep = meta["block"].isin(blocks).to_numpy()
        X, Y, meta = X[keep], Y[keep], meta[keep].reset_index(drop=True)
    return X, Y, meta


def regions_with(cfg, role):
    return [n for n, r in cfg["regions"].items() if r["role"] == role]


def source_split(cfg):
    X, Y, meta = load_domain(cfg, regions_with(cfg, "source"))
    test = meta["block"].to_numpy() == cfg["n_blocks"] - 1
    return (X[~test], Y[~test]), (X[test], Y[test])


def target_split(cfg):
    last, names = cfg["n_blocks"] - 1, regions_with(cfg, "target")
    return load_domain(cfg, names, list(range(last))), load_domain(cfg, names, [last])


def band_stats(X):
    r = X.astype(np.float32) / SCALE
    return r.mean(axis=(0, 2, 3)), r.std(axis=(0, 2, 3)) + 1e-6


class Chips(Dataset):
    def __init__(self, X, Y, stats, augment=False, gain_to=None):
        self.X, self.Y, self.augment = X, Y, augment
        self.gain_to = None if gain_to is None else torch.tensor(gain_to, dtype=torch.float32).view(-1, 1, 1)
        self.mean, self.std = (torch.tensor(s).view(-1, 1, 1) for s in stats)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i].astype(np.float32)) / SCALE
        y = torch.from_numpy(self.Y[i].astype(np.int64))
        if self.augment:
            k = int(torch.randint(4, ()))
            x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
            if torch.rand(()) < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            x = x * torch.empty(len(x), 1, 1).uniform_(0.9, 1.1) + torch.empty(len(x), 1, 1).uniform_(-0.01, 0.01)
            if self.gain_to is not None:
                x = x * (1 + torch.rand(()) * (self.gain_to - 1))
        return (x - self.mean) / self.std, y


def loader(X, stats, batch, Y=None, train=False, gain_to=None):
    Y = np.zeros((len(X), 1, 1), np.uint8) if Y is None else Y
    return DataLoader(Chips(X, Y, stats, train, gain_to), batch_size=batch, shuffle=train, drop_last=train)
