import copy

import numpy as np
from scipy.stats import spearmanr

from sahel import load_config, save_json
from sahel.adapt import Regional, adabn, entropy, test_time
from sahel.analysis import band_shift, ndvi
from sahel.data import load_domain, source_split, target_split
from sahel.model import evaluate, load, predict

cfg = load_config()
names = cfg["classes"]
model, stats = load(cfg, cfg["models_dir"] / "source.pt")
(Xs, Ys), (Xs_test, Ys_test) = source_split(cfg)
(Xt, Yt, meta_t), (Xa, Ya, meta_a) = target_split(cfg)
reg_t, reg_a = meta_t["region"].to_numpy(), meta_a["region"].to_numpy()
out = {"band_shift": band_shift(Xs, Xt, cfg["bands"], np.random.default_rng(0))}

bins = np.linspace(-0.2, 0.9, 45)
out["ndvi_hist"] = {"bins": bins.tolist()}
for key, X, Y, c in [("iowa_cropland", Xs, Ys, "cropland"), ("sahel_cropland", Xt, Yt, "cropland"),
                     ("sahel_grass_shrub", Xt, Yt, "grass_shrub")]:
    out["ndvi_hist"][key] = np.histogram(ndvi(X)[Y == names.index(c)], bins, density=True)[0].tolist()

out["ndvi_median"] = {}
for region in cfg["regions"]:
    X, Y, _ = load_domain(cfg, [region])
    v = ndvi(X)
    out["ndvi_median"][region] = {c: float(np.median(v[Y == i])) for i, c in enumerate(names) if (Y == i).sum() > 5000}

probs = {"source_only": predict(model, Xa, stats),
         "adabn_other_scenes": Regional.fit(copy.deepcopy(model), Xt, reg_t, stats).predict(Xa, reg_a, stats),
         "adabn_test_time": test_time(model, Xa, reg_a, stats)}
out["bn_swap"] = {k: {**evaluate(Ya, p, names), "entropy": float(entropy(p).mean())} for k, p in probs.items()}
print({k: round(v["miou"], 3) for k, v in out["bn_swap"].items()})

acc = ((probs["source_only"].argmax(1) == Ya) | (Ya == 255)).mean(axis=(1, 2))
out["chip_rho"] = {"ndvi": float(spearmanr(ndvi(Xa).mean(axis=(1, 2)), acc)[0]),
                   "brightness": float(spearmanr(Xa[:, :3].mean(axis=(1, 2, 3)), acc)[0])}

cells = {"iowa_summer": (Xs_test, Ys_test), **{n: load_domain(cfg, [n])[:2] for n in ("iowa_autumn", "niger_dry", "niger_wet")}}
out["season_geo"] = {n: {"source_only": evaluate(Y, predict(model, X, stats), names)["miou"],
                         "adabn": evaluate(Y, predict(adabn(copy.deepcopy(model), X, stats), X, stats), names)["miou"]}
                     for n, (X, Y) in cells.items()}
print(out["season_geo"], out["chip_rho"])
save_json(out, cfg["results_dir"] / "diagnosis.json")
