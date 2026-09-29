import numpy as np

from sahel import load_config, save_json, seed_everything
from sahel.data import band_stats, load_domain, regions_with, target_split
from sahel.model import build, evaluate, fit, predict, save

cfg = load_config()
names = cfg["classes"]
X, Y, meta = load_domain(cfg, regions_with(cfg, "source"))
_, (Xa, Ya, _) = target_split(cfg)

rows = []
for fold in range(cfg["n_blocks"]):
    test = np.flatnonzero(meta["block"] == fold)
    perm = np.random.default_rng(fold).permutation(len(X))
    splits = {"random": (perm[len(test):], perm[:len(test)]), "block": (np.flatnonzero(meta["block"] != fold), test)}
    for split, (tr, te) in splits.items():
        seed_everything(cfg["seed"] + fold)
        stats = band_stats(X[tr])
        model = fit(build(cfg), X[tr], Y[tr], stats, cfg)
        res = evaluate(Y[te], predict(model, X[te], stats), names)
        row = {"split": split, "fold": fold, "source_miou": res["miou"], "source_iou": res["iou"]}
        if split == "block":
            row["sahel_miou"] = evaluate(Ya, predict(model, Xa, stats), names)["miou"]
        print(row["split"], fold, round(row["source_miou"], 3), round(row.get("sahel_miou", 0), 3))
        rows.append(row)
save(model, stats, cfg["models_dir"] / "source.pt")

summary = {s: {"mean": float(np.mean([r["source_miou"] for r in rows if r["split"] == s])),
               "std": float(np.std([r["source_miou"] for r in rows if r["split"] == s]))} for s in splits}
summary["sahel"] = float(np.mean([r["sahel_miou"] for r in rows if "sahel_miou" in r]))
print(summary)
save_json({"summary": summary, "runs": rows}, cfg["results_dir"] / "source_baseline.json")
