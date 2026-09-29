import numpy as np

from sahel import load_config, load_json, plots
from sahel.data import target_split
from sahel.model import confusion, load, predict, scores

cfg = load_config()
res, names = cfg["results_dir"], cfg["classes"]
fig = res / "figures"
fig.mkdir(exist_ok=True)
base, diag = load_json(res / "source_baseline.json"), load_json(res / "diagnosis.json")
abl, labels = load_json(res / "ablation.json"), load_json(res / "labels.json")["rows"]
rows = abl["rows"]
methods = list(dict.fromkeys(r["method"] for r in rows))
mean = lambda m, k="audit_miou": float(np.mean([r[k] for r in rows if r["method"] == m]))
first = lambda m: next(r["audit_iou"] for r in rows if r["method"] == m)

_, (Xa, Ya, meta_a) = target_split(cfg)
majority = scores(confusion(np.full(Ya.size, names.index("cropland")), Ya.ravel(), len(names)), names)
best = max((r for r in labels if r["strategy"] == "uncertain+diverse"), key=lambda r: r["budget"])

plots.staircase([("Iowa, random split", base["summary"]["random"]["mean"]),
                 ("Iowa, block split", base["summary"]["block"]["mean"]),
                 ("Sahel, source only", mean("source only")),
                 ("+ AdaBN per region", mean("AdaBN per region")),
                 ("+ radiometric aug", mean("+ radiometric aug")),
                 ("+ self-training + FDA", mean("+ self-train + FDA")),
                 (f"+ {best['budget']} labelled chips", best["audit_miou"])], majority["miou"], fig / "staircase.png")
plots.shift(diag, fig / "shift.png")
plots.season_geo(diag["season_geo"], fig / "season_geo.png")
plots.active_learning(labels, fig / "labels.png")
plots.proxies(rows, abl["proxy_rho"], fig / "proxies.png")
plots.per_class({"Iowa test": base["runs"][-1]["source_iou"], "Sahel, source only": first("source only"),
                 "Sahel, adapted": first("+ self-train + FDA"),
                 f"Sahel, +{best['budget']} labels": best["audit_iou"]}, names, fig / "per_class.png")

pick = np.concatenate([np.random.default_rng(0).choice(np.flatnonzero(meta_a["region"] == r), 2, replace=False)
                       for r in meta_a["region"].unique()])
source, stats = load(cfg, cfg["models_dir"] / "source.pt")
preds = [predict(source, Xa[pick], stats).argmax(1), np.load(res / "adapted_audit_probs.npy")[pick].argmax(1)]
plots.qualitative(Xa[pick], Ya[pick], preds, ["source only", "adapted"], fig / "qualitative.png")

lines = ["| Method | mIoU | OA | fwIoU | cropland IoU | grass/shrub IoU | ECE | Iowa mIoU |", "|---" * 8 + "|",
         f"| all-cropland map | {majority['miou']:.3f} | {majority['oa']:.3f} | {majority['fwiou']:.3f} | "
         f"{majority['iou']['cropland']:.2f} | 0.00 | - | - |"]
for m in methods:
    sel = [r for r in rows if r["method"] == m]
    iou = lambda c: np.mean([r["audit_iou"][c] for r in sel])
    lines.append(f"| {m} | {mean(m):.3f} | {mean(m, 'audit_oa'):.3f} | {mean(m, 'audit_fwiou'):.3f} | "
                 f"{iou('cropland'):.2f} | {iou('grass_shrub'):.2f} | {mean(m, 'audit_ece'):.2f} | {mean(m, 'iowa_miou'):.3f} |")
lines.append(f"| + {best['budget']} labelled chips | {best['audit_miou']:.3f} | | | "
             f"{best['audit_iou']['cropland']:.2f} | {best['audit_iou']['grass_shrub']:.2f} | | |")
(res / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
