import copy

import numpy as np

from sahel import load_config, save_json, seed_everything
from sahel.adapt import Regional, pick_chips, test_time
from sahel.analysis import encoder_features
from sahel.data import source_split, target_split
from sahel.model import evaluate, fit, load

cfg = load_config()
names = cfg["classes"]
adapted, stats = load(cfg, cfg["models_dir"] / "adapted.pt")
(Xs, Ys), _ = source_split(cfg)
(Xt, Yt, meta_t), (Xa, Ya, meta_a) = target_split(cfg)
reg_t, reg_a = meta_t["region"].to_numpy(), meta_a["region"].to_numpy()
probs = test_time(adapted, Xt, reg_t, stats)
regional = Regional.fit(copy.deepcopy(adapted), Xt, reg_t, stats)
features = np.zeros((len(Xt), 512), np.float32)
for r in np.unique(reg_t):
    features[reg_t == r] = encoder_features(regional.use(r), Xt[reg_t == r], stats)
audit = lambda m: evaluate(Ya, test_time(m, Xa, reg_a, stats), names)

rows = [{"budget": 0, "strategy": "none", "seed": 0, "audit_miou": audit(adapted)["miou"]}]
for budget in cfg["label_budgets"]:
    for strategy, seeds in (("uncertain+diverse", [0]), ("random", [0, 1, 2])):
        for seed in seeds:
            seed_everything(seed)
            picked = pick_chips(probs, features, budget, strategy, np.random.default_rng(seed))
            reps = max(1, len(Xs) // (2 * budget))
            X = np.concatenate([Xs, np.repeat(Xt[picked], reps, axis=0)])
            Y = np.concatenate([Ys, np.repeat(Yt[picked], reps, axis=0)])
            res = audit(fit(copy.deepcopy(adapted), X, Y, stats, cfg, epochs=4, lr=1e-4))
            rows.append({"budget": budget, "strategy": strategy, "seed": seed, "audit_miou": res["miou"],
                         "audit_iou": res["iou"]})
            print(strategy, budget, seed, round(res["miou"], 3))
save_json({"rows": rows}, cfg["results_dir"] / "labels.json")
