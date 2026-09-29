import copy

import numpy as np
from scipy.stats import spearmanr

from sahel import load_config, save_json, seed_everything
from sahel.adapt import Regional, adabn, entropy, proxies, self_train, tent, test_time
from sahel.analysis import DriftMonitor
from sahel.data import band_stats, source_split, target_split
from sahel.model import build, evaluate, fit, load, predict, save

cfg = load_config()
names = cfg["classes"]
source, stats = load(cfg, cfg["models_dir"] / "source.pt")
(Xs, Ys), (Xs_test, Ys_test) = source_split(cfg)
(Xt, Yt, meta_t), (Xa, Ya, meta_a) = target_split(cfg)
reg_t, reg_a = meta_t["region"].to_numpy(), meta_a["region"].to_numpy()
prior = np.bincount(Yt[Yt != 255].ravel(), minlength=len(names))


def score(name, predict_fn, weights, seed=0, extra=None):
    probs = predict_fn(Xa, reg_a)
    audit = evaluate(Ya, probs, names)
    iowa = adabn(copy.deepcopy(weights), Xs, stats)
    row = {"method": name, "seed": seed, **{f"audit_{k}": v for k, v in audit.items()},
           "iowa_miou": evaluate(Ys_test, predict(iowa, Xs_test, stats), names)["miou"],
           **proxies(lambda X: predict_fn(X, reg_t), Xt, prior), **(extra or {})}
    print(f"{name:<32} seed {seed}  audit {row['audit_miou']:.3f}  iowa {row['iowa_miou']:.3f}")
    return row, probs


def band_standardised(X, regions):
    out = np.empty((len(X), len(names)) + X.shape[2:], np.float16)
    for r in np.unique(regions):
        out[regions == r] = predict(source, X[regions == r], band_stats(Xt[reg_t == r]))
    return out


def with_tent(X, regions):
    fn = lambda m, Xr, s: tent(adabn(m, Xr, s), Xr, s, cfg["adapt"]["tent_steps"])
    return Regional.fit(copy.deepcopy(source), X, regions, stats, fn, affine=True).predict(X, regions, stats)


at_test_time = lambda m, pooled=False: lambda X, r: test_time(m, X, np.zeros(len(X)) if pooled else r, stats)
gain_to = band_stats(Xt)[0] / band_stats(Xs)[0]
seed_everything(0)
augmented = fit(build(cfg), Xs, Ys, stats, cfg, gain_to=gain_to)

rows = [score("source only", lambda X, r: predict(source, X, stats), source)[0],
        score("per-region band standardisation", band_standardised, source)[0],
        score("AdaBN, one 'Sahel'", at_test_time(source, pooled=True), source)[0],
        score("AdaBN per region", at_test_time(source), source)[0],
        score("AdaBN per region + TENT", with_tent, source)[0],
        score("+ radiometric aug", at_test_time(augmented), augmented)[0]]
for seed in range(2):
    for use_fda in (False, True):
        seed_everything(seed)
        adapted, history = self_train(copy.deepcopy(augmented), Xs, Ys, Xt, reg_t, stats, cfg, use_fda, seed, gain_to)
        row, probs = score("+ self-train" + " + FDA" * use_fda, at_test_time(adapted.model), adapted.model, seed,
                           {"history": history})
        rows.append(row)
        if use_fda and seed == 0:
            save(adapted.model, stats, cfg["models_dir"] / "adapted.pt", adapted.states)
            final, final_probs = adapted.model, probs

keys = ["entropy", "confident_share", "rotation_agreement", "prior_gap"]
proxy_rho = {k: float(spearmanr([r[k] for r in rows], [r["audit_miou"] for r in rows])[0]) for k in keys}
chip_acc = ((final_probs.argmax(1) == Ya) | (Ya == 255)).mean(axis=(1, 2))
chip_rho = float(spearmanr(entropy(final_probs).mean(axis=(1, 2)), chip_acc)[0])
print(proxy_rho, chip_rho)

np.save(cfg["results_dir"] / "adapted_audit_probs.npy", final_probs)
save_json(DriftMonitor.calibrate(Xs_test, predict(source, Xs_test, stats)).ref, cfg["results_dir"] / "monitor_source.json")
save_json(DriftMonitor.calibrate(Xt, test_time(final, Xt, reg_t, stats)).ref, cfg["results_dir"] / "monitor_adapted.json")
save_json({"rows": rows, "proxy_rho": proxy_rho, "chip_entropy_rho": chip_rho}, cfg["results_dir"] / "ablation.json")
