import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

CLASS_COLORS = ["#006400", "#ffbb22", "#f096ff", "#fa0000", "#b4b4b4", "#0064c8"]
WAVELENGTH = [490, 560, 665, 705, 740, 783, 842, 865]
IOWA, SAHEL, ADAPTED, LABELS, GREY = "#2b6cb0", "#c05621", "#2f855a", "#6b46c1", "#718096"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def staircase(steps, baseline, path):
    labels, values = zip(*steps)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.axhline(baseline, color=GREY, ls="--", lw=1, label=f"all-cropland map ({baseline:.3f})")
    ax.bar(range(len(values)), values, width=0.65,
           color=[IOWA, IOWA, SAHEL] + [ADAPTED] * (len(values) - 4) + [LABELS])
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.2f}" + ("" if i == 0 else f"\n({v - values[i - 1]:+.2f})"), ha="center", fontsize=8)
    ax.set_xticks(range(len(values)), labels, rotation=15, ha="right")
    ax.set(ylabel="mIoU", ylim=(0, max(values) * 1.25))
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    save(fig, path)


def shift(diag, path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    b = diag["band_shift"].values()
    axes[0].plot(WAVELENGTH, [v["source_mean"] for v in b], "o-", color=IOWA, label="Iowa, Jul-Aug")
    axes[0].plot(WAVELENGTH, [v["target_mean"] for v in b], "o-", color=SAHEL, label="Sahel, Jan-Mar")
    axes[0].set(xlabel="wavelength (nm)", ylabel="mean reflectance", title="Mean spectrum")
    h = diag["ndvi_hist"]
    mid = (np.array(h["bins"][1:]) + h["bins"][:-1]) / 2
    for key, color in [("iowa_cropland", IOWA), ("sahel_cropland", SAHEL), ("sahel_grass_shrub", GREY)]:
        axes[1].plot(mid, h[key], color=color, label=key.replace("_", " "))
    axes[1].set(xlabel="NDVI", title="Is Sahel cropland still green?")
    for ax in axes:
        ax.legend(fontsize=8)
    save(fig, path)


def season_geo(cells, path):
    order = [["iowa_summer", "iowa_autumn"], ["niger_wet", "niger_dry"]]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for ax, key in zip(axes, ["source_only", "adabn"]):
        grid = np.array([[cells[n][key] for n in row] for row in order])
        ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=grid.max())
        for (i, j), v in np.ndenumerate(grid):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11)
        ax.set_xticks([0, 1], ["green season", "dry / post-harvest"])
        ax.set_yticks([0, 1], ["Iowa", "Niger"])
        ax.set_title("source model" + (" + AdaBN" if key == "adabn" else ""))
        ax.grid(False)
    save(fig, path)


def per_class(series, classes, path):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    w = 0.8 / len(series)
    for i, ((label, iou), c) in enumerate(zip(series.items(), [IOWA, SAHEL, ADAPTED, LABELS])):
        ax.bar(np.arange(len(classes)) + i * w, [np.nan_to_num(iou[k]) for k in classes], w, label=label, color=c)
    ax.set_xticks(np.arange(len(classes)) + w * (len(series) - 1) / 2, classes)
    ax.set_ylabel("IoU")
    ax.legend(fontsize=8, ncol=2)
    save(fig, path)


def active_learning(rows, path):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    base = rows[0]["audit_miou"]
    for strategy, color in [("uncertain+diverse", LABELS), ("random", GREY)]:
        budgets = sorted({r["budget"] for r in rows if r["strategy"] == strategy})
        vals = [[r["audit_miou"] for r in rows if r["strategy"] == strategy and r["budget"] == b] for b in budgets]
        ax.errorbar([0] + budgets, [base] + [np.mean(v) for v in vals], yerr=[0] + [np.std(v) for v in vals],
                    fmt="o-", color=color, capsize=3, label=strategy)
    ax.set(xlabel="labelled Sahel chips", ylabel="Sahel audit mIoU")
    ax.legend(fontsize=8)
    save(fig, path)


def proxies(rows, rho, path):
    fig, axes = plt.subplots(1, len(rho), figsize=(3 * len(rho), 2.8), sharey=True)
    for ax, p in zip(axes, rho):
        ax.scatter([r[p] for r in rows], [r["audit_miou"] for r in rows], color=SAHEL, s=18)
        ax.set(xlabel=p.replace("_", " "), title=f"Spearman rho = {rho[p]:+.2f}")
    axes[0].set_ylabel("audit mIoU")
    save(fig, path)


def rgb(x):
    img = x[[2, 1, 0]].transpose(1, 2, 0).astype(np.float32)
    lo, hi = np.percentile(img, [2, 98])
    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)


def qualitative(X, Y, preds, titles, path):
    cmap, cols = ListedColormap(CLASS_COLORS), 2 + len(preds)
    fig, axes = plt.subplots(len(X), cols, figsize=(2.2 * cols, 2.2 * len(X)))
    for i in range(len(X)):
        for j, im in enumerate([rgb(X[i]), np.ma.masked_equal(Y[i], 255)] + [p[i] for p in preds]):
            axes[i, j].imshow(im, cmap=None if j == 0 else cmap, vmin=0, vmax=5, interpolation="nearest")
            axes[i, j].set(xticks=[], yticks=[])
            axes[i, j].grid(False)
            if i == 0:
                axes[i, j].set_title((["RGB", "WorldCover"] + titles)[j], fontsize=9)
    save(fig, path)
