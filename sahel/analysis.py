import numpy as np
import torch
from scipy.stats import ks_2samp

from .adapt import entropy, prior_gap
from .data import SCALE, loader
from .model import DEVICE

RED, NIR = 2, 6


def band_shift(Xs, Xt, bands, rng, n=100_000):
    pick = lambda X: X.transpose(1, 0, 2, 3).reshape(X.shape[1], -1)[:, rng.choice(X[:, 0].size, n)] / SCALE
    s, t = pick(Xs), pick(Xt)
    return {b: {"source_mean": float(s[i].mean()), "target_mean": float(t[i].mean()),
                "ks": float(ks_2samp(s[i], t[i]).statistic)} for i, b in enumerate(bands)}


def ndvi(X):
    r, n = X[:, RED].astype(np.float32), X[:, NIR].astype(np.float32)
    return (n - r) / (n + r + 1e-6)


@torch.no_grad()
def encoder_features(model, X, stats):
    model.to(DEVICE).eval()
    return np.concatenate([model.encoder(x.to(DEVICE))[-1].mean((2, 3)).cpu().numpy() for x, _ in loader(X, stats, 16)])


class DriftMonitor:
    def __init__(self, ref):
        self.ref = ref

    @classmethod
    def calibrate(cls, X, probs):
        chip_means = X.mean(axis=(2, 3)) / SCALE
        share = np.bincount(probs.argmax(1).ravel(), minlength=probs.shape[1]) / probs[:, 0].size
        return cls({"band_mean": chip_means.mean(0).tolist(), "band_spread": (chip_means.std(0) + 1e-6).tolist(),
                    "entropy_p95": float(np.quantile(entropy(probs).mean(axis=(1, 2)), 0.95)),
                    "class_share": share.tolist()})

    def check(self, X, probs, max_z=3.0, max_uncertain=0.3, max_gap=0.3):
        z = float((np.abs(X.mean(axis=(0, 2, 3)) / SCALE - self.ref["band_mean"]) / self.ref["band_spread"]).max())
        uncertain = float((entropy(probs).mean(axis=(1, 2)) > self.ref["entropy_p95"]).mean())
        gap = prior_gap(probs, self.ref["class_share"])
        status = "NEEDS_LABELS" if uncertain > max_uncertain or gap > max_gap else "ADAPT" if z > max_z else "OK"
        return {"status": status, "max_band_z": z, "uncertain_chip_share": uncertain, "prior_gap": gap}
