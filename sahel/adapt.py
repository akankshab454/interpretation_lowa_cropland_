import copy
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm import tqdm

from .data import loader
from .model import DEVICE, IGNORE, predict


def bn_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]


def bn_state(model, affine=False):
    names = {n for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)}
    keep = ("running_mean", "running_var", "num_batches_tracked") + (("weight", "bias") if affine else ())
    return {k: v.detach().clone() for k, v in model.state_dict().items()
            if k.rsplit(".", 1)[0] in names and k.rsplit(".", 1)[1] in keep}


@torch.no_grad()
def adabn(model, X, stats, batch=8):
    for bn in bn_layers(model):
        bn.reset_running_stats()
        bn.momentum = None
    model.to(DEVICE).train()
    for x, _ in loader(X, stats, batch):
        model(x.to(DEVICE))
    for bn in bn_layers(model):
        bn.momentum = 0.1
    return model.eval()


class Regional:
    def __init__(self, model, states):
        self.model, self.states = model, states

    @classmethod
    def fit(cls, model, X, regions, stats, adapt_fn=adabn, affine=False):
        base, states = bn_state(model, affine=True), {}
        for r in np.unique(regions):
            model.load_state_dict(base, strict=False)
            adapt_fn(model, X[regions == r], stats)
            states[r] = bn_state(model, affine)
        return cls(model, states)

    def use(self, region):
        self.model.load_state_dict(self.states[region], strict=False)
        return self.model

    def predict(self, X, regions, stats):
        out = None
        for r in np.unique(regions):
            p = predict(self.use(r), X[regions == r], stats)
            out = np.empty((len(X),) + p.shape[1:], p.dtype) if out is None else out
            out[regions == r] = p
        return out


def test_time(model, X, regions, stats):
    return Regional.fit(copy.deepcopy(model), X, regions, stats).predict(X, regions, stats)


def fda(src, tgt, beta):
    fs, ft = torch.fft.fft2(src.float()), torch.fft.fft2(tgt.float())
    amp, b = fs.abs(), max(1, int(min(src.shape[-2:]) * beta))
    for rows in (slice(0, b), slice(-b, None)):
        for cols in (slice(0, b), slice(-b, None)):
            amp[..., rows, cols] = ft.abs()[..., rows, cols]
    return torch.fft.ifft2(torch.polar(amp, fs.angle())).real


def strong_view(x, mask_ratio=0.3, patch=32):
    n, c, h, w = x.shape
    x = x * torch.empty(n, c, 1, 1, device=x.device).uniform_(0.8, 1.2)
    x = x + 0.2 * torch.randn(n, c, 1, 1, device=x.device) + 0.05 * torch.randn_like(x)
    keep = (torch.rand(n, 1, h // patch, w // patch, device=x.device) > mask_ratio).float()
    return x * F.interpolate(keep, size=(h, w), mode="nearest")


def class_thresholds(probs, k, quantile, cap=0.9):
    conf, pred = probs.max(1), probs.argmax(1)
    thr = np.full(k, cap, np.float32)
    for c in range(k):
        vals = conf[pred == c]
        if vals.size:
            sample = vals[:: max(1, vals.size // 200_000)].astype(np.float32)
            thr[c] = min(float(np.quantile(sample, 1 - quantile)), cap)
    return thr


def self_train(model, Xs, Ys, Xt, regions, stats, cfg, use_fda=True, seed=0, gain_to=None):
    a, k, half = cfg["adapt"], len(cfg["classes"]), cfg["train"]["batch"] // 2
    rng, names = np.random.default_rng(seed), np.unique(regions)
    student, teacher = model.to(DEVICE), copy.deepcopy(model)
    opt = torch.optim.AdamW(student.parameters(), lr=a["lr"], weight_decay=1e-4)
    src = cycle(loader(Xs, stats, half, Ys, train=True, gain_to=gain_to))
    tgt = {r: cycle(loader(Xt[regions == r], stats, half, train=True)) for r in names}
    history = []
    for rnd in range(a["rounds"]):
        regional = Regional.fit(teacher, Xt, regions, stats)
        thr = class_thresholds(regional.predict(Xt, regions, stats), k, a["class_quantile"])
        thr_t, kept = torch.tensor(thr, device=DEVICE), np.zeros(k)
        student.train()
        for _ in tqdm(range(a["steps_per_round"]), leave=False):
            region = rng.choice(names)
            (xs, ys), (xt, _) = next(src), next(tgt[region])
            xs, ys, xt = xs.to(DEVICE), ys.to(DEVICE), xt.to(DEVICE)
            if use_fda:
                xs = fda(xs, xt, a["fda_beta"])
            with torch.no_grad():
                conf, pseudo = regional.use(region)(xt).softmax(1).max(1)
            pseudo[conf < thr_t[pseudo]] = IGNORE
            kept += np.bincount(pseudo[pseudo != IGNORE].cpu().numpy(), minlength=k)
            loss = F.cross_entropy(student(xs), ys, ignore_index=IGNORE) + \
                F.cross_entropy(student(strong_view(xt)), pseudo, ignore_index=IGNORE)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for pt, ps in zip(teacher.parameters(), student.parameters()):
                    pt.mul_(a["ema"]).add_(ps, alpha=1 - a["ema"])
        history.append({"round": rnd + 1, "thresholds": thr.round(3).tolist(),
                        "pseudo_share": (kept / max(kept.sum(), 1)).round(4).tolist()})
        print(history[-1])
    return Regional.fit(teacher, Xt, regions, stats), history


def tent(model, X, stats, steps, lr=1e-4, batch=8):
    model.to(DEVICE).train().requires_grad_(False)
    params = [p for bn in bn_layers(model) for p in (bn.weight, bn.bias)]
    for p in params:
        p.requires_grad_(True)
    opt, data = torch.optim.Adam(params, lr=lr), cycle(loader(X, stats, batch, train=True))
    for _ in range(steps):
        p = model(next(data)[0].to(DEVICE)).softmax(1)
        loss = -(p * p.clamp_min(1e-8).log()).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.requires_grad_(True)
    return model.eval()


def entropy(probs):
    p = probs.astype(np.float32)
    return -(p * np.log(np.clip(p, 1e-8, None))).sum(1)


def prior_gap(probs, prior):
    p = np.bincount(probs.argmax(1).ravel(), minlength=len(prior)) / probs[:, 0].size
    q = np.asarray(prior, float) / np.sum(prior)
    m = (p + q) / 2
    kl = lambda a, b: np.sum(np.where(a > 0, a * np.log(np.where(a > 0, a, 1) / b), 0))
    return float(np.sqrt((kl(p, m) + kl(q, m)) / 2))


def proxies(predict_fn, X, prior):
    probs = predict_fn(X)
    rotated = predict_fn(np.ascontiguousarray(np.rot90(X, 1, axes=(2, 3))))
    agree = np.rot90(rotated.argmax(1), -1, axes=(1, 2)) == probs.argmax(1)
    return {"entropy": float(entropy(probs).mean()), "confident_share": float((probs.max(1) > 0.9).mean()),
            "rotation_agreement": float(agree.mean()), "prior_gap": prior_gap(probs, prior)}


def pick_chips(probs, features, budget, strategy, rng):
    if strategy == "random":
        return rng.choice(len(probs), budget, replace=False)
    uncertainty = entropy(probs).mean(axis=(1, 2))
    groups = KMeans(budget, n_init=10, random_state=0).fit_predict(features)
    return np.array([np.flatnonzero(groups == g)[uncertainty[groups == g].argmax()] for g in range(budget)])
