import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import loader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IGNORE = 255


def build(cfg, weights="imagenet"):
    return smp.Unet(cfg["train"]["encoder"], encoder_weights=weights, in_channels=len(cfg["bands"]),
                    classes=len(cfg["classes"]))


def save(model, stats, path, bn_states=None):
    torch.save({"state": model.state_dict(), "mean": stats[0], "std": stats[1], "bn_states": bn_states}, path)


def load(cfg, path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = build(cfg, weights=None).to(DEVICE)
    model.load_state_dict(ckpt["state"])
    model.bn_states = ckpt.get("bn_states") or {}
    return model, (ckpt["mean"], ckpt["std"])


def fit(model, X, Y, stats, cfg, epochs=None, lr=None, gain_to=None):
    t = cfg["train"]
    epochs, lr = epochs or t["epochs"], lr or t["lr"]
    dl = loader(X, stats, t["batch"], Y, train=True, gain_to=gain_to)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(dl))
    model.to(DEVICE).train()
    for _ in tqdm(range(epochs), leave=False):
        for x, y in dl:
            loss = F.cross_entropy(model(x.to(DEVICE)), y.to(DEVICE), ignore_index=IGNORE)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model


@torch.no_grad()
def predict(model, X, stats, batch=16):
    model.to(DEVICE).eval()
    return np.concatenate([model(x.to(DEVICE)).softmax(1).half().cpu().numpy() for x, _ in loader(X, stats, batch)])


def confusion(pred, target, k):
    m = target != IGNORE
    return np.bincount(k * target[m].astype(np.int64) + pred[m], minlength=k * k).reshape(k, k)


def scores(cm, names):
    tp = np.diag(cm).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = tp / (cm.sum(0) + cm.sum(1) - tp)
    total, freq = cm.sum(), cm.sum(1) / cm.sum()
    share = lambda v: dict(zip(names, np.round(v, 4).tolist()))
    return {"miou": float(np.nanmean(iou)), "fwiou": float(np.nansum(freq * iou)), "oa": float(tp.sum() / total),
            "iou": share(iou), "true_share": share(freq), "pred_share": share(cm.sum(0) / total)}


def ece(conf, correct, bins=15):
    b = np.clip((conf * bins).astype(int), 0, bins - 1)
    n = np.bincount(b, minlength=bins)
    gap = np.abs(np.bincount(b, conf, minlength=bins) - np.bincount(b, correct, minlength=bins))
    return float(gap.sum() / n.sum())


def evaluate(Y, probs, names):
    pred, m = probs.argmax(1), Y != IGNORE
    cm = confusion(pred.ravel(), Y.ravel(), len(names))
    res = scores(cm, names)
    res["ece"] = ece(probs.max(1)[m].astype(np.float32), (pred == Y)[m].astype(np.float32))
    return res
