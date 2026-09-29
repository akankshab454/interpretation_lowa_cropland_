import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    for key in ("data_dir", "results_dir", "models_dir"):
        cfg[key] = ROOT / cfg[key]
        cfg[key].mkdir(exist_ok=True)
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def load_json(path):
    return json.loads(Path(path).read_text())
