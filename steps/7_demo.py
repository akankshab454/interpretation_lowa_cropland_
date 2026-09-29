import json

from sahel import load_config, save_json
from sahel.predict import run

cfg = load_config()
res, models, data = cfg["results_dir"], cfg["models_dir"], cfg["data_dir"]
deployments = {"source model": ("source", False), "adapted model + scene AdaBN": ("adapted", True)}
summary = {f"{scene} / {name}": run(data / f"{scene}.tif", models / f"{ckpt}.pt", res / f"monitor_{ckpt}.json",
                                    res / f"{scene}_{ckpt}.tif", cfg, scene_bn)
           for scene in cfg["demo_scenes"] for name, (ckpt, scene_bn) in deployments.items()}
print(json.dumps(summary, indent=2))
save_json(summary, res / "demo.json")
