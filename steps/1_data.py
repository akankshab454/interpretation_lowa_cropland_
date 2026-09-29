from sahel import load_config
from sahel.data import fetch_region, fetch_scene

cfg = load_config()
for i, (name, region) in enumerate(cfg["regions"].items()):
    if not (cfg["data_dir"] / f"{name}.npz").exists():
        fetch_region(name, region, cfg, seed=cfg["seed"] + i)
for name, (lon, lat, dates) in cfg["demo_scenes"].items():
    if not (cfg["data_dir"] / f"{name}.tif").exists():
        fetch_scene(lon, lat, dates, cfg, cfg["data_dir"] / f"{name}.tif")
