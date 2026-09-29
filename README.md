# Iowa = Sahel land-cover shift

## Setup

```bash
docker build -t sahel .
```

Put the data zip contents in `data/`. If `data/` is empty, step 1 downloads everything from Planetary Computer (~15 min).

## Run everything

```bash
docker run --rm --gpus all --shm-size 2g -v "$PWD:/work" sahel sh run.sh
```

This runs `steps/1_data.py` to `steps/7_demo.py` in order and writes the JSON results and figures to `results/` and the weights to `models/`. It takes about 1.5 h on a 4 GB GPU.

## Run one step

```bash
docker run --rm --gpus all --shm-size 2g -v "$PWD:/work" sahel python steps/4_adapt.py
```

## Map a new scene

```bash
docker run --rm --gpus all -v "$PWD:/work" sahel python -m sahel.predict data/demo_mali_dry.tif --scene-adabn
```

This writes a class-map GeoTIFF and prints the drift report (OK / ADAPT / NEEDS_LABELS).

## Tests

```bash
docker run --rm -v "$PWD:/work" sahel python tests/test_core.py
```
