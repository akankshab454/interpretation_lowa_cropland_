# Iowa → Sahel: why the land-cover model breaks and what I would do

A segmentation model trained on Iowa does well there and falls apart in a semi-arid region in another season. No data came with the task, so I rebuilt the setup:
- **Imagery:** Sentinel-2, 8 bands from 490 to 865 nm, close to Pixxel Firefly's range.
- **Labels:** ESA WorldCover.
- **Source:** Iowa in July–August.
- **Target:** Niger and Senegal in January–March.

The Sahel labels were never used for training, only for scoring one held-out strip.

## What I found

![staircase](results/figures/staircase.png)

- Iowa scores **0.51** mIoU and the Sahel **0.11**. I checked whether random chip splits leak: they gave the same mean as holding out whole regions (0.50 vs 0.51), just noisier. So the drop is real, not a validation artefact.
- The model thinks the Sahel is a city. It labels **60%** of Sahel pixels "built-up" (the real share is under 1%), because in Iowa "bright and not green" meant roads and buildings.
- The shift is physical. The Sahel is 4–8× brighter in the visible bands, while NIR is almost the same. Per-chip accuracy tracks greenness (ρ = +0.83) and brightness (ρ = −0.87).

![shift](results/figures/shift.png)

- Season alone does a lot of damage. Iowa after harvest drops from 0.50 to **0.24** without leaving Iowa.

![season](results/figures/season_geo.png)

- In Niger, cropland and grass/shrub have the **same NDVI** (0.14 vs 0.16 in the dry season, 0.28 vs 0.28 in the wet season). From one image, "cropland" is a land-use label, not something the pixel shows.

## What I tried

| Method | mIoU | Pixel accuracy | Cropland IoU | Grass/shrub IoU |
|---|---|---|---|---|
| Map that says "cropland" everywhere | 0.115 | 0.58 | 0.58 | 0 |
| Source model as is | 0.113 | 0.25 | 0.24 | 0.19 |
| Standardise each region's bands | 0.032 | 0.11 | 0.16 | 0.02 |
| AdaBN, statistics from the scene being mapped | 0.094 | 0.46 | 0.51 | 0.04 |
| + TENT | 0.104 | 0.51 | 0.57 | 0.03 |
| + brightness augmentation toward Sahel | 0.102 | 0.48 | 0.51 | 0.06 |
| + class-balanced self-training + FDA | 0.098 | 0.48 | 0.52 | 0.03 |
| + 20 labelled Sahel chips | **0.131** | | **0.59** | 0.09 |

![per class](results/figures/per_class.png)
![examples](results/figures/qualitative.png)

## How I read it

1. **AdaBN fixes the "city" mistake.** Pixel accuracy goes from 25% to about 50%. But the model then calls almost everything cropland, so mIoU does not move. Four of the six classes are under 1.5% of Sahel pixels, and mIoU swings on them.
2. **BN statistics should come from the scene being mapped.** Statistics from other Sentinel-2 dates in the same region gave lower pixel accuracy (0.39 vs 0.46).
3. **Standardising bands to look like Iowa was the worst idea.** It turns bright soil into "green-looking" numbers.
4. **Self-training fed on its own mistakes.** "Water" pseudo-labels grew from 7% to 10% between rounds, with no water in the region. FDA added nothing on top of the brightness augmentation.
5. **Nothing without labels beat the all-cropland map.** So the leftover gap is about what the labels mean, not how the images look.
6. **I could not have picked a method without labels.** Entropy, confidence, rotation agreement and prior gap all failed to rank the methods (every |ρ| ≤ 0.15). Entropy does flag bad chips inside one model (ρ = −0.51), so I use it for monitoring.

![proxies](results/figures/proxies.png)

7. **A few labels beat every trick.** 5 labelled chips (≈0.12–0.13) already beat everything above. At this size, choosing uncertain and diverse chips was no better than random, so the number of labels mattered more than which ones.

![labels](results/figures/labels.png)

WorldCover itself is weak on Sahel cropland, so these Sahel scores measure agreement with WorldCover, not true accuracy.

## What I would ship

- **Per-scene BN statistics at inference**, plus a drift monitor that returns OK / ADAPT / NEEDS_LABELS.
- **A small labelling loop:** 10–20 chips per new zone.
- **Next:** add a second-season image as input, because the cropland signal is in time, not in one date.

On two scenes the model never saw:

| Scene | Model | Monitor |
|---|---|---|
| Ségou, Mali | Iowa model | NEEDS_LABELS (51% "built", 16% "water") |
| Ségou, Mali | adapted + scene BN | OK |
| Western Iowa | Iowa model | ADAPT |
| Western Iowa | adapted + scene BN | OK |

"OK" means the scene looks like what the model was checked on, not that the map is right.

## Run

```bash
docker build -t sahel .
docker run --rm --gpus all --shm-size 2g -v "$PWD:/work" sahel sh run.sh
docker run --rm --gpus all -v "$PWD:/work" sahel python -m sahel.predict data/demo_mali_dry.tif --scene-adabn
docker run --rm -v "$PWD:/work" sahel python tests/test_core.py
```

Put the data zip in `data/`, or let `steps/1_data.py` download it from Planetary Computer (~15 min). A full run takes about 1.5 h on a 4 GB GTX 1650.
