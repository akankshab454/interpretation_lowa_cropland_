| Method | mIoU | OA | fwIoU | cropland IoU | grass/shrub IoU | ECE | Iowa mIoU |
|---|---|---|---|---|---|---|---|
| all-cropland map | 0.115 | 0.576 | 0.331 | 0.58 | 0.00 | - | - |
| source only | 0.113 | 0.252 | 0.213 | 0.24 | 0.19 | 0.18 | 0.495 |
| per-region band standardisation | 0.032 | 0.110 | 0.102 | 0.16 | 0.02 | 0.72 | 0.495 |
| AdaBN, one 'Sahel' | 0.095 | 0.483 | 0.315 | 0.53 | 0.03 | 0.39 | 0.495 |
| AdaBN per region | 0.094 | 0.457 | 0.311 | 0.51 | 0.04 | 0.38 | 0.495 |
| AdaBN per region + TENT | 0.104 | 0.513 | 0.340 | 0.57 | 0.03 | 0.36 | 0.495 |
| + radiometric aug | 0.102 | 0.478 | 0.319 | 0.51 | 0.06 | 0.35 | 0.487 |
| + self-train | 0.098 | 0.484 | 0.317 | 0.53 | 0.03 | 0.37 | 0.474 |
| + self-train + FDA | 0.098 | 0.484 | 0.312 | 0.52 | 0.03 | 0.38 | 0.479 |
| + 20 labelled chips | 0.131 | | | 0.59 | 0.09 | | |
