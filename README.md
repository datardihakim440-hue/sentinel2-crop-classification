# Crop-Type Classification from Multi-Temporal Sentinel-2 Imagery

A clean, reproducible workflow for **pixel-wise crop-type classification** on a
subset of the public [PASTIS benchmark](https://github.com/VSainteuf/pastis-benchmark)
(multi-temporal Sentinel-2, 10 bands, southern France).

The model is a **Random Forest on engineered temporal–spectral features**,
followed by a light spatial majority-filter post-process. It runs end-to-end on
a CPU laptop in a few minutes and reaches **76.0 % overall accuracy** (74.5 %
before smoothing) on a held-out test split. See [`report.md`](report.md) for the
full analysis, results and interpretation.

---

## 1. Project overview

Each Sentinel-2 patch is a time series of satellite images over one growing
season. Different crops have different *phenology* (seasonal growth patterns), so
we summarise every pixel's multi-date, multi-band signature into compact
statistics and classify it with a Random Forest. The emphasis is on clear
reasoning, reproducibility and honest interpretation rather than leaderboard
accuracy.

## 2. Dataset structure

The data is **not** included in this repository (~1.5 GB). Place it locally as:

```
PASTIS_subset/
├── DATA_S2/
│   └── S2_<patch_id>.npy          # (46, 10, 128, 128) int16  — T × bands × H × W
├── ANNOTATIONS/
│   └── TARGET_<patch_id>.npy      # (1, 128, 128) uint8       — semantic labels
└── metadata.geojson               # per-patch folds, acquisition dates, geometry
```

- **Bands (index → name):** 0 B2, 1 B3, 2 B4, 3 B5, 4 B6, 5 B7, 6 B8, 7 B8A,
  8 B11, 9 B12.
- **Labels:** `0` background, `1–18` crop types, `19` void/border (ignored).
  Full name map in [`configs/config.yaml`](configs/config.yaml).

The dataset root and all paths are configurable in `configs/config.yaml`.

## 3. Repository layout

```
├── README.md              ├── report.md            ├── requirements.txt
├── environment.yml        ├── configs/config.yaml
├── src/
│   ├── data_loading.py      # read .npy / metadata, patch discovery
│   ├── preprocessing.py     # reflectance scaling, vegetation indices, temporal features
│   ├── dataset.py           # assemble per-pixel (X, y) matrices with sampling
│   ├── splits.py            # reproducible train/val/test from official PASTIS folds
│   ├── train.py             # fit + persist the Random Forest, validation metrics
│   ├── evaluate.py          # metrics (acc, P/R/F1, IoU/mIoU, confusion matrix)
│   ├── postprocess.py       # spatial majority-filter smoothing of predictions
│   ├── predict.py           # dense test predictions + output figures
│   ├── run_eda.py           # exploratory-analysis figures + summary
│   └── visualization.py     # all plotting helpers
├── notebooks/exploration_and_training.ipynb   # narrative walkthrough
├── splits/{train,val,test}.txt                # patch IDs per split (committed)
└── outputs/
    ├── figures/     # EDA + confusion matrix + prediction triptychs (committed)
    ├── metrics/     # metrics_*.json, feature_importances.json (committed)
    ├── models/      # trained model — NOT committed (428 MB), reproduce locally
    └── predictions/ # predicted label maps — NOT committed
```

## 4. Environment setup

Using conda (recommended):

```bash
conda env create -f environment.yml
conda activate sola_agri
```

Or with pip into any Python 3.11 environment:

```bash
pip install -r requirements.txt
```

## 5. How to run

Run from the project root (so `src` is importable). Each step reads
`configs/config.yaml`.

```bash
# 1. Create the reproducible train/val/test split -> splits/*.txt
python -m src.splits

# 2. Exploratory data analysis -> outputs/figures + outputs/metrics/eda_summary.json
python -m src.run_eda

# 3. Train the Random Forest -> outputs/models/rf_model.joblib + validation metrics
python -m src.train

# 4. Evaluate on the test split + generate output figures/predictions
python -m src.predict

# (optional) evaluate any split explicitly
python -m src.evaluate --split test
```

Everything is deterministic (`seed: 42`), so a fresh run reproduces the reported
numbers.

> **Windows note:** the code was developed and validated on Windows with a conda
> env named `sola_agri`. If `python -m src.<module>` cannot find `src`, ensure
> you are in the project root, or set `PYTHONPATH` to it.

## 6. Reproducing the main results

1. Place the data under `PASTIS_subset/` (see §2).
2. `conda activate sola_agri` (or install `requirements.txt`).
3. `python -m src.splits && python -m src.train && python -m src.predict`.

Expected (seed 42): **test** accuracy ≈ 0.745 per-pixel, **≈ 0.760 with spatial
smoothing** (mIoU 0.27 → 0.30); **validation** ≈ 0.734 → 0.744. Figures and
metric JSONs are written to `outputs/`.

### Trained model

The fitted model (`outputs/models/rf_model.joblib`, ~428 MB) exceeds GitHub's
file-size limit and is intentionally **not committed** (see `.gitignore`).
Reproduce it in ~80 s with `python -m src.train`. If you need to share the
binary, use Git LFS or an external download link — do not push it directly.

## 7. Approach summary

- **Features:** per-pixel temporal statistics (`mean/std/min/max/p25/p50/p75/
  amplitude`) of 10 bands + NDVI/NDWI/NDRE/EVI → 112 features.
- **Model:** `RandomForestClassifier` (300 trees, `class_weight=
  balanced_subsample`).
- **Post-process:** unsupervised spatial majority filter (window 5) to remove
  salt-and-pepper noise — no label leakage.
- **Split:** official PASTIS 5-fold, at patch level (no pixel leakage).
- **Sampling:** ≤ 400 training pixels per class per patch; dense evaluation.

## 8. Key assumptions

- Standard PASTIS 10-band order and Sentinel-2 L2A reflectance scale (÷10000).
- Class 19 = void, excluded from training and metrics; background (0) kept as a
  class (configurable).
- All patches in this subset share the same 46 acquisition dates (verified).

## 9. Known limitations

- Small subset (102 patches, one tile, one season); some classes have **zero**
  pixels in the test fold and cannot be scored.
- The per-pixel model ignores spatial context → salt-and-pepper predictions and
  ragged parcel boundaries.
- Rare classes (< 1 % of pixels) are largely missed; macro metrics reflect this.

See [`report.md`](report.md) §7 for prioritised next steps (U-TAE / U-Net,
parcel-level majority voting, explicit cloud masking, richer temporal features).
