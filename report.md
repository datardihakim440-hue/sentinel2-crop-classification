# Crop-Type Classification — Analysis Report

**Task.** Pixel-wise crop-type classification from multi-temporal Sentinel-2
imagery, using a subset of the public [PASTIS benchmark](https://github.com/VSainteuf/pastis-benchmark)
(102 patches, tile `t31tfm`, southern France).

**Headline result.** A Random-Forest classifier on engineered temporal–spectral
features reaches **74.5 % overall accuracy** on the held-out test split
(macro-F1 0.37, mean IoU 0.27), with validation numbers essentially identical
(73.4 % / 0.38 / 0.26) — i.e. no overfitting. A light **spatial majority-filter**
post-process (which injects the spatial prior the per-pixel model lacks) lifts
this to **76.0 % accuracy / mIoU 0.30** with no label leakage.

---

## 1. Approach and why this model

The brief explicitly values *reasoning, reproducibility and interpretation over
raw accuracy*, and prefers a well-justified simple baseline. I therefore chose a
**per-pixel Random Forest on temporal–spectral summary features** rather than a
deep segmentation network.

Rationale:

- **The signal is temporal.** Crops are separable mainly by *phenology* — when
  they green up, peak and senesce — not by a single-date appearance. The NDVI
  profiles (`outputs/figures/ndvi_profiles.png`) show distinct seasonal curves
  per class. Summarising each pixel's 46-date × 10-band time series into
  temporal statistics captures most of that signal compactly.
- **Compute.** Random Forest trains on CPU in ~80 s and needs no GPU, so the
  whole pipeline is runnable on a laptop and fully reproducible.
- **Interpretability.** Gini feature importances tell us *which* bands and dates
  matter (see §5), which a black-box CNN would not.
- **Robustness.** Trees are insensitive to feature scaling and monotonic
  outliers, which matters given residual cloud/atmospheric noise.

A deep spatio-temporal model (U-TAE, the PASTIS reference architecture) would
add spatial context and likely improve boundary quality and rare-class recall —
this is discussed as the primary next step in §7.

## 2. Data understanding (EDA)

| Property | Value |
|---|---|
| Patches | 102 (all tile `t31tfm`) |
| Array shape (S2) | `(46, 10, 128, 128)` int16 — T×bands×H×W |
| Array shape (target) | `(1, 128, 128)` uint8, 0th annotation layer |
| Temporal range | 20 Sep 2018 → 25 Oct 2019 (46 acquisitions) |
| Classes | 0 = background, 1–18 crops, 19 = void (ignored) |

Key observations (figures in `outputs/figures/`):

- **Severe class imbalance.** Background (30 %), meadow (18 %), soft winter
  wheat, corn and soybeans dominate; grapevine, orchard, potatoes, durum wheat
  and mixed cereal are each < 1 % of pixels. This shapes every result below.
- **Cloud contamination is date-specific.** In the NDVI phenology plot, *all*
  classes dip simultaneously on a handful of acquisitions (indices ~3, 6, 26,
  36) — these are cloudy dates, not real vegetation change. Using temporal
  percentiles/medians (rather than the raw series) makes the features robust to
  these dips.
- **Parcel structure.** Labels are piecewise-constant over field parcels with
  thin void borders — the landscape is a patchwork of medium-sized fields.

## 3. Preprocessing & features

1. Convert int16 DN → reflectance (÷10000), clip to `[0, 1]`.
2. Compute four vegetation indices per date: **NDVI, NDWI, NDRE, EVI**
   (EVI clipped to `[-1, 1]`).
3. Collapse each of the 10 bands and 4 indices over time into 8 statistics —
   `mean, std, min, max, p25, p50, p75, amplitude` → **112 features / pixel**.

Void pixels (class 19) are excluded everywhere. For training, pixels are
randomly subsampled to ≤ 400 per class per patch, which bounds memory and
partly rebalances the classes; validation/test are evaluated on **all** valid
pixels for honest metrics.

## 4. Split strategy

I reuse the **official PASTIS 5-fold** assignment stored in `metadata.geojson`
(folds 1–3 = train, 4 = val, 5 = test), splitting at the **patch** level so no
pixels leak between train and evaluation. The folds are geographically
block-designed to limit spatial autocorrelation leakage, and reusing them keeps
the protocol comparable to the published benchmark. Patch IDs per split are
saved in `splits/{train,val,test}.txt` for exact reproduction.

## 5. Results

**Overall (test):** accuracy 0.745 · macro-F1 0.369 · mIoU 0.273 (per-pixel).

**Effect of spatial smoothing.** A majority filter over each prediction map —
window size selected on *validation* (w=5) — improves every metric on both
splits, confirming the per-pixel model's main weakness is a lack of spatial
context rather than the features:

| | Accuracy | Macro-F1 | mIoU |
|---|---|---|---|
| Test — per-pixel | 0.745 | 0.369 | 0.273 |
| **Test — + smoothing** | **0.760** | **0.379** | **0.303** |
| Val — per-pixel | 0.734 | 0.383 | 0.264 |
| Val — + smoothing | 0.744 | 0.387 | 0.287 |

Per-class F1 / IoU (test), grouped by outcome:

| Performance | Classes (F1) |
|---|---|
| **Strong** | Winter rapeseed 0.90 · Background 0.80 · Soft winter wheat 0.78 · Corn 0.74 · Meadow 0.72 · Soybeans 0.67 |
| **Moderate** | Winter barley 0.55 |
| **Poor / failed** | Sunflower 0.24 (precision 0.97 but recall 0.14) · Leguminous fodder 0.12 · Spring barley, triticale, durum wheat, FVF, mixed cereal, sorghum ≈ 0.00 |

Top features (Gini importance): SWIR (`B11_p25`, `B12_p25`), red-edge
(`B6_p75`, `B5_p25`), blue mean, and NDVI/NDRE percentiles — i.e. the model
leans on SWIR + red-edge seasonality, which is consistent with agronomic
knowledge (moisture and canopy-chlorophyll dynamics separate crop types).

## 6. Interpretation

- **What works.** Spectrally and phenologically distinct, well-sampled crops
  (rapeseed's bright-yellow flowering, corn/soybean summer cycle, winter wheat)
  are classified well. Rapeseed is the standout because its flowering signature
  is unmistakable.
- **Confusions (see confusion matrix).** The main errors are *agronomically
  sensible*: winter **durum wheat → soft winter wheat**, **triticale → wheat**,
  and **mixed cereal → soybeans/wheat** — winter cereals share very similar
  phenology and spectra. Spring barley is diffusely confused with the winter
  cereals. Sunflower is almost never predicted (recall 0.14) but is right when
  it is (precision 0.97) — the balanced weighting still can't overcome how few
  sunflower pixels exist.
- **Class imbalance dominates the failures.** Every ≈0-F1 class has a tiny
  support (spring barley 153, durum 306, mixed cereal 674, sorghum 665 test
  pixels). Four classes (grapevine, beet, potatoes, orchard) have **zero** test
  pixels in fold 5, so they cannot be scored at all — a direct consequence of a
  small 102-patch subset.
- **No spatial context → salt-and-pepper.** Because each pixel is classified
  independently, raw predictions are noisy inside parcels and ragged at
  boundaries (see the `comparison_*.png` figures: RGB | truth | per-pixel |
  smoothed). The spatial majority filter demonstrably cleans this up (+1.5 pp
  accuracy, +3 pp mIoU on test) — direct evidence that adding spatial structure
  is the highest-value improvement, motivating a spatio-temporal model next.
- **Cloud & timing effects.** Cloudy dates inject noise; temporal
  median/percentile features mitigate but do not remove it. Missing a crop's key
  phenological window (e.g. a cloudy flowering date) disproportionately hurts
  rare crops.

## 7. Limitations & next steps

**Limitations.** Small subset (102 patches, one tile, one season); some classes
absent from the test fold; per-pixel model ignores spatial structure; macro
metrics are pulled down hard by rare classes; simple cloud handling.

**Recommended next steps (in priority order).**
1. **Add spatial context** — move to the PASTIS reference **U-TAE** (or a
   lightweight U-Net on temporal composites). Biggest expected gain: boundary
   quality and coherence.
2. **Parcel-level post-processing** — a spatial majority filter is already
   implemented (§5) and helps; the next refinement is to majority-vote within
   true parcel polygons (or unsupervised superpixels) instead of a fixed window,
   which respects field boundaries exactly.
3. **Better rare-class handling** — targeted oversampling / focal-style
   weighting, or merging agronomically-identical cereals into super-classes.
4. **Explicit cloud masking** — use the scene classification / cloud probability
   to drop or down-weight contaminated dates before feature extraction.
5. **Richer temporal features** — harmonic/Fourier fits or phenology metrics
   (green-up date, peak date, season length) instead of order-agnostic stats.
6. **More data** — additional tiles/folds and multi-year data to cover rare
   crops and improve generalisation.

## 8. Assumptions

- Band order is the standard PASTIS 10-band order (B2,B3,B4,B5,B6,B7,B8,B8A,
  B11,B12); reflectance scale factor 10000 (Sentinel-2 L2A convention).
- Class 19 is the void/border label and is excluded from training and metrics.
- Background (0) is kept as a normal class (toggleable in the config).
- All 102 patches share identical 46-date acquisition timing (verified true for
  this subset), so temporal features are directly comparable across patches.
