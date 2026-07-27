"""Preprocessing and per-pixel feature engineering.

The core modelling idea for the Random-Forest baseline is to turn every pixel
into a single feature vector that summarises its *spectral-temporal* signature
over the growing season. Different crops have distinct phenology (green-up,
peak, senescence), so temporal statistics of reflectance and vegetation indices
are highly discriminative while remaining compact and CPU-friendly.

Pipeline per patch (T, C, H, W) int16  ->  (H*W, F) float32 feature matrix.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Reflectance scaling
# --------------------------------------------------------------------------- #
def to_reflectance(s2: np.ndarray, scale: float = 10000.0,
                   clip=(0.0, 1.0)) -> np.ndarray:
    """Convert raw int16 DN to surface reflectance in ~[0, 1].

    Sentinel-2 L2A products are stored as reflectance * 10000. A few pixels can
    be slightly negative (atmospheric correction) or >1 (bright targets); we
    clip to a sane range to stabilise index computation.
    """
    refl = s2.astype(np.float32) / scale
    if clip is not None:
        refl = np.clip(refl, clip[0], clip[1])
    return refl


# --------------------------------------------------------------------------- #
# Vegetation indices  (input: reflectance (T, C, H, W))
# --------------------------------------------------------------------------- #
# Band indices in the provided array order.
B_BLUE, B_GREEN, B_RED, B_RE1, B_RE2, B_RE3, B_NIR, B_NIRA, B_SWIR1, B_SWIR2 = range(10)
_EPS = 1e-6


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return num / (den + _EPS)


def least_cloudy_timestep(s2: np.ndarray, scale: float = 10000.0) -> int:
    """Heuristically pick the least-cloudy acquisition for display.

    Clouds are bright across the visible bands, so we score each timestep by its
    mean blue-band reflectance and pick the darkest (lowest) one. Cheap and good
    enough for choosing a clear frame to visualise; not used for modelling.
    """
    blue = s2[:, B_BLUE].astype(np.float32) / scale
    return int(np.argmin(blue.reshape(blue.shape[0], -1).mean(axis=1)))


def compute_indices(refl: np.ndarray, which=("ndvi", "ndwi", "ndre", "evi")) -> dict:
    """Compute vegetation-index time series from reflectance ``(T, C, H, W)``.

    Returns a dict of index-name -> ``(T, H, W)`` arrays.
    """
    red = refl[:, B_RED]
    nir = refl[:, B_NIR]
    green = refl[:, B_GREEN]
    blue = refl[:, B_BLUE]
    re1 = refl[:, B_RE1]

    out = {}
    if "ndvi" in which:
        out["ndvi"] = _safe_ratio(nir - red, nir + red)
    if "ndwi" in which:  # McFeeters water/moisture index
        out["ndwi"] = _safe_ratio(green - nir, green + nir)
    if "ndre" in which:  # red-edge, sensitive to canopy chlorophyll
        out["ndre"] = _safe_ratio(nir - re1, nir + re1)
    if "evi" in which:
        # EVI's denominator can cross zero for atypical spectra, producing
        # extreme values; clip to EVI's physically meaningful range.
        evi = 2.5 * _safe_ratio(nir - red, nir + 6.0 * red - 7.5 * blue + 1.0)
        out["evi"] = np.clip(evi, -1.0, 1.0)
    return out


# --------------------------------------------------------------------------- #
# Temporal statistics
# --------------------------------------------------------------------------- #
_STAT_FUNCS = {
    "mean": lambda a: np.mean(a, axis=0),
    "std": lambda a: np.std(a, axis=0),
    "min": lambda a: np.min(a, axis=0),
    "max": lambda a: np.max(a, axis=0),
    "p25": lambda a: np.percentile(a, 25, axis=0),
    "p50": lambda a: np.percentile(a, 50, axis=0),
    "p75": lambda a: np.percentile(a, 75, axis=0),
    "amplitude": lambda a: np.max(a, axis=0) - np.min(a, axis=0),
}


def _temporal_stats(stack: np.ndarray, stats) -> list[np.ndarray]:
    """Collapse a ``(T, H, W)`` stack into a list of ``(H, W)`` stat maps."""
    return [_STAT_FUNCS[s](stack) for s in stats]


# --------------------------------------------------------------------------- #
# Full feature builder
# --------------------------------------------------------------------------- #
def build_feature_stack(s2: np.ndarray, cfg: dict) -> tuple[np.ndarray, list[str]]:
    """Build a ``(F, H, W)`` feature stack and the matching feature names.

    Features = temporal stats of each of the 10 bands  +  temporal stats of
    each vegetation index.
    """
    pp = cfg["preprocess"]
    feat_cfg = cfg["features"]
    band_names = cfg["data"]["band_names"]
    stats = feat_cfg["temporal_stats"]

    refl = to_reflectance(
        s2,
        scale=pp["reflectance_scale"],
        clip=tuple(pp["clip_reflectance"]),
    )

    maps: list[np.ndarray] = []
    names: list[str] = []

    # Per-band temporal statistics.
    for c, bname in enumerate(band_names):
        for smap, sname in zip(_temporal_stats(refl[:, c], stats), stats):
            maps.append(smap)
            names.append(f"{bname}_{sname}")

    # Vegetation-index temporal statistics.
    indices = compute_indices(refl, which=feat_cfg["indices"])
    for iname, istack in indices.items():
        for smap, sname in zip(_temporal_stats(istack, stats), stats):
            maps.append(smap)
            names.append(f"{iname}_{sname}")

    stack = np.stack(maps, axis=0).astype(np.float32)  # (F, H, W)
    return stack, names


def stack_to_pixels(stack: np.ndarray) -> np.ndarray:
    """Reshape ``(F, H, W)`` -> ``(H*W, F)`` pixel feature matrix."""
    f, h, w = stack.shape
    return stack.reshape(f, h * w).T


def feature_names(cfg: dict) -> list[str]:
    """Return the ordered feature names without building the arrays."""
    stats = cfg["features"]["temporal_stats"]
    names = []
    for bname in cfg["data"]["band_names"]:
        names += [f"{bname}_{s}" for s in stats]
    for iname in cfg["features"]["indices"]:
        names += [f"{iname}_{s}" for s in stats]
    return names
