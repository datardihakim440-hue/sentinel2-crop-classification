"""Generate exploratory-data-analysis artefacts (figures + summary JSON).

Run::

    python -m src.run_eda

Produces:
    outputs/figures/class_distribution.png
    outputs/figures/ndvi_profiles.png
    outputs/figures/patch_overview_<id>.png   (a few example patches)
    outputs/figures/aoi_footprints.png
    outputs/metrics/eda_summary.json
"""

from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np

from .data_loading import (DataPaths, list_patch_ids, load_config,
                           load_metadata, load_s2, load_target, parse_dates)
from . import preprocessing as pp
from . import visualization as viz


def _class_distribution(paths, ids):
    counts = Counter()
    for pid in ids:
        t = load_target(paths, pid)
        for c, n in zip(*np.unique(t, return_counts=True)):
            counts[int(c)] += int(n)
    return dict(sorted(counts.items()))


def _ndvi_profiles(paths, ids, cfg, max_patches=25):
    """Mean NDVI per class per timestep, aggregated over a subset of patches."""
    scale = cfg["preprocess"]["reflectance_scale"]
    void = cfg["labels"]["void_class"]
    sums, pix = {}, {}
    for pid in ids[:max_patches]:
        s2 = load_s2(paths, pid)
        target = load_target(paths, pid)
        refl = pp.to_reflectance(s2, scale=scale)
        red, nir = refl[:, pp.B_RED], refl[:, pp.B_NIR]
        ndvi = (nir - red) / (nir + red + 1e-6)          # (T, H, W)
        for c in np.unique(target):
            if c == void:
                continue
            m = target == c
            if m.sum() == 0:
                continue
            series = ndvi[:, m].mean(axis=1)             # (T,)
            sums[int(c)] = sums.get(int(c), 0) + series * m.sum()
            pix[int(c)] = pix.get(int(c), 0) + m.sum()
    return {c: sums[c] / pix[c] for c in sums}


def main(config_path: str = "configs/config.yaml") -> None:
    cfg = load_config(config_path)
    paths = DataPaths.from_config(cfg)
    label_names = {int(k): v for k, v in cfg["labels"]["names"].items()}
    fig_dir = cfg["paths"]["figures"]

    ids = list_patch_ids(paths)
    meta = load_metadata(paths)

    # --- summary stats ------------------------------------------------------
    s2_example = load_s2(paths, ids[0], mmap=True)
    class_counts = _class_distribution(paths, ids)
    total = sum(class_counts.values())
    summary = {
        "n_patches": len(ids),
        "s2_shape_example": list(s2_example.shape),
        "patch_size": cfg["data"]["patch_size"],
        "n_bands": cfg["data"]["n_bands"],
        "n_timesteps": int(s2_example.shape[0]),
        "tiles": sorted(meta["TILE"].unique().tolist())
        if "TILE" in meta else None,
        "class_pixel_counts": class_counts,
        "class_pixel_fraction": {k: v / total for k, v in class_counts.items()},
    }
    os.makedirs(cfg["paths"]["metrics"], exist_ok=True)
    with open(os.path.join(cfg["paths"]["metrics"], "eda_summary.json"),
              "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("EDA summary:", json.dumps(summary, indent=2)[:600], "...")

    # --- figures ------------------------------------------------------------
    viz.plot_class_distribution(
        class_counts, label_names,
        os.path.join(fig_dir, "class_distribution.png"))

    profiles = _ndvi_profiles(paths, ids, cfg)
    named = {label_names.get(c, str(c)): s for c, s in profiles.items()}
    viz.plot_ndvi_profiles(named, None, os.path.join(fig_dir, "ndvi_profiles.png"))

    # a couple of representative patches: pick the one with most classes
    best = max(ids, key=lambda p: len(np.unique(load_target(paths, p))))
    for pid in [ids[0], best]:
        s2 = load_s2(paths, pid)
        t = pp.least_cloudy_timestep(s2, cfg["preprocess"]["reflectance_scale"])
        viz.plot_patch_overview(
            s2, load_target(paths, pid), t, label_names,
            os.path.join(fig_dir, f"patch_overview_{pid}.png"),
            title=f"Patch {pid}")

    viz.plot_aoi(meta, os.path.join(fig_dir, "aoi_footprints.png"))
    print(f"Figures written to {fig_dir}")


if __name__ == "__main__":
    main()
