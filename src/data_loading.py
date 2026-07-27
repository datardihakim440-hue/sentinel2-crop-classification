"""Data loading utilities for the PASTIS Sentinel-2 subset.

The dataset is organised as::

    <root>/DATA_S2/S2_<patch_id>.npy        # (T, 10, H, W)  int16
    <root>/ANNOTATIONS/TARGET_<patch_id>.npy # (1, H, W)      uint8
    <root>/metadata.geojson                  # per-patch metadata (folds, dates, geometry)

This module keeps *only* the responsibility of reading files and exposing them
as clean NumPy arrays / DataFrames. Preprocessing and feature engineering live
in their own modules so each stage can be tested independently.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str = "configs/config.yaml") -> dict:
    """Load the YAML configuration file."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class DataPaths:
    """Resolved absolute paths for the dataset."""

    root: str
    s2_dir: str
    ann_dir: str
    metadata: str

    @classmethod
    def from_config(cls, cfg: dict, base_dir: str = ".") -> "DataPaths":
        d = cfg["data"]
        root = os.path.join(base_dir, d["root"])
        return cls(
            root=root,
            s2_dir=os.path.join(root, d["s2_dir"]),
            ann_dir=os.path.join(root, d["ann_dir"]),
            metadata=os.path.join(root, d["metadata"]),
        )


# --------------------------------------------------------------------------- #
# Patch discovery
# --------------------------------------------------------------------------- #
_PATCH_RE = re.compile(r"S2_(\d+)\.npy$")


def list_patch_ids(paths: DataPaths) -> list[int]:
    """Return the sorted list of patch IDs that have both S2 and target files."""
    ids = []
    for f in glob.glob(os.path.join(paths.s2_dir, "S2_*.npy")):
        m = _PATCH_RE.search(os.path.basename(f))
        if not m:
            continue
        pid = int(m.group(1))
        target = os.path.join(paths.ann_dir, f"TARGET_{pid}.npy")
        if os.path.exists(target):
            ids.append(pid)
    return sorted(ids)


# --------------------------------------------------------------------------- #
# Array loading
# --------------------------------------------------------------------------- #
def load_s2(paths: DataPaths, patch_id: int, mmap: bool = False) -> np.ndarray:
    """Load a Sentinel-2 patch as ``(T, C, H, W)`` int16 array."""
    fp = os.path.join(paths.s2_dir, f"S2_{patch_id}.npy")
    return np.load(fp, mmap_mode="r" if mmap else None)


def load_target(paths: DataPaths, patch_id: int) -> np.ndarray:
    """Load the semantic label map as a 2-D ``(H, W)`` array.

    The stored array has a leading annotation-layer axis of size 1; we squeeze
    it because only the 0th layer is provided for this assignment.
    """
    fp = os.path.join(paths.ann_dir, f"TARGET_{patch_id}.npy")
    arr = np.load(fp)
    if arr.ndim == 3:
        arr = arr[0]
    return arr.astype(np.int64)


def describe_patch(paths: DataPaths, patch_id: int) -> dict:
    """Cheap inspection of a single patch (uses memory-mapping, no full read)."""
    s2 = load_s2(paths, patch_id, mmap=True)
    tgt = load_target(paths, patch_id)
    classes, counts = np.unique(tgt, return_counts=True)
    return {
        "patch_id": patch_id,
        "s2_shape": tuple(s2.shape),
        "s2_dtype": str(s2.dtype),
        "target_shape": tuple(tgt.shape),
        "classes": classes.tolist(),
        "class_pixel_counts": dict(zip(classes.tolist(), counts.tolist())),
    }


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def load_metadata(paths: DataPaths):
    """Load ``metadata.geojson`` as a GeoDataFrame.

    Falls back to a plain DataFrame (geometry dropped) if geopandas is missing,
    so the core workflow never hard-depends on the geospatial stack.
    """
    try:
        import geopandas as gpd

        gdf = gpd.read_file(paths.metadata)
        return gdf
    except Exception:  # pragma: no cover - optional dependency path
        import json

        import pandas as pd

        with open(paths.metadata, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        rows = [f["properties"] for f in raw["features"]]
        return pd.DataFrame(rows)


def parse_dates(metadata, patch_id: int) -> Optional[np.ndarray]:
    """Return acquisition dates (as ``YYYYMMDD`` ints) for a patch, or None."""
    row = metadata[metadata["ID_PATCH"] == patch_id]
    if len(row) == 0:
        return None
    dates = row.iloc[0]["dates-S2"]
    if isinstance(dates, str):
        import json

        dates = json.loads(dates.replace("'", '"'))
    ordered = [dates[k] for k in sorted(dates, key=int)]
    return np.asarray(ordered, dtype=np.int64)


if __name__ == "__main__":
    cfg = load_config()
    paths = DataPaths.from_config(cfg)
    ids = list_patch_ids(paths)
    print(f"Found {len(ids)} patches. First: {ids[:5]}")
    print(describe_patch(paths, ids[0]))
