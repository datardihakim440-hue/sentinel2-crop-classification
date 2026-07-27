"""Assemble per-pixel training/evaluation datasets from patches.

Bridges the gap between single-patch feature engineering (``preprocessing``)
and the flat ``(N, F)`` design matrices that scikit-learn expects. Handles the
two sampling regimes:

* **Training** – randomly subsample a capped number of labelled pixels per
  class per patch (keeps memory bounded, mitigates the dominance of a few
  patches, and speeds up fitting).
* **Validation / test** – use *all* valid pixels so metrics reflect true
  dense-prediction performance.

Void pixels (class 19) are always excluded; they are unlabelled parcel borders.
"""

from __future__ import annotations

import numpy as np

from . import preprocessing as pp
from .data_loading import DataPaths, load_s2, load_target


def _valid_mask(target: np.ndarray, void_class: int,
                ignore_background: bool, background_class: int) -> np.ndarray:
    mask = target != void_class
    if ignore_background:
        mask &= target != background_class
    return mask


def _sample_indices(target_flat: np.ndarray, valid_flat: np.ndarray,
                    max_per_class: int, rng: np.random.Generator) -> np.ndarray:
    """Choose flat pixel indices, capped per class, among valid pixels."""
    chosen = []
    valid_idx = np.nonzero(valid_flat)[0]
    for cls in np.unique(target_flat[valid_idx]):
        cls_idx = valid_idx[target_flat[valid_idx] == cls]
        if len(cls_idx) > max_per_class:
            cls_idx = rng.choice(cls_idx, size=max_per_class, replace=False)
        chosen.append(cls_idx)
    return np.concatenate(chosen) if chosen else np.array([], dtype=int)


def build_dataset(paths: DataPaths, patch_ids, cfg: dict, *,
                  sample: bool, rng: np.random.Generator):
    """Build ``(X, y, groups)`` for a list of patch IDs.

    ``groups`` holds the source patch ID per row (useful for group-aware
    diagnostics). When ``sample`` is True the per-class cap is applied.
    """
    void = cfg["labels"]["void_class"]
    bg = cfg["labels"]["background_class"]
    ignore_bg = cfg["labels"]["ignore_background"]
    max_per_class = cfg["sampling"]["max_pixels_per_class_per_patch"]

    X_parts, y_parts, g_parts = [], [], []
    for pid in patch_ids:
        s2 = load_s2(paths, pid)
        target = load_target(paths, pid)

        stack, _ = pp.build_feature_stack(s2, cfg)
        X = pp.stack_to_pixels(stack)              # (H*W, F)
        y = target.reshape(-1)                     # (H*W,)

        valid = _valid_mask(target, void, ignore_bg, bg).reshape(-1)
        if sample:
            idx = _sample_indices(y, valid, max_per_class, rng)
        else:
            idx = np.nonzero(valid)[0]

        X_parts.append(X[idx])
        y_parts.append(y[idx])
        g_parts.append(np.full(len(idx), pid, dtype=np.int64))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    groups = np.concatenate(g_parts, axis=0)
    return X, y, groups


def build_patch_matrix(paths: DataPaths, patch_id: int, cfg: dict):
    """Return dense ``(X_all, target_2d, valid_mask_2d)`` for one patch.

    Used at inference time to render full-patch prediction maps.
    """
    s2 = load_s2(paths, patch_id)
    target = load_target(paths, patch_id)
    stack, _ = pp.build_feature_stack(s2, cfg)
    X = pp.stack_to_pixels(stack)
    valid = _valid_mask(
        target,
        cfg["labels"]["void_class"],
        cfg["labels"]["ignore_background"],
        cfg["labels"]["background_class"],
    )
    return X, target, valid
