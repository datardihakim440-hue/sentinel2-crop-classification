"""Plotting utilities for EDA and model-output visualisation.

All functions save figures to disk (headless-friendly, ``Agg`` backend) and are
deterministic so the notebook and CLI produce identical artefacts.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from .preprocessing import B_BLUE, B_GREEN, B_RED, to_reflectance

# A distinct qualitative palette for the 20 PASTIS classes (0..19).
_CLASS_COLORS = [
    "#000000", "#4daf4a", "#ffff33", "#ff7f00", "#984ea3", "#e41a1c",
    "#a6cee3", "#f781bf", "#6a3d9a", "#b15928", "#1f78b4", "#33a02c",
    "#fb9a99", "#fdbf6f", "#cab2d6", "#00ced1", "#8dd3c7", "#bc80bd",
    "#d9d9d9", "#ffffff",
]


def _ensure(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def label_cmap():
    cmap = ListedColormap(_CLASS_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 20.5, 1), cmap.N)
    return cmap, norm


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #
def plot_class_distribution(class_counts: dict, label_names: dict, out: str) -> None:
    """Bar chart of pixel counts per class (log scale)."""
    items = sorted(class_counts.items())
    ids = [k for k, _ in items]
    vals = [v for _, v in items]
    names = [f"{k}:{label_names.get(k, k)}" for k in ids]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(ids)), vals, color="#377eb8")
    ax.set_yscale("log")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(names, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Pixel count (log)")
    ax.set_title("Class distribution across all patches")
    fig.tight_layout()
    _ensure(out)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def rgb_composite(s2: np.ndarray, t: int, scale: float = 10000.0,
                  gain: float = 3.0) -> np.ndarray:
    """Build a display RGB image from timestep ``t`` (B4,B3,B2)."""
    refl = to_reflectance(s2[t:t + 1], scale=scale)[0]
    rgb = np.stack([refl[B_RED], refl[B_GREEN], refl[B_BLUE]], axis=-1)
    rgb = np.clip(rgb * gain, 0, 1)  # simple brightness gain for visibility
    return rgb


def plot_patch_overview(s2: np.ndarray, target: np.ndarray, t: int,
                        label_names: dict, out: str, title: str = "") -> None:
    """RGB composite next to its ground-truth label map."""
    cmap, norm = label_cmap()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb_composite(s2, t))
    axes[0].set_title(f"Sentinel-2 RGB (t={t})")
    axes[0].axis("off")
    im = axes[1].imshow(target, cmap=cmap, norm=norm)
    axes[1].set_title("Ground truth")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, ticks=range(20))
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    _ensure(out)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_ndvi_profiles(profiles: dict, dates, out: str) -> None:
    """Mean NDVI temporal profile per class (phenology signatures)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(next(iter(profiles.values())))) if dates is None else dates
    for name, series in profiles.items():
        ax.plot(range(len(series)), series, marker="o", ms=3, label=name)
    ax.set_xlabel("Acquisition index (Sep 2018 -> Oct 2019)")
    ax.set_ylabel("Mean NDVI")
    ax.set_title("Class NDVI phenology profiles")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _ensure(out)
    fig.savefig(out, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Model outputs
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(cm, labels, label_names, out: str,
                          normalize: bool = True) -> None:
    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        row = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)

    names = [label_names.get(l, str(l)) for l in labels]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalize else ""))
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    _ensure(out)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_prediction_triptych(s2, target, pred, t, label_names, out,
                             void_class=19) -> None:
    """Sentinel-2 RGB | ground truth | prediction (void masked out)."""
    cmap, norm = label_cmap()
    pred_disp = np.where(target == void_class, void_class, pred)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb_composite(s2, t))
    axes[0].set_title(f"Sentinel-2 RGB (t={t})")
    axes[1].imshow(target, cmap=cmap, norm=norm)
    axes[1].set_title("Ground truth")
    im = axes[2].imshow(pred_disp, cmap=cmap, norm=norm)
    axes[2].set_title("Prediction")
    for a in axes:
        a.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.02, ticks=range(20))
    _ensure(out)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_comparison(s2, target, pred_raw, pred_smooth, t,
                               label_names, out, void_class=19) -> None:
    """RGB | ground truth | raw prediction | spatially-smoothed prediction."""
    cmap, norm = label_cmap()
    raw = np.where(target == void_class, void_class, pred_raw)
    smooth = np.where(target == void_class, void_class, pred_smooth)

    fig, axes = plt.subplots(1, 4, figsize=(19, 5))
    axes[0].imshow(rgb_composite(s2, t))
    axes[0].set_title(f"Sentinel-2 RGB (t={t})")
    axes[1].imshow(target, cmap=cmap, norm=norm)
    axes[1].set_title("Ground truth")
    axes[2].imshow(raw, cmap=cmap, norm=norm)
    axes[2].set_title("Prediction (per-pixel)")
    im = axes[3].imshow(smooth, cmap=cmap, norm=norm)
    axes[3].set_title("Prediction (+ spatial smoothing)")
    for a in axes:
        a.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.02, ticks=range(20))
    _ensure(out)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importances(importances, out: str, top: int = 20) -> None:
    top_items = importances[:top][::-1]
    names = [d["feature"] for d in top_items]
    vals = [d["importance"] for d in top_items]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(range(len(names)), vals, color="#4daf4a")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Gini importance")
    ax.set_title(f"Top {top} feature importances")
    fig.tight_layout()
    _ensure(out)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_aoi(metadata, out: str) -> None:
    """Plot patch footprints to show approximate geographic coverage."""
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        metadata.plot(ax=ax, facecolor="none", edgecolor="#e41a1c")
        ax.set_title("AOI: patch footprints (tile t31tfm)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.tight_layout()
        _ensure(out)
        fig.savefig(out, dpi=130)
        plt.close(fig)
    except Exception as exc:  # geometry may be absent in DataFrame fallback
        print(f"[plot_aoi] skipped: {exc}")
