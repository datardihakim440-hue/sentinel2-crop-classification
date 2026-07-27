"""Evaluation metrics and reporting.

Provides the metric computations used across the project (overall accuracy,
per-class precision/recall/F1, per-class IoU + mIoU, confusion matrix) plus a
CLI entry point that loads a trained model and evaluates it densely on a split.
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def per_class_iou(cm: np.ndarray) -> np.ndarray:
    """IoU per class from a confusion matrix (rows = true, cols = pred)."""
    intersection = np.diag(cm).astype(np.float64)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, intersection / union, np.nan)
    return iou


def compute_metrics(y_true, y_pred, labels, label_names) -> dict:
    """Return a JSON-serialisable dict of all evaluation metrics."""
    labels = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    iou = per_class_iou(cm)

    per_class = {}
    for i, lab in enumerate(labels):
        per_class[int(lab)] = {
            "name": label_names.get(lab, str(lab)),
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "iou": None if np.isnan(iou[i]) else float(iou[i]),
            "support": int(support[i]),
        }

    present = support > 0  # only average over classes that actually appear
    return {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1[present])) if present.any() else 0.0,
        "mean_iou": float(np.nanmean(iou)) if present.any() else 0.0,
        "n_samples": int(len(y_true)),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def print_report(y_true, y_pred, labels, label_names) -> None:
    target_names = [label_names.get(l, str(l)) for l in labels]
    print(classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, zero_division=0
    ))


def save_metrics(metrics: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


# --------------------------------------------------------------------------- #
# CLI: evaluate a trained model densely on a split
# --------------------------------------------------------------------------- #
def _labels_for_eval(cfg: dict) -> list[int]:
    names = cfg["labels"]["names"]
    void = cfg["labels"]["void_class"]
    labels = [int(k) for k in names if int(k) != void]
    if cfg["labels"]["ignore_background"]:
        labels = [l for l in labels if l != cfg["labels"]["background_class"]]
    return sorted(labels)


def evaluate_split(cfg: dict, split_name: str, base_dir: str = ".") -> dict:
    import joblib

    from .data_loading import DataPaths
    from .dataset import build_dataset
    from .splits import read_split

    paths = DataPaths.from_config(cfg, base_dir)
    model = joblib.load(os.path.join(cfg["paths"]["models"], "rf_model.joblib"))
    rng = np.random.default_rng(cfg["seed"])

    patch_ids = read_split(cfg["paths"]["splits"], split_name)
    X, y, _ = build_dataset(paths, patch_ids, cfg, sample=False, rng=rng)
    y_pred = model.predict(X)

    labels = _labels_for_eval(cfg)
    label_names = {int(k): v for k, v in cfg["labels"]["names"].items()}
    metrics = compute_metrics(y, y_pred, labels, label_names)
    metrics["split"] = split_name
    metrics["patch_ids"] = patch_ids
    return metrics


def evaluate_split_spatial(cfg: dict, split_name: str, base_dir: str = ".",
                           window: int | None = None) -> dict:
    """Evaluate densely *with* spatial majority-filter post-processing.

    Predicts each patch as a 2-D map, applies the majority filter, then scores
    over all valid (non-void) pixels. Returns metrics with an added
    ``postprocess_window`` field. Lets us report raw-vs-smoothed cleanly.
    """
    import joblib

    from .data_loading import DataPaths
    from .dataset import build_patch_matrix
    from .postprocess import majority_filter
    from .splits import read_split

    paths = DataPaths.from_config(cfg, base_dir)
    model = joblib.load(os.path.join(cfg["paths"]["models"], "rf_model.joblib"))
    labels = _labels_for_eval(cfg)
    void = cfg["labels"]["void_class"]
    if window is None:
        window = cfg["postprocess"]["window"]

    y_true_all, y_pred_all = [], []
    for pid in read_split(cfg["paths"]["splits"], split_name):
        X, target, valid = build_patch_matrix(paths, pid, cfg)
        pred2d = model.predict(X).reshape(target.shape)
        pred2d = majority_filter(pred2d, labels, window=window, void_class=void)
        y_true_all.append(target[valid])
        y_pred_all.append(pred2d[valid])

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    label_names = {int(k): v for k, v in cfg["labels"]["names"].items()}
    metrics = compute_metrics(y_true, y_pred, labels, label_names)
    metrics["split"] = split_name
    metrics["postprocess_window"] = window
    return metrics


if __name__ == "__main__":
    import argparse

    from .data_loading import load_config

    ap = argparse.ArgumentParser(description="Evaluate trained RF on a split.")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    label_names = {int(k): v for k, v in cfg["labels"]["names"].items()}
    metrics = evaluate_split(cfg, args.split)

    print(f"\n=== {args.split.upper()} ===")
    print(f"Overall accuracy : {metrics['overall_accuracy']:.4f}")
    print(f"Macro F1         : {metrics['macro_f1']:.4f}")
    print(f"Mean IoU         : {metrics['mean_iou']:.4f}")
    out = os.path.join(cfg["paths"]["metrics"], f"metrics_{args.split}.json")
    save_metrics(metrics, out)
    print(f"Saved -> {out}")
