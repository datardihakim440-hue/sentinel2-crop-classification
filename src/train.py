"""Train the Random-Forest crop classifier.

Usage::

    python -m src.train --config configs/config.yaml

Steps:
    1. Build (or reuse) the reproducible patch-level split.
    2. Assemble a class-balanced sampled pixel training set.
    3. Fit a RandomForestClassifier on temporal-spectral features.
    4. Evaluate on the validation split and persist model + metrics + feature
       importances.

Why Random Forest: the brief explicitly prefers a well-reasoned baseline over a
complex model. RF handles the moderate-dimensional engineered features well,
needs no GPU, is robust to feature scaling, exposes feature importances for
interpretation, and trains in minutes on this subset.
"""

from __future__ import annotations

import os
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .data_loading import DataPaths, load_config
from .dataset import build_dataset
from .evaluate import _labels_for_eval, compute_metrics, save_metrics
from .preprocessing import feature_names
from .splits import make_splits, read_split, write_splits


def build_model(cfg: dict) -> RandomForestClassifier:
    rf = cfg["model"]["random_forest"]
    return RandomForestClassifier(
        n_estimators=rf["n_estimators"],
        max_depth=rf["max_depth"],
        min_samples_leaf=rf["min_samples_leaf"],
        max_features=rf["max_features"],
        class_weight=rf["class_weight"],
        n_jobs=rf["n_jobs"],
        random_state=cfg["seed"],
    )


def ensure_splits(cfg: dict, base_dir: str = ".") -> dict:
    split_dir = cfg["paths"]["splits"]
    needed = ["train", "val", "test"]
    if all(os.path.exists(os.path.join(split_dir, f"{n}.txt")) for n in needed):
        return {n: read_split(split_dir, n) for n in needed}
    splits = make_splits(cfg, base_dir)
    write_splits(splits, split_dir)
    return splits


def main(config_path: str = "configs/config.yaml", base_dir: str = ".") -> dict:
    cfg = load_config(config_path)
    paths = DataPaths.from_config(cfg, base_dir)
    rng = np.random.default_rng(cfg["seed"])

    splits = ensure_splits(cfg, base_dir)
    print(f"Split sizes (patches): "
          f"train={len(splits['train'])} "
          f"val={len(splits['val'])} test={len(splits['test'])}")

    # ---- assemble training data (sampled) ---------------------------------
    t0 = time.time()
    X_tr, y_tr, _ = build_dataset(paths, splits["train"], cfg, sample=True, rng=rng)
    print(f"Training matrix: {X_tr.shape}  ({time.time()-t0:.1f}s to build)")
    cls, cnt = np.unique(y_tr, return_counts=True)
    print("Training class counts:", dict(zip(cls.tolist(), cnt.tolist())))

    # ---- fit ---------------------------------------------------------------
    model = build_model(cfg)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    print(f"Fitted RandomForest in {time.time()-t0:.1f}s")

    # ---- persist model + feature importances ------------------------------
    os.makedirs(cfg["paths"]["models"], exist_ok=True)
    model_path = os.path.join(cfg["paths"]["models"], "rf_model.joblib")
    joblib.dump(model, model_path, compress=3)
    print(f"Saved model -> {model_path}")

    fnames = feature_names(cfg)
    order = np.argsort(model.feature_importances_)[::-1]
    importances = [
        {"feature": fnames[i], "importance": float(model.feature_importances_[i])}
        for i in order
    ]
    save_metrics({"feature_importances": importances},
                 os.path.join(cfg["paths"]["metrics"], "feature_importances.json"))
    print("Top 10 features:",
          [importances[i]["feature"] for i in range(min(10, len(importances)))])

    # ---- validation metrics -----------------------------------------------
    X_val, y_val, _ = build_dataset(paths, splits["val"], cfg, sample=False, rng=rng)
    y_pred = model.predict(X_val)
    labels = _labels_for_eval(cfg)
    label_names = {int(k): v for k, v in cfg["labels"]["names"].items()}
    metrics = compute_metrics(y_val, y_pred, labels, label_names)
    metrics["split"] = "val"
    save_metrics(metrics, os.path.join(cfg["paths"]["metrics"], "metrics_val.json"))

    print(f"\nVALIDATION  acc={metrics['overall_accuracy']:.4f}  "
          f"macroF1={metrics['macro_f1']:.4f}  mIoU={metrics['mean_iou']:.4f}")
    return metrics


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train RF crop classifier.")
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
