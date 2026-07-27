"""Generate dense predictions + output visualisations for the test split.

Run::

    python -m src.predict          # evaluates test split, writes figures + maps

Produces:
    outputs/metrics/metrics_test.json
    outputs/figures/confusion_matrix_test.png
    outputs/figures/feature_importances.png
    outputs/figures/prediction_<id>.png            (triptychs)
    outputs/predictions/PRED_<id>.npy              (predicted label maps)
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np

from .data_loading import DataPaths, load_config, load_s2
from .dataset import build_patch_matrix
from .evaluate import (_labels_for_eval, evaluate_split,
                       evaluate_split_spatial, save_metrics)
from .postprocess import majority_filter
from .preprocessing import least_cloudy_timestep
from .splits import read_split
from . import visualization as viz


def predict_patch(model, paths, patch_id, cfg):
    """Predict raw + spatially-smoothed label maps for one patch.

    Void pixels are kept as void in both. Returns ``(raw, smoothed, target)``.
    """
    X, target, valid = build_patch_matrix(paths, patch_id, cfg)
    void = cfg["labels"]["void_class"]
    raw = np.where(valid, model.predict(X).reshape(target.shape), void)

    if cfg["postprocess"]["spatial_majority"]:
        labels = _labels_for_eval(cfg)
        smoothed = majority_filter(raw, labels,
                                   window=cfg["postprocess"]["window"],
                                   void_class=void)
        smoothed = np.where(valid, smoothed, void)
    else:
        smoothed = raw
    return raw, smoothed, target


def main(config_path: str = "configs/config.yaml", n_examples: int = 6) -> None:
    cfg = load_config(config_path)
    paths = DataPaths.from_config(cfg)
    label_names = {int(k): v for k, v in cfg["labels"]["names"].items()}
    labels = _labels_for_eval(cfg)

    model = joblib.load(os.path.join(cfg["paths"]["models"], "rf_model.joblib"))

    # --- test metrics: raw (per-pixel) and spatially-smoothed --------------
    metrics = evaluate_split(cfg, "test")
    save_metrics(metrics, os.path.join(cfg["paths"]["metrics"], "metrics_test.json"))
    print(f"TEST raw       acc={metrics['overall_accuracy']:.4f}  "
          f"macroF1={metrics['macro_f1']:.4f}  mIoU={metrics['mean_iou']:.4f}")

    if cfg["postprocess"]["spatial_majority"]:
        sm = evaluate_split_spatial(cfg, "test")
        save_metrics(sm, os.path.join(cfg["paths"]["metrics"],
                                      "metrics_test_smoothed.json"))
        print(f"TEST smoothed  acc={sm['overall_accuracy']:.4f}  "
              f"macroF1={sm['macro_f1']:.4f}  mIoU={sm['mean_iou']:.4f}  "
              f"(window={sm['postprocess_window']})")

    # --- confusion matrix + importances ------------------------------------
    viz.plot_confusion_matrix(
        metrics["confusion_matrix"], labels, label_names,
        os.path.join(cfg["paths"]["figures"], "confusion_matrix_test.png"))

    fi_path = os.path.join(cfg["paths"]["metrics"], "feature_importances.json")
    if os.path.exists(fi_path):
        with open(fi_path, encoding="utf-8") as fh:
            fi = json.load(fh)["feature_importances"]
        viz.plot_feature_importances(
            fi, os.path.join(cfg["paths"]["figures"], "feature_importances.png"))

    # --- per-patch prediction maps + comparison figures --------------------
    test_ids = read_split(cfg["paths"]["splits"], "test")
    os.makedirs(cfg["paths"]["predictions"], exist_ok=True)
    void = cfg["labels"]["void_class"]
    for pid in test_ids[:n_examples]:
        raw, smoothed, target = predict_patch(model, paths, pid, cfg)
        np.save(os.path.join(cfg["paths"]["predictions"], f"PRED_{pid}.npy"), smoothed)
        s2 = load_s2(paths, pid)
        t = least_cloudy_timestep(s2, cfg["preprocess"]["reflectance_scale"])
        viz.plot_prediction_triptych(
            s2, target, smoothed, t, label_names,
            os.path.join(cfg["paths"]["figures"], f"prediction_{pid}.png"),
            void_class=void)
        viz.plot_prediction_comparison(
            s2, target, raw, smoothed, t, label_names,
            os.path.join(cfg["paths"]["figures"], f"comparison_{pid}.png"),
            void_class=void)
    print(f"Wrote predictions + figures for {min(n_examples, len(test_ids))} "
          f"test patches.")


if __name__ == "__main__":
    main()
