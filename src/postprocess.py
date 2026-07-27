"""Spatial post-processing for per-pixel predictions.

The Random Forest classifies each pixel independently, which produces
salt-and-pepper noise inside fields (visible in the prediction triptychs). Crop
parcels are spatially coherent, so a simple **spatial majority filter** — assign
each pixel the most common predicted class in its local neighbourhood — injects
that spatial prior and typically improves accuracy and mIoU.

This is an *unsupervised* post-process: it uses only the model's own predictions,
never the ground-truth labels, so there is no label leakage. (True parcel-level
voting would require parcel polygons, which are not provided in this subset.)

Implementation is fully vectorised: for each candidate class we compute a
windowed count with a uniform filter, then take the per-pixel argmax.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def majority_filter(pred: np.ndarray, labels, window: int = 5,
                    void_class: int = 19) -> np.ndarray:
    """Return a spatially smoothed copy of a 2-D prediction map.

    Parameters
    ----------
    pred : (H, W) int array of predicted class IDs.
    labels : iterable of candidate class IDs to consider (void excluded).
    window : odd window size for the neighbourhood vote.
    void_class : label kept unchanged (parcel borders / ignored pixels).
    """
    labels = [int(l) for l in labels if int(l) != void_class]
    void_mask = pred == void_class

    # Windowed count (as a fraction) of each class in the local neighbourhood.
    counts = np.empty((len(labels),) + pred.shape, dtype=np.float32)
    for i, c in enumerate(labels):
        counts[i] = uniform_filter((pred == c).astype(np.float32),
                                   size=window, mode="reflect")

    smoothed = np.asarray(labels)[np.argmax(counts, axis=0)]
    smoothed = smoothed.astype(pred.dtype)
    smoothed[void_mask] = void_class
    return smoothed
