"""Create reproducible train / validation / test splits.

Strategy: reuse the **official PASTIS 5-fold** assignment stored per patch in
``metadata.geojson`` (column ``Fold``). This is preferable to a random pixel or
patch split because:

* PASTIS folds are geographically block-designed to limit spatial
  autocorrelation leakage between train and test.
* Reusing them makes results comparable to the published benchmark protocol.
* The split is deterministic and needs no random seed.

Splitting is done at the **patch** level (not pixel level): all pixels of a
patch stay together, so no pixel from a training patch can leak into evaluation.

Run as a script to (re)write ``splits/{train,val,test}.txt``.
"""

from __future__ import annotations

import os

from .data_loading import DataPaths, list_patch_ids, load_config, load_metadata


def make_splits(cfg: dict, base_dir: str = ".") -> dict[str, list[int]]:
    paths = DataPaths.from_config(cfg, base_dir)
    available = set(list_patch_ids(paths))
    meta = load_metadata(paths)

    sp = cfg["split"]
    fold_of = dict(zip(meta["ID_PATCH"].astype(int), meta["Fold"].astype(int)))

    def collect(folds):
        ids = [pid for pid, fold in fold_of.items()
               if fold in folds and pid in available]
        return sorted(ids)

    return {
        "train": collect(sp["train_folds"]),
        "val": collect(sp["val_folds"]),
        "test": collect(sp["test_folds"]),
    }


def write_splits(splits: dict[str, list[int]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, ids in splits.items():
        fp = os.path.join(out_dir, f"{name}.txt")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write("\n".join(str(i) for i in ids) + "\n")


def read_split(out_dir: str, name: str) -> list[int]:
    fp = os.path.join(out_dir, f"{name}.txt")
    with open(fp, "r", encoding="utf-8") as fh:
        return [int(line) for line in fh if line.strip()]


if __name__ == "__main__":
    cfg = load_config()
    splits = make_splits(cfg)
    write_splits(splits, cfg["paths"]["splits"])
    for name, ids in splits.items():
        print(f"{name:5s}: {len(ids):3d} patches -> {ids}")
