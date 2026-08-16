#!/usr/bin/env python3
"""
Export the exact calibration set (120 images, 20 per class, drawn from the
TRAIN split only) that pipeline.py feeds to static int8 quantisation, into:

    calibration/<class>/   (copies of the original images, 20 each)
    calibration/MANIFEST.txt

Run:  python export_calibration.py

It reuses pipeline.py's own loader / split / RNG so the selected images are
bit-for-bit identical to what quantize_static saw during the pipeline run.
"""

import shutil
from pathlib import Path

import numpy as np

import pipeline  # CLASSES, ROOT, rng, load_dataset, stratified_split

OUT = pipeline.HERE / "calibration"
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def file_paths(root):
    """File list in EXACTLY the order pipeline.load_dataset builds X:
    class order, then sorted paths within each class."""
    root = Path(root)
    paths = []
    for c in pipeline.CLASSES:
        paths += sorted(f for f in (root / c).glob("*")
                        if f.suffix.lower() in EXTS)
    return paths


def main():
    X, y = pipeline.load_dataset(pipeline.ROOT)
    if X is None:
        raise SystemExit(f"dataset not found at {pipeline.ROOT}")
    paths = file_paths(pipeline.ROOT)
    assert len(paths) == len(y), "path/array length mismatch"

    # Mirror main()'s exact RNG sequence: split first (6 permutations), then
    # 6 choices -> identical calibration indices.
    tr, va, te = pipeline.stratified_split(y)
    calib_idx = []
    for c in range(6):
        calib_idx += pipeline.rng.choice(
            np.where(y[tr] == c)[0], pipeline.CALIB_PER_CLASS, replace=False
        ).tolist()
    # calib_idx are positions INSIDE the train subset; map to full indices.
    full_idx = tr[calib_idx]
    assert len(full_idx) == 6 * pipeline.CALIB_PER_CLASS

    if OUT.exists():
        shutil.rmtree(OUT)
    counts = {c: 0 for c in pipeline.CLASSES}
    for fi in full_idx:
        c = pipeline.CLASSES[int(y[fi])]
        d = OUT / c
        d.mkdir(parents=True, exist_ok=True)
        counts[c] += 1
        shutil.copy2(paths[fi], d / f"{counts[c]:02d}_{paths[fi].name}")

    lines = [
        "# calibration set exported by export_calibration.py",
        f"# total {len(full_idx)} images, {pipeline.CALIB_PER_CLASS}/class, "
        "train split only (no test/val leakage)",
        "# class,count",
    ]
    for c in pipeline.CLASSES:
        lines.append(f"{c},{counts[c]}")
    (OUT / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"exported {len(full_idx)} calibration images to {OUT}")
    for c in pipeline.CLASSES:
        print(f"  {c:<16} {counts[c]}")


if __name__ == "__main__":
    main()
