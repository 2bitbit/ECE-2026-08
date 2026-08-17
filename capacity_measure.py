#!/usr/bin/env python3
"""
Same-machine, same-config capacity A/B measurement (narrow 115k vs wide 240k).

The committed capacity numbers (fig_capacity_ab.py) came from machine A
(Intel i7-10xxx, commit 8a9a5de): narrow 0.8667/0.8306 vs wide 0.8250/0.8139.
Reproducing pipeline_v2.py on THIS machine gave wide 0.8917/0.8806 — a 6.7 pt
swing — which means the "more params = worse" conclusion does NOT hold across
machines (this machine's wide model is BETTER than its narrow one). To make
the A/B honest we re-run the NARROW model here with the exact v2 quantisation
config (per-tensor QInt8 + QOperator + ORT_ENABLE_ALL) so both models are
measured on the same machine, same seed, same epochs, same quantisation.

Run:  python capacity_measure.py    ->  writes capacity_measure.json
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pipeline_v2 as v2
from starter_p1 import CLASSES, IMG

SEED = 0
EPOCHS = 20
HERE = Path(__file__).resolve().parent


def main():
    v2.rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X, y = v2.load_dataset(v2.ROOT)
    if X is None:
        raise SystemExit(f"dataset not found at {v2.ROOT}")
    tr, va, te = v2.stratified_split(y)
    Xt = torch.tensor(X[tr]); yt = torch.tensor(y[tr])
    Xv = torch.tensor(X[va]); yv = torch.tensor(y[va])
    Xe = torch.tensor(X[te]); ye = torch.tensor(y[te])
    print(f"split train {len(tr)} / val {len(va)} / test {len(te)}")

    # ---- narrow model, EXACTLY the v2 config (per-tensor + QOperator) ----
    model = v2.task4_model_wide if False else _narrow()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"narrow model -> {n_par:,} parameters")
    model, _ = v2.train_model(model, Xt, yt, Xv, yv, EPOCHS)
    acc_pt, preds_pt = v2.evaluate_pt(model, Xe, ye)
    print(f"narrow fp32 test accuracy  {acc_pt:.4f}")

    fp32 = HERE / "_cap_narrow_fp32.onnx"
    int8 = HERE / "_cap_narrow_int8.onnx"
    v2.export_onnx(model, fp32)
    preds_onnx = v2.onnx_predict(str(fp32), Xe.numpy())
    parity = float((preds_onnx == preds_pt).mean())
    print(f"onnx parity {parity * 100:.2f}%")

    calib_idx = []
    for c in range(6):
        calib_idx += v2.rng.choice(np.where(y[tr] == c)[0],
                                   v2.CALIB_PER_CLASS, replace=False).tolist()
    calib_X = X[tr[calib_idx]].astype(np.float32)
    v2.quantize_int8(str(fp32), str(int8), calib_X,
                     per_channel=False, calib_method="percentile")
    preds_i8 = v2.onnx_predict(str(int8), Xe.numpy())
    acc_i8 = float((preds_i8 == ye.numpy()).mean())
    print(f"narrow int8 test accuracy   {acc_i8:.4f}")

    x1 = Xe[:1].numpy().astype(np.float32)
    lat_f = v2.benchmark_onnx(str(fp32), x1)
    lat_i = v2.benchmark_onnx(str(int8), x1)
    print(f"narrow latency fp32 med {lat_f['median']:.3f} p99 {lat_f['p99']:.3f} ms")
    print(f"narrow latency int8 med {lat_i['median']:.3f} p99 {lat_i['p99']:.3f} ms")

    size_f = fp32.stat().st_size / 1024
    size_i = int8.stat().st_size / 1024
    print(f"narrow size fp32 {size_f:.1f} / int8 {size_i:.1f} kB")

    out = {
        "machine": {"os": sys.platform, "cpu": __import__("platform").processor() or "unknown"},
        "config": {"seed": SEED, "epochs": EPOCHS,
                   "quant": "per-tensor QInt8 + QOperator + ORT_ENABLE_ALL"},
        "narrow": {
            "params": n_par,
            "fp32_acc": acc_pt, "int8_acc": acc_i8,
            "fp32_median_ms": lat_f["median"], "fp32_p99_ms": lat_f["p99"],
            "int8_median_ms": lat_i["median"], "int8_p99_ms": lat_i["p99"],
            "fp32_kB": size_f, "int8_kB": size_i,
        },
        "wide": _read_wide(),
    }
    (HERE / "capacity_measure.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote capacity_measure.json")

    fp32.unlink(missing_ok=True)
    int8.unlink(missing_ok=True)


def _read_wide():
    """Read the wide-model numbers from the latest v2_output (this machine's
    pipeline_v2.py run) instead of hard-coding them."""
    import csv
    rows = list(csv.reader(open(HERE / "v2_output" / "results_table.csv", encoding="utf-8")))
    hdr = [h.strip() for h in rows[0]]
    get = {r[0]: r for r in rows[1:]}
    fp = get["float32"]; i8 = get["int8"]
    meta = json.loads((HERE / "v2_output" / "results_meta.json").read_text(encoding="utf-8"))
    return {
        "params": meta["params"],
        "fp32_acc": float(fp[hdr.index("test_accuracy")]),
        "int8_acc": float(i8[hdr.index("test_accuracy")]),
        "fp32_median_ms": float(fp[hdr.index("latency_median_ms")]),
        "fp32_p99_ms": float(fp[hdr.index("latency_p99_ms")]),
        "int8_median_ms": float(i8[hdr.index("latency_median_ms")]),
        "int8_p99_ms": float(i8[hdr.index("latency_p99_ms")]),
        "fp32_kB": float(fp[hdr.index("model_size_kB")]),
        "int8_kB": float(i8[hdr.index("model_size_kB")]),
    }


def _narrow():
    n_features = 32 * (IMG // 4) * (IMG // 4)
    layers = [
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
    ]
    layers.append(nn.Linear(n_features, len(CLASSES)))
    return nn.Sequential(*layers)


if __name__ == "__main__":
    main()