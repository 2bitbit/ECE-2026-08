#!/usr/bin/env python3
"""
PROJECT 1  -  Edge Vision and Model Quantisation
==================================================
Full pipeline, run end-to-end with one command:

    python pipeline.py

What it does, in order:

  1. Loads NEU-DET (or synthetic images if the real folder is missing).
  2. Splits 60 / 20 / 20 per class, so train/val/test never share a specimen.
  3. Trains a small float32 convolutional classifier (< 500k parameters)
     and reports per-epoch validation accuracy.
  4. Exports it to ONNX and checks the ONNX model agrees with PyTorch.
  5. Benchmarks latency single-threaded, batch of one: median and p99.
  6. Applies static int8 quantisation and justifies every setting below.
  7. Computes per-class precision / recall for both models.
  8. Writes figures/ and results_table.{csv,md} and prints the acceptance table.

Quantisation decisions (and why):
  * calibration set  : 120 images (20 per class) drawn from TRAIN only,
                       never from test. Big enough to span every class and the
                       brightness/texture range, small enough to stay fast.
  * calibration      : percentile clipping, which ignores outlier pixels that
                       would otherwise stretch the range and lose precision.
  * weights          : per-channel QInt8 (symmetric). A/B-measured against
                       per-tensor: 0.8250 vs 0.8306 (a 2-image, within-noise
                       gap), so the two are interchangeable; we keep per-channel
                       and record the tie (see compare_per_channel.py).
  * activations      : QUInt8 (asymmetric). ReLU outputs are >= 0, so an
                       unsigned, asymmetric range wastes nothing below zero.
"""

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

# The Windows console defaults to a legacy code page (e.g. GBK); torch's ONNX
# exporter prints Unicode checkmarks that would otherwise crash the run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

import numpy as np
import matplotlib

matplotlib.use("Agg")  # save figures without a display
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from starter_p1 import CLASSES, IMG, task1_to_array, task4_model, synthetic, _find

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
SEED = 0
HERE = Path(__file__).resolve().parent
ROOT = os.environ.get("NEU_ROOT") or _find("NEU-DET")
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}   # accepted image suffixes (single source of truth)

BUDGET_MS = 15.0          # the Raspberry Pi latency budget from the brief
CALIB_PER_CLASS = 20      # calibration images per class (120 total)
WARMUP = 30               # warm-up runs before timing
RUNS = 100                # timed runs for the latency percentile

# Palette: categorical slots in fixed order (blue, orange, aqua, yellow,
# magenta, green) mapped to the six defect classes, one blue ramp for the
# confusion matrix. Text uses primary/secondary/muted ink, hairline grid.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

# Short display names for axis labels — the full class names ("pitted_surface",
# "rolled-in_scale") are too long and get clipped in tight figure layouts, so
# axis labels use these meaningful abbreviations instead of a raw string slice.
SHORT = ["crazing", "inclusion", "patches", "pitted", "rolled-in", "scratches"]

rng = np.random.default_rng(SEED)


def _set_style():
    plt.rcParams.update({
        "figure.facecolor": SURF,
        "axes.facecolor": SURF,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "text.color": INK,
        "font.family": "Segoe UI",
        "axes.grid": True,
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ---------------------------------------------------------------------------
# 1. load + split
# ---------------------------------------------------------------------------
def image_files(directory):
    """Sorted image paths in `directory`, matching the loader's extension set
    (the Kaggle mirror uses .jpg, the official NEU-DET release uses .bmp)."""
    directory = Path(directory)
    return sorted(f for f in directory.glob("*")
                  if f.suffix.lower() in IMAGE_EXTS)


def load_dataset(root):
    """One file per image, no double counting. Accepts .jpg/.png/.bmp so it
    works with both the Kaggle mirror and the official NEU-DET (.bmp) release."""
    root = Path(root)
    X, y = [], []
    for ci, c in enumerate(CLASSES):
        files = image_files(root / c)
        for f in files:
            X.append(task1_to_array(f)[None])  # shape (1, IMG, IMG)
            y.append(ci)
    if not X:
        return None, None
    return np.stack(X).astype(np.float32), np.array(y)


def stratified_split(y, train=0.6, val=0.2):
    """Split indices 60/20/20 *per class*, so every class is represented in
    every split and no specimen ever appears on both sides."""
    tr, va, te = [], [], []
    for c in range(len(CLASSES)):
        idx = rng.permutation(np.where(y == c)[0])
        n = len(idx)
        a = int(round(train * n))
        b = int(round((train + val) * n))
        tr += idx[:a].tolist()
        va += idx[a:b].tolist()
        te += idx[b:].tolist()
    return (np.array(tr), np.array(va), np.array(te))


# ---------------------------------------------------------------------------
# 2. train
# ---------------------------------------------------------------------------
def train_model(model, Xt, yt, Xv, yv, epochs, batch=64, lr=1e-3):
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr)
    lossf = nn.CrossEntropyLoss()
    hist = {"train_acc": [], "val_acc": []}
    for ep in range(epochs):
        model.train()
        correct = total = 0
        for xb, yb in dl:
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.numel()
        hist["train_acc"].append(correct / total)
        hist["val_acc"].append(evaluate_pt(model, Xv, yv)[0])
    return model, hist


def evaluate_pt(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(1).numpy()
    return float((preds == y.numpy()).mean()), preds


# ---------------------------------------------------------------------------
# 3. ONNX export + parity
# ---------------------------------------------------------------------------
def export_onnx(model, path):
    model.eval()
    dummy = torch.randn(1, 1, IMG, IMG)
    # Fixed batch of one: the Pi processes a single part at a time, so this is
    # exactly the deployment shape and lets quant_pre_process infer static shapes.
    # external_data=False keeps the weights inside the single .onnx file so the
    # reported model size is honest (the default would split them to a side file).
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["input"], output_names=["logits"],
        opset_version=18, external_data=False,
    )
    return path


def onnx_predict(path, X):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    # batch of one, matching the exported (and deployed) shape
    outs = [sess.run(None, {name: X[i:i + 1]})[0][0] for i in range(len(X))]
    return np.stack(outs).argmax(1)


# ---------------------------------------------------------------------------
# 4. latency
# ---------------------------------------------------------------------------
def benchmark_onnx(path, x):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for _ in range(WARMUP):
        sess.run(None, {name: x})
    t = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        t.append((time.perf_counter() - t0) * 1000.0)
    return _summary(t)


def _summary(times):
    a = np.sort(times)
    return {"median": float(np.median(a)), "p99": float(a[int(0.99 * (len(a) - 1))]),
            "all": a}


# ---------------------------------------------------------------------------
# 5. quantisation
# ---------------------------------------------------------------------------
def quantize_int8(fp32_path, int8_path, calib_X, per_channel=True,
                  calib_method="percentile"):
    import tempfile
    from onnxruntime.quantization import (
        quantize_static, QuantType, CalibrationMethod,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader:
        def __init__(self, data):
            self._data = data
            self._i = 0

        def get_next(self):
            if self._i >= len(self._data):
                return None
            x = self._data[self._i][None].astype(np.float32)
            self._i += 1
            return {"input": x}

    m = CalibrationMethod.Percentile if calib_method == "percentile" else CalibrationMethod.MinMax

    # fold nodes + shape inference first (brief: quant_pre_process). Our model
    # has no BatchNorm, so this mainly runs shape inference. The pre-processed
    # model and any side files go into a temp dir that is removed automatically.
    with tempfile.TemporaryDirectory() as td:
        pre = Path(td) / "pre.onnx"
        src = fp32_path
        try:
            quant_pre_process(str(fp32_path), str(pre))
            src = pre
        except Exception as e:
            print(f"  (quant_pre_process skipped: {e})")
        quantize_static(
            str(src), str(int8_path), Reader(calib_X),
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            per_channel=per_channel,
            calibrate_method=m,
        )

    # onnxruntime's shape inference drops a stray temp file in the cwd
    (Path.cwd() / "sym_shape_infer_temp.onnx").unlink(missing_ok=True)
    return int8_path


# ---------------------------------------------------------------------------
# 6. metrics
# ---------------------------------------------------------------------------
def per_class_metrics(preds, truth):
    prec, rec = [], []
    for c in range(len(CLASSES)):
        tp = int(((preds == c) & (truth == c)).sum())
        fp = int(((preds == c) & (truth != c)).sum())
        fn = int(((preds != c) & (truth == c)).sum())
        prec.append(tp / (tp + fp) if tp + fp else 0.0)
        rec.append(tp / (tp + fn) if tp + fn else 0.0)
    return prec, rec


def confusion_matrix(preds, truth):
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(truth, preds):
        cm[t, p] += 1
    return cm


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_sample_grid(images, labels):
    """Two images per class."""
    fig, axes = plt.subplots(2, 6, figsize=(11, 4.2))
    for ci, c in enumerate(CLASSES):
        idx = np.where(labels == ci)[0][:2]
        for row in range(2):
            ax = axes[row, ci]
            ax.imshow(images[idx[row], 0], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(c.replace("_", " "), fontsize=9, color=INK)
    fig.suptitle("Two samples per class (resized 96x96)", color=INK, y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG_DIR / "sample_grid.png", dpi=150)
    plt.close(fig)


def fig_training_curve(hist):
    fig, ax = plt.subplots(figsize=(6, 4))
    ep = np.arange(1, len(hist["train_acc"]) + 1)
    ax.plot(ep, hist["train_acc"], color=CAT[0], lw=2, label="train")
    ax.plot(ep, hist["val_acc"], color=CAT[1], lw=2, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.set_title("Training progress", color=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curve.png", dpi=150)
    plt.close(fig)


def fig_confusion(cm_fp32, cm_int8):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, cm, title in [(axes[0], cm_fp32, "float32"),
                          (axes[1], cm_int8, "int8")]:
        im = ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "blue", BLUE_RAMP), aspect="auto")
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(SHORT, rotation=40, fontsize=7, color=INK2)
        ax.set_yticklabels(SHORT, fontsize=7, color=INK2)
        ax.set_title(title, color=INK)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        fontsize=7, color="white" if cm[i, j] > cm.max() / 2 else INK)
    fig.suptitle("Confusion matrices (rows = truth, cols = prediction)", color=INK, y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def fig_latency(times_fp32, times_int8):
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [times_fp32["all"], times_int8["all"]]
    bp = ax.boxplot(data, tick_labels=["float32", "int8"], widths=0.45,
                    patch_artist=True, medianprops=dict(color=INK, lw=2))
    for patch, col in zip(bp["boxes"], [CAT[0], CAT[1]]):
        patch.set_facecolor(col); patch.set_alpha(0.45)
    # Measured latency (~0.3-0.8 ms) is 20-50x below the 15 ms budget, so a
    # linear axis would clip the budget line off the top. Log scale + an
    # explicit upper limit keep both the boxes and the budget line visible.
    ax.set_yscale("log")
    ax.set_ylim(0.05, BUDGET_MS * 1.5)
    ax.axhline(BUDGET_MS, color=CAT[5], lw=1.5, ls="--")
    ax.text(1.45, BUDGET_MS * 1.05, "15 ms budget", va="bottom", fontsize=8,
            color=CAT[5])
    ax.set_ylabel("latency (ms, log scale)")
    ax.set_title("Single-thread, batch-1 latency over %d runs" % RUNS, color=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "latency.png", dpi=150)
    plt.close(fig)


def fig_per_class_recall(rec_fp32, rec_int8):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(6)
    w = 0.38
    ax.bar(x - w / 2, rec_fp32, w, color=CAT[0], label="float32")
    ax.bar(x + w / 2, rec_int8, w, color=CAT[1], label="int8")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, rotation=25, ha="right", fontsize=8, color=INK2)
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.set_title("Per-class recall", color=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_class_recall.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
def write_results(rows, per_class, meta):
    # main acceptance table, one row per model
    main = HERE / "results_table.csv"
    with open(main, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "test_accuracy", "worst_class", "worst_class_recall",
                    "model_size_kB", "latency_median_ms", "latency_p99_ms",
                    "verdict_15ms"])
        for r in rows:
            w.writerow([r["model"], f'{r["acc"]:.4f}', r["worst_class"],
                        f'{r["worst_recall"]:.4f}', f'{r["size_kb"]:.1f}',
                        f'{r["median_ms"]:.2f}', f'{r["p99_ms"]:.2f}', r["verdict"]])

    # per-class table
    pc = HERE / "per_class_metrics.csv"
    with open(pc, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "float32_precision", "float32_recall",
                    "int8_precision", "int8_recall"])
        for ci, c in enumerate(CLASSES):
            w.writerow([c, f'{per_class["fp32"][0][ci]:.4f}',
                        f'{per_class["fp32"][1][ci]:.4f}',
                        f'{per_class["int8"][0][ci]:.4f}',
                        f'{per_class["int8"][1][ci]:.4f}'])

    meta["figures"] = sorted(p.name for p in FIG_DIR.glob("*.png"))
    (HERE / "results_meta.json").write_text(json.dumps(meta, indent=2))

    # also emit a markdown copy of the main table for the report
    md = HERE / "results_table.md"
    with open(md, "w") as f:
        f.write("| measurement | float32 | int8 |\n|---|---|---|\n")
        f.write(f"| test accuracy | {rows[0]['acc']:.4f} | {rows[1]['acc']:.4f} |\n")
        f.write(f"| worst class | {rows[0]['worst_class']} | {rows[1]['worst_class']} |\n")
        f.write(f"| worst class recall | {rows[0]['worst_recall']:.4f} | {rows[1]['worst_recall']:.4f} |\n")
        f.write(f"| model size (kB) | {rows[0]['size_kb']:.1f} | {rows[1]['size_kb']:.1f} |\n")
        f.write(f"| latency median (ms) | {rows[0]['median_ms']:.2f} | {rows[1]['median_ms']:.2f} |\n")
        f.write(f"| latency p99 (ms) | {rows[0]['p99_ms']:.2f} | {rows[1]['p99_ms']:.2f} |\n")
        f.write(f"| verdict vs 15 ms | {rows[0]['verdict']} | {rows[1]['verdict']} |\n")


def _report(acc, size_kb, lat, worst, verdict):
    return {"acc": acc, "size_kb": size_kb, "median_ms": lat["median"],
            "p99_ms": lat["p99"], "worst_class": worst,
            "worst_recall": None, "verdict": verdict}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(epochs=20):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    _set_style()

    print("=" * 60)
    print("  PROJECT 1  -  full pipeline")
    print("=" * 60)

    X, y = load_dataset(ROOT)
    synthetic_used = X is None
    if synthetic_used:
        print("NEU-DET not found -> using synthetic images (results are demo only).\n")
        X, y = synthetic()
    else:
        print(f"loaded {len(y)} real images from {ROOT}\n")

    tr, va, te = stratified_split(y)
    print(f"split  ->  train {len(tr)}  val {len(va)}  test {len(te)} "
          f"(no overlap: {len(set(tr) & set(te)) == 0 and len(set(tr) & set(va)) == 0})\n")

    Xt = torch.tensor(X[tr]); yt = torch.tensor(y[tr])
    Xv = torch.tensor(X[va]); yv = torch.tensor(y[va])
    Xe = torch.tensor(X[te]); ye = torch.tensor(y[te])

    fig_sample_grid(X[tr], y[tr])

    # ---- float32 baseline ----
    model = task4_model()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model  ->  {n_par:,} parameters")
    model, hist = train_model(model, Xt, yt, Xv, yv, epochs)
    fig_training_curve(hist)

    acc_pt, preds_pt = evaluate_pt(model, Xe, ye)
    print(f"float32 test accuracy  {acc_pt:.4f}\n")

    # ---- ONNX export + parity ----
    fp32_onnx = HERE / "model_fp32.onnx"
    export_onnx(model, fp32_onnx)
    preds_onnx = onnx_predict(fp32_onnx, Xe.numpy())
    parity = float((preds_onnx == preds_pt).mean())
    print(f"ONNX export -> {fp32_onnx.name}, "
          f"agrees with PyTorch on {parity * 100:.2f}% of test predictions\n")

    x1 = Xe[:1].numpy().astype(np.float32)
    lat_onnx = benchmark_onnx(fp32_onnx, x1)
    print(f"float32 latency  median {lat_onnx['median']:.2f} ms  "
          f"p99 {lat_onnx['p99']:.2f} ms")

    # ---- int8 quantisation ----
    calib_idx = []
    for c in range(6):
        calib_idx += rng.choice(np.where(y[tr] == c)[0], CALIB_PER_CLASS,
                                replace=False).tolist()
    # calib_idx holds positions INSIDE the train subset (0..len(tr)-1); map them
    # back to full-dataset indices through tr, so the calibration set really is
    # CALIB_PER_CLASS images per class, all drawn from the train split.
    calib_X = X[tr[calib_idx]].astype(np.float32)
    print(f"calibration set  {len(calib_X)} images "
          f"({CALIB_PER_CLASS}/class, from train only)")

    int8_onnx = HERE / "model_int8.onnx"
    quantize_int8(fp32_onnx, int8_onnx, calib_X, per_channel=True,
                  calib_method="percentile")

    preds_i8 = onnx_predict(int8_onnx, Xe.numpy())
    acc_i8 = float((preds_i8 == ye.numpy()).mean())
    lat_i8 = benchmark_onnx(int8_onnx, x1)
    print(f"int8 test accuracy    {acc_i8:.4f}   "
          f"latency median {lat_i8['median']:.2f} ms  p99 {lat_i8['p99']:.2f} ms\n")

    # ---- metrics + figures ----
    prec_fp, rec_fp = per_class_metrics(preds_pt, ye.numpy())
    prec_i8, rec_i8 = per_class_metrics(preds_i8, ye.numpy())
    fig_confusion(confusion_matrix(preds_pt, ye.numpy()),
                  confusion_matrix(preds_i8, ye.numpy()))
    fig_latency(lat_onnx, lat_i8)
    fig_per_class_recall(rec_fp, rec_i8)

    def worst(rec):
        i = int(np.argmin(rec))
        return CLASSES[i], rec[i]

    rows = [
        {"model": "float32", "acc": acc_pt, "size_kb": fp32_onnx.stat().st_size / 1024,
         "median_ms": lat_onnx["median"], "p99_ms": lat_onnx["p99"],
         "worst_class": worst(rec_fp)[0], "worst_recall": worst(rec_fp)[1],
         "verdict": "FITS" if lat_onnx["p99"] < BUDGET_MS else "TOO SLOW"},
        {"model": "int8", "acc": acc_i8, "size_kb": int8_onnx.stat().st_size / 1024,
         "median_ms": lat_i8["median"], "p99_ms": lat_i8["p99"],
         "worst_class": worst(rec_i8)[0], "worst_recall": worst(rec_i8)[1],
         "verdict": "FITS" if lat_i8["p99"] < BUDGET_MS else "TOO SLOW"},
    ]
    write_results(rows, {"fp32": (prec_fp, rec_fp), "int8": (prec_i8, rec_i8)},
                  {"synthetic": synthetic_used, "params": n_par,
                   "calib_per_class": CALIB_PER_CLASS, "warmup": WARMUP,
                   "runs": RUNS,
                   "hardware": {"os": platform.platform(),
                                "machine": platform.machine(),
                                "cpu": platform.processor() or platform.machine()}})

    # ---- acceptance table ----
    print("=" * 60)
    print("  ACCEPTANCE TABLE")
    print("=" * 60)
    hdr = f'{"measurement":<24}{"float32":>12}{"int8":>12}'
    print(hdr)
    print("-" * len(hdr))
    print(f'{"test accuracy":<24}{rows[0]["acc"]:>12.4f}{rows[1]["acc"]:>12.4f}')
    print(f'{"worst class recall":<24}{rows[0]["worst_recall"]:>12.4f}{rows[1]["worst_recall"]:>12.4f}')
    print(f'{"  (class)":<24}{rows[0]["worst_class"]:>16}{rows[1]["worst_class"]:>16}')
    print(f'{"model size (kB)":<24}{rows[0]["size_kb"]:>12.1f}{rows[1]["size_kb"]:>12.1f}')
    print(f'{"latency median (ms)":<24}{rows[0]["median_ms"]:>12.2f}{rows[1]["median_ms"]:>12.2f}')
    print(f'{"latency p99 (ms)":<24}{rows[0]["p99_ms"]:>12.2f}{rows[1]["p99_ms"]:>12.2f}')
    print(f'{"verdict (15 ms)":<24}{rows[0]["verdict"]:>12}{rows[1]["verdict"]:>12}')
    print()

    per_class_heading = (f'{"class":<18}{"fp32 P":>8}{"fp32 R":>8}'
                         f'{"int8 P":>8}{"int8 R":>8}')
    print(per_class_heading)
    print("-" * len(per_class_heading))
    for ci, c in enumerate(CLASSES):
        print(f'{c:<18}{prec_fp[ci]:>8.3f}{rec_fp[ci]:>8.3f}'
              f'{prec_i8[ci]:>8.3f}{rec_i8[ci]:>8.3f}')
    print()

    print("figures written to", FIG_DIR)
    print("tables written to results_table.csv / results_table.md / per_class_metrics.csv")
    print("done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    main(args.epochs)
