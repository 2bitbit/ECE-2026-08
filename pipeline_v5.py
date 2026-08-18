#!/usr/bin/env python3
"""
PROJECT 1  -  Edge Vision and Model Quantisation (V5: Ultimate Form)
====================================================================
Full pipeline, run end-to-end with one command:

    python pipeline_v5.py

V5 ULTIMATE UPGRADES (Combining V3 brute force with V4 academic elegance):
  1. Resolution Scaling: 128x128 inputs to preserve fine scratches.
  2. Width Expansion: Channels expanded to [32, 64, 128, 128]. Total 
     params ~240k, perfectly under the 500k budget.
  3. Global Average Pooling (GAP) & Label Smoothing: Eliminates the 
     massive Flatten layer, eradicating over-fitting and making the 
     activations extremely stable for int8 quantization.
  4. Cutout & Augmentation: Dynamic erasing prevents memorization.
  5. 4x Test-Time Augmentation (TTA): Averages logits from 4 rotated/flipped
     versions of the input during inference to squeeze every drop of accuracy.
     
By heavily optimizing the ONNX graph and using QOperator per-tensor 
quantization, the 4x TTA execution will aggressively push against the 15ms 
budget, successfully exchanging idle processor time for peak accuracy.
"""

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

# Safe console encoding for Windows
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from starter_p1 import CLASSES, _find

SEED = 0
HERE = Path(__file__).resolve().parent
ROOT = os.environ.get("NEU_ROOT") or _find("NEU-DET")

# --- V5 DIRECTORY SETUP ---
OUT_DIR = HERE / "v5_output"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# --- V5 ULTIMATE PARAMETERS ---
V5_IMG = 128              # High-Res input
BUDGET_MS = 15.0          # The hard deadline
CALIB_PER_CLASS = 20      
WARMUP = 30               
RUNS = 100                

# Visual styling
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SHORT = ["crazing", "inclusion", "patches", "pitted", "rolled-in", "scratches"]

rng = np.random.default_rng(SEED)

def _set_style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "grid.color": GRID,
        "text.color": INK, "font.family": "Segoe UI",
        "axes.grid": True, "grid.linewidth": 0.7,
        "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False,
    })


def v5_to_array(path):
    """Load and scale images to 128x128."""
    from PIL import Image
    im = Image.open(path).convert("L").resize((V5_IMG, V5_IMG))
    arr = np.asarray(im, dtype=np.float32)
    return arr / 255.0

def v5_synthetic(n_per_class=120, seed=0):
    """Generate 128x128 synthetic fallback images."""
    rng_syn = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(V5_IMG), np.arange(V5_IMG), indexing="ij")
    X, y = [], []
    for c in range(6):
        for _ in range(n_per_class):
            b = rng_syn.normal(0.5, 0.14, (V5_IMG, V5_IMG))
            if c == 0:   b += 0.11 * np.sin(xx * 1.7 + rng_syn.uniform(0, 6))
            elif c == 1:
                cy, cx = rng_syn.integers(10, V5_IMG - 10, 2)
                b += 0.24 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 40)
            elif c == 2: b += 0.18 * (np.abs(yy - V5_IMG / 2) < rng_syn.integers(8, 20))
            elif c == 3: b -= 0.30 * (rng_syn.random((V5_IMG, V5_IMG)) < 0.02)
            elif c == 4: b += 0.14 * np.sin(yy * 0.25 + rng_syn.uniform(0, 6))
            else:        b += 0.26 * (np.abs(yy - xx - rng_syn.integers(-20, 20)) < 2)
            X.append(np.clip(b, 0.0, 1.0)[None].astype(np.float32)); y.append(c)
    return np.stack(X), np.array(y)

def load_dataset_v5(root):
    root = Path(root)
    X, y = [], []
    for ci, c in enumerate(CLASSES):
        directory = root / c
        files = sorted(f for f in directory.glob("*") if f.suffix.lower() in IMAGE_EXTS)
        for f in files:
            X.append(v5_to_array(f)[None])
            y.append(ci)
    if not X:
        return None, None
    return np.stack(X).astype(np.float32), np.array(y)

def stratified_split(y, train=0.6, val=0.2):
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


def augment_batch(xb):
    """
    Academic Augmentation: Flips + Cutout (Dynamic Random Erasing).
    Forces model to look globally rather than memorizing local spots.
    """
    B, C, H, W = xb.shape
    xb_aug = xb.clone()
    for i in range(B):
        # 50% horizontal flip
        if torch.rand(1).item() > 0.5: xb_aug[i] = torch.flip(xb_aug[i], [2])
        # 50% vertical flip
        if torch.rand(1).item() > 0.5: xb_aug[i] = torch.flip(xb_aug[i], [1])
            
        # Cutout: 50% chance to drop a random block (scaled for 128x128)
        if torch.rand(1).item() > 0.5:
            h_c = torch.randint(15, 40, (1,)).item()
            w_c = torch.randint(15, 40, (1,)).item()
            y_pos = torch.randint(0, H - h_c, (1,)).item()
            x_pos = torch.randint(0, W - w_c, (1,)).item()
            xb_aug[i, 0, y_pos:y_pos+h_c, x_pos:x_pos+w_c] = 0.5
    return xb_aug


def v5_model():
    """
    V5 Ultimate Architecture.
    - Width increased to 128 channels (Capacity upgrade).
    - Global Average Pooling eradicates over-fitting and keeps params ~240k.
    """
    layers = [
        # 128x128 -> 64x64
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        # 64x64 -> 32x32
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        # 32x32 -> 16x16
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        # 16x16 -> 8x8
        nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        
        # GAP: Condenses spatial dimensions into 1x1 per channel
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        # Tiny linear layer (128 inputs instead of tens of thousands)
        nn.Linear(128, len(CLASSES))
    ]
    return nn.Sequential(*layers)


def train_model(model, Xt, yt, Xv, yv, epochs, batch=64, lr=1e-3):
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr)
    
    # Label Smoothing = 0.1 protects against over-confidence & aids int8 stability
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    hist = {"train_acc": [], "val_acc": []}
    for ep in range(epochs):
        model.train()
        correct = total = 0
        for xb, yb in dl:
            xb_aug = augment_batch(xb) # Apply Cutout
            opt.zero_grad()
            out = model(xb_aug)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.numel()
            
        hist["train_acc"].append(correct / total)
        # Validate using TTA
        hist["val_acc"].append(evaluate_pt_tta(model, Xv, yv)[0])
    return model, hist

def evaluate_pt_tta(model, X, y):
    """Evaluates using 4x Test-Time Augmentation (TTA)."""
    model.eval()
    with torch.no_grad():
        out1 = model(X)
        out2 = model(torch.flip(X, [3]))                      # Horizontal flip
        out3 = model(torch.flip(X, [2]))                      # Vertical flip
        out4 = model(torch.rot90(X, k=1, dims=[2, 3]))        # Rotate 90
        preds = (out1 + out2 + out3 + out4).argmax(1).numpy()
    return float((preds == y.numpy()).mean()), preds


def export_onnx(model, path):
    model.eval()
    dummy = torch.randn(1, 1, V5_IMG, V5_IMG)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["input"], output_names=["logits"],
        opset_version=18, external_data=False,
    )
    return path

def onnx_predict_tta(path, X):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    
    X_fh = np.flip(X, axis=3).copy()
    X_fv = np.flip(X, axis=2).copy()
    X_r90 = np.rot90(X, k=1, axes=(2, 3)).copy()
    
    outs = []
    for i in range(len(X)):
        l1 = sess.run(None, {name: X[i:i + 1]})[0][0]
        l2 = sess.run(None, {name: X_fh[i:i + 1]})[0][0]
        l3 = sess.run(None, {name: X_fv[i:i + 1]})[0][0]
        l4 = sess.run(None, {name: X_r90[i:i + 1]})[0][0]
        outs.append((l1 + l2 + l3 + l4).argmax())
    return np.array(outs)


def benchmark_onnx_tta(path, x):
    """Measures the latency of evaluating ONE image via 4 passes (TTA)."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    # OPTIMIZATION: Maximize graph optimizations to counter TTA overhead
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    
    x_fh = np.flip(x, axis=3).copy()
    x_fv = np.flip(x, axis=2).copy()
    x_r90 = np.rot90(x, k=1, axes=(2, 3)).copy()

    for _ in range(WARMUP):
        sess.run(None, {name: x})
        
    t = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        # A single deployment inference consists of 4 forward passes
        sess.run(None, {name: x})
        sess.run(None, {name: x_fh})
        sess.run(None, {name: x_fv})
        sess.run(None, {name: x_r90})
        t.append((time.perf_counter() - t0) * 1000.0)
        
    a = np.sort(t)
    return {"median": float(np.median(a)), "p99": float(a[int(0.99 * (len(a) - 1))]), "all": a}


def quantize_int8(fp32_path, int8_path, calib_X, calib_method="percentile"):
    import tempfile
    from onnxruntime.quantization import (
        quantize_static, QuantType, CalibrationMethod, QuantFormat
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader:
        def __init__(self, data):
            self._data = data; self._i = 0
        def get_next(self):
            if self._i >= len(self._data): return None
            x = self._data[self._i][None].astype(np.float32)
            self._i += 1
            return {"input": x}

    m = CalibrationMethod.Percentile if calib_method == "percentile" else CalibrationMethod.MinMax

    with tempfile.TemporaryDirectory() as td:
        pre = Path(td) / "pre.onnx"
        src = fp32_path
        try:
            quant_pre_process(str(fp32_path), str(pre))
            src = pre
        except Exception as e:
            pass
            
        # V5 OPTIMIZATIONS:
        # per_channel=False (per-tensor speeds up 4x TTA without losing accuracy)
        # quant_format=QOperator (fuses nodes, vital for keeping TTA under 15ms)
        quantize_static(
            str(src), str(int8_path), Reader(calib_X),
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            per_channel=False, 
            calibrate_method=m,
            quant_format=QuantFormat.QOperator 
        )

    (Path.cwd() / "sym_shape_infer_temp.onnx").unlink(missing_ok=True)
    return int8_path


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

def fig_training_curve(hist):
    fig, ax = plt.subplots(figsize=(6, 4))
    ep = np.arange(1, len(hist["train_acc"]) + 1)
    ax.plot(ep, hist["train_acc"], color=CAT[0], lw=2, label="train (w/ Cutout)")
    ax.plot(ep, hist["val_acc"], color=CAT[1], lw=2, label="validation (4x TTA)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.set_title("V5 Training progress (GAP + TTA)", color=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curve.png", dpi=150)
    plt.close(fig)

def fig_confusion(cm_fp32, cm_int8):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, cm, title in [(axes[0], cm_fp32, "float32 (V5)"),
                          (axes[1], cm_int8, "int8 (V5)")]:
        ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "blue", BLUE_RAMP), aspect="auto")
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(SHORT, rotation=40, fontsize=7, color=INK2)
        ax.set_yticklabels(SHORT, fontsize=7, color=INK2)
        ax.set_title(title, color=INK)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        fontsize=7, color="white" if cm[i, j] > cm.max() / 2 else INK)
    fig.suptitle("V5 Confusion matrices (128x128 + 4x TTA)", color=INK, y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def write_results(rows, per_class, meta):
    main = OUT_DIR / "results_table.csv"
    with open(main, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "test_accuracy", "worst_class", "worst_class_recall",
                    "model_size_kB", "latency_median_ms", "latency_p99_ms",
                    "verdict_15ms"])
        for r in rows:
            w.writerow([r["model"], f'{r["acc"]:.4f}', r["worst_class"],
                        f'{r["worst_recall"]:.4f}', f'{r["size_kb"]:.1f}',
                        f'{r["median_ms"]:.2f}', f'{r["p99_ms"]:.2f}', r["verdict"]])

    pc = OUT_DIR / "per_class_metrics.csv"
    with open(pc, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "float32_precision", "float32_recall", "int8_precision", "int8_recall"])
        for ci, c in enumerate(CLASSES):
            w.writerow([c, f'{per_class["fp32"][0][ci]:.4f}', f'{per_class["fp32"][1][ci]:.4f}',
                        f'{per_class["int8"][0][ci]:.4f}', f'{per_class["int8"][1][ci]:.4f}'])

def main(epochs=40): # 40 epochs allows Cutout to fully train robust features
    global rng
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    _set_style()

    print("=" * 60)
    print("  PROJECT 1  -  V5 Ultimate Pipeline (GAP + 128x128 + TTA)")
    print("=" * 60)
    print(f"  -> Redirecting all outputs to {OUT_DIR.name}/")

    X, y = load_dataset_v5(ROOT)
    if X is None:
        print("NEU-DET not found -> generating 128x128 synthetic images.\n")
        X, y = v5_synthetic()
    else:
        print(f"loaded {len(y)} real high-res (128x128) images\n")

    tr, va, te = stratified_split(y)
    Xt = torch.tensor(X[tr]); yt = torch.tensor(y[tr])
    Xv = torch.tensor(X[va]); yv = torch.tensor(y[va])
    Xe = torch.tensor(X[te]); ye = torch.tensor(y[te])

    # ---- 1. V5 Float32 Baseline ----
    model = v5_model()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model  ->  {n_par:,} parameters (Wide structure with GAP)")
    
    model, hist = train_model(model, Xt, yt, Xv, yv, epochs)
    fig_training_curve(hist)

    acc_pt, preds_pt = evaluate_pt_tta(model, Xe, ye)
    print(f"float32 V5 test accuracy (+ 4x TTA): {acc_pt:.4f}\n")

    # ---- 2. ONNX Export ----
    fp32_onnx = OUT_DIR / "model_fp32.onnx"
    export_onnx(model, fp32_onnx)
    
    preds_onnx = onnx_predict_tta(fp32_onnx, Xe.numpy())
    x1 = Xe[:1].numpy().astype(np.float32)
    lat_onnx = benchmark_onnx_tta(fp32_onnx, x1)

    # ---- 3. Int8 Quantisation ----
    calib_idx = []
    for c in range(6):
        calib_idx += rng.choice(np.where(y[tr] == c)[0], CALIB_PER_CLASS, replace=False).tolist()
    calib_X = X[tr[calib_idx]].astype(np.float32)

    int8_onnx = OUT_DIR / "model_int8.onnx"
    quantize_int8(fp32_onnx, int8_onnx, calib_X)

    preds_i8 = onnx_predict_tta(int8_onnx, Xe.numpy())
    acc_i8 = float((preds_i8 == ye.numpy()).mean())
    lat_i8 = benchmark_onnx_tta(int8_onnx, x1)
    
    # ---- 4. Report Generation ----
    prec_fp, rec_fp = per_class_metrics(preds_pt, ye.numpy())
    prec_i8, rec_i8 = per_class_metrics(preds_i8, ye.numpy())
    fig_confusion(confusion_matrix(preds_pt, ye.numpy()), confusion_matrix(preds_i8, ye.numpy()))

    def worst(rec):
        i = int(np.argmin(rec))
        return CLASSES[i], rec[i]

    rows = [
        {"model": "float32 (V5)", "acc": acc_pt, "size_kb": fp32_onnx.stat().st_size / 1024,
         "median_ms": lat_onnx["median"], "p99_ms": lat_onnx["p99"],
         "worst_class": worst(rec_fp)[0], "worst_recall": worst(rec_fp)[1],
         "verdict": "FITS" if lat_onnx["p99"] < BUDGET_MS else "TOO SLOW"},
        {"model": "int8 (V5)", "acc": acc_i8, "size_kb": int8_onnx.stat().st_size / 1024,
         "median_ms": lat_i8["median"], "p99_ms": lat_i8["p99"],
         "worst_class": worst(rec_i8)[0], "worst_recall": worst(rec_i8)[1],
         "verdict": "FITS" if lat_i8["p99"] < BUDGET_MS else "TOO SLOW"},
    ]
    write_results(rows, {"fp32": (prec_fp, rec_fp), "int8": (prec_i8, rec_i8)}, {})

    print("=" * 60)
    print("  V5 ACCEPTANCE TABLE (The Ultimate Form)")
    print("=" * 60)
    hdr = f'{"measurement":<24}{"float32":>12}{"int8":>12}'
    print(hdr); print("-" * len(hdr))
    print(f'{"test accuracy":<24}{rows[0]["acc"]:>12.4f}{rows[1]["acc"]:>12.4f}')
    print(f'{"worst class recall":<24}{rows[0]["worst_recall"]:>12.4f}{rows[1]["worst_recall"]:>12.4f}')
    print(f'{"  (class)":<24}{rows[0]["worst_class"]:>16}{rows[1]["worst_class"]:>16}')
    print(f'{"model size (kB)":<24}{rows[0]["size_kb"]:>12.1f}{rows[1]["size_kb"]:>12.1f}')
    print(f'{"latency p99 (ms)":<24}{rows[0]["p99_ms"]:>12.2f}{rows[1]["p99_ms"]:>12.2f}')
    print(f'{"verdict (15 ms)":<24}{rows[0]["verdict"]:>12}{rows[1]["verdict"]:>12}')
    print("\nAll V5 artifacts strictly saved to:", OUT_DIR)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40) 
    main(ap.parse_args().epochs)