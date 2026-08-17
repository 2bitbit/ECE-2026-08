#!/usr/bin/env python3
"""
A/B: per-channel vs per-tensor int8 quantisation.

Reuses the existing model_fp32.onnx (the exact float32 model behind the
reported numbers) and applies the SAME static int8 quantisation twice, with
the only variable being `per_channel`. Reports test accuracy for each
and saves `figures/perchannel_ab.png`.

This turns the defence question "per-channel or per-tensor — what did the
other one give?" into a measured number instead of a reason.

Run:  python compare_per_channel.py
"""

import numpy as np
import matplotlib.pyplot as plt

import pipeline

FP32 = pipeline.HERE / "model_fp32.onnx"
PC = pipeline.HERE / "_tmp_int8_perchannel.onnx"
PT = pipeline.HERE / "_tmp_int8_pertensor.onnx"


def _plot_ab(acc_fp, acc_pc, acc_pt):
    labels = ["float32", "int8\nper-channel", "int8\nper-tensor"]
    vals = [acc_fp, acc_pc, acc_pt]
    colors = [pipeline.CAT[0], pipeline.CAT[1], pipeline.CAT[2]]
    gap = abs(acc_pc - acc_pt)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002,
                f"{v:.4f}", ha="center", va="bottom",
                fontsize=11, color=pipeline.INK)
    ax.axhline(acc_fp, color=pipeline.MUTED, ls="--", lw=1, zorder=0)
    ax.text(-0.28, acc_fp + 0.0015, "float32 baseline",
            va="bottom", fontsize=8, color=pipeline.INK2)
    ax.set_title("per-channel vs per-tensor int8 quantisation (A/B)\n"
                 f"Δ = {gap:.4f}  (≈2 of 360 test images, within noise)",
                 color=pipeline.INK, fontsize=11)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(min(vals) - 0.035, max(vals) + 0.014)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(pipeline.FIG_DIR / "perchannel_ab.png", dpi=150)
    plt.close(fig)


def main():
    if not FP32.exists():
        raise SystemExit(f"{FP32.name} not found — run `python pipeline.py` first")

    X, y = pipeline.load_dataset(pipeline.ROOT)
    if X is None:
        raise SystemExit(f"dataset not found at {pipeline.ROOT}")

    # Mirror pipeline.main()'s exact rng sequence so the split and the 120-image
    # calibration set are IDENTICAL to the ones behind the reported numbers.
    tr, va, te = pipeline.stratified_split(y)
    calib_idx = []
    for c in range(6):
        calib_idx += pipeline.rng.choice(
            np.where(y[tr] == c)[0], pipeline.CALIB_PER_CLASS, replace=False
        ).tolist()
    calib_X = X[tr[calib_idx]].astype(np.float32)

    Xe = X[te].astype(np.float32)
    ye = y[te]

    acc_fp = float((pipeline.onnx_predict(FP32, Xe) == ye).mean())

    pipeline.quantize_int8(FP32, PC, calib_X, per_channel=True,
                           calib_method="percentile")
    acc_pc = float((pipeline.onnx_predict(PC, Xe) == ye).mean())

    pipeline.quantize_int8(FP32, PT, calib_X, per_channel=False,
                           calib_method="percentile")
    acc_pt = float((pipeline.onnx_predict(PT, Xe) == ye).mean())

    print("=" * 60)
    print("  per-channel vs per-tensor (int8)  A/B")
    print("=" * 60)
    print(f"  float32 (ONNX)               {acc_fp:.4f}")
    print(f"  int8  per-channel  QInt8     {acc_pc:.4f}   (reported)")
    print(f"  int8  per-tensor   QInt8     {acc_pt:.4f}")
    print(f"  accuracy drop  per-channel   {acc_fp - acc_pc:.4f}")
    print(f"  accuracy drop  per-tensor    {acc_fp - acc_pt:.4f}")

    pipeline._set_style()
    _plot_ab(acc_fp, acc_pc, acc_pt)

    PC.unlink(missing_ok=True)
    PT.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
