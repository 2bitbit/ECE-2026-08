#!/usr/bin/env python3
"""
Plot test accuracy vs training epochs for the float32 and int8 models.

Re-runs training for a few epoch counts (deterministic, seed=0) and saves
figures/epochs_vs_accuracy.png. The split and the 120-image calibration set
are computed once and shared, so the only variable is the number of epochs.

Run:  python plot_epochs.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import pipeline  # loader / split / model / train / evaluate / quantise

EPOCHS = [10, 15, 20, 30, 40]
FP32_ONNX = pipeline.HERE / "_tmp_fp32.onnx"
INT8_ONNX = pipeline.HERE / "_tmp_int8.onnx"


def main():
    pipeline._set_style()

    X, y = pipeline.load_dataset(pipeline.ROOT)
    if X is None:
        raise SystemExit(f"dataset not found at {pipeline.ROOT}")

    tr, va, te = pipeline.stratified_split(y)
    Xt = torch.tensor(X[tr]); yt = torch.tensor(y[tr])
    Xv = torch.tensor(X[va]); yv = torch.tensor(y[va])
    Xe = torch.tensor(X[te]); ye = torch.tensor(y[te])

    # calibration set, computed once (identical across all epoch counts) —
    # mirrors pipeline.main()'s exact rng sequence after stratified_split.
    calib_idx = []
    for c in range(6):
        calib_idx += pipeline.rng.choice(
            np.where(y[tr] == c)[0], pipeline.CALIB_PER_CLASS, replace=False
        ).tolist()
    calib_X = X[tr[calib_idx]].astype(np.float32)

    fp32, int8 = [], []
    for ep in EPOCHS:
        torch.manual_seed(pipeline.SEED)
        np.random.seed(pipeline.SEED)
        model = pipeline.task4_model()
        model, _ = pipeline.train_model(model, Xt, yt, Xv, yv, ep)
        acc_pt, _ = pipeline.evaluate_pt(model, Xe, ye)
        fp32.append(acc_pt)

        pipeline.export_onnx(model, FP32_ONNX)
        pipeline.quantize_int8(FP32_ONNX, INT8_ONNX, calib_X,
                               per_channel=True, calib_method="percentile")
        acc_i8 = float((pipeline.onnx_predict(INT8_ONNX, Xe.numpy())
                        == ye.numpy()).mean())
        int8.append(acc_i8)
        print(f"epochs={ep:3d}  fp32={acc_pt:.4f}  int8={acc_i8:.4f}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(EPOCHS, fp32, marker="o", color=pipeline.CAT[0], lw=2, label="float32")
    ax.plot(EPOCHS, int8, marker="s", color=pipeline.CAT[1], lw=2, label="int8")

    # mark the chosen default
    ax.axvline(20, color=pipeline.MUTED, lw=1.2, ls=":")
    ax.text(20, 0.02, " default=20", color=pipeline.INK2, fontsize=8,
            ha="left", va="bottom")

    # direct labels at the endpoints (no number on every point)
    ax.text(EPOCHS[-1], fp32[-1] + 0.012, f"{fp32[-1]:.3f}", color=pipeline.CAT[0],
            fontsize=8, ha="right")
    ax.text(EPOCHS[-1], int8[-1] - 0.022, f"{int8[-1]:.3f}", color=pipeline.CAT[1],
            fontsize=8, ha="right")

    ax.set_xlabel("training epochs")
    ax.set_ylabel("test accuracy")
    ax.set_xticks(EPOCHS)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.set_title("Test accuracy vs training epochs", color=pipeline.INK)
    fig.tight_layout()

    out = pipeline.FIG_DIR / "epochs_vs_accuracy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    FP32_ONNX.unlink(missing_ok=True)
    INT8_ONNX.unlink(missing_ok=True)
    print("saved", out)


if __name__ == "__main__":
    main()
