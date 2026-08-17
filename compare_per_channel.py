#!/usr/bin/env python3
"""
A/B: per-channel vs per-tensor int8 quantisation.

Reuses the existing model_fp32.onnx (the exact float32 model behind the
reported numbers) and applies the SAME static int8 quantisation twice, with
the only variable being `per_channel`. Reports test accuracy for each.

This turns the defence question "per-channel or per-tensor — what did the
other one give?" into a measured number instead of a reason.

Run:  python compare_per_channel.py
"""

import numpy as np

import pipeline

FP32 = pipeline.HERE / "model_fp32.onnx"
PC = pipeline.HERE / "_tmp_int8_perchannel.onnx"
PT = pipeline.HERE / "_tmp_int8_pertensor.onnx"


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

    PC.unlink(missing_ok=True)
    PT.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
