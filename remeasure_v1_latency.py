#!/usr/bin/env python3
"""Re-measure V1 baseline latency, global stats over all samples."""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import onnxruntime as ort
import time

import pipeline

WARMUP = pipeline.WARMUP
RUNS = pipeline.RUNS


def benchmark(path, x, n=10):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    all_t = []
    for r in range(n):
        for _ in range(WARMUP):
            sess.run(None, {name: x})
        for _ in range(RUNS):
            t0 = time.perf_counter()
            sess.run(None, {name: x})
            all_t.append((time.perf_counter() - t0) * 1000.0)
    a = np.sort(all_t)
    return {
        "median": float(np.median(a)),
        "p99": float(a[int(0.99 * (len(a) - 1))]),
        "mean": float(np.mean(a)),
        "min": float(a[0]),
        "max": float(a[-1]),
        "n": len(a),
    }


def main():
    X, y = pipeline.load_dataset(pipeline.ROOT)
    x1 = X[:1].astype(np.float32)
    print("== fp32 (10x100 = 1000 samples) ==")
    fp = benchmark("model_fp32.onnx", x1)
    print(f"  median {fp['median']:.3f}  p99 {fp['p99']:.3f}  mean {fp['mean']:.3f}  "
          f"min {fp['min']:.3f}  max {fp['max']:.3f}  (n={fp['n']})")
    print("== int8 (10x100 = 1000 samples) ==")
    i8 = benchmark("model_int8.onnx", x1)
    print(f"  median {i8['median']:.3f}  p99 {i8['p99']:.3f}  mean {i8['mean']:.3f}  "
          f"min {i8['min']:.3f}  max {i8['max']:.3f}  (n={i8['n']})")
    print("\nverdict: fp32 p99 {:.2f}ms {} | int8 p99 {:.2f}ms {}".format(
        fp["p99"], "FITS" if fp["p99"] < pipeline.BUDGET_MS else "TOO SLOW",
        i8["p99"], "FITS" if i8["p99"] < pipeline.BUDGET_MS else "TOO SLOW"))


if __name__ == "__main__":
    main()