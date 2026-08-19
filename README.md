# Project 1 —— Edge Vision Recognition & Model Quantisation

## 📌 Project Overview
This project aims to develop a lightweight surface defect visual recognition system for a hot-rolled steel strip production line. In our latest V5 iteration, we trained a Convolutional Neural Network (CNN) widened to **241,030 parameters** to classify 6 typical classes of steel surface defects (e.g., crazing, patches, pitted surface) with high accuracy (~93%).

To meet the extremely strict deployment constraints of edge devices (target: Raspberry Pi 5), we exported the trained PyTorch model to the ONNX format and applied **Static INT8 Quantisation (Static PTQ)**. Notably, the capacity upgrade in V5 caused the base Float32 model to violate the strict "15 ms per item" inference latency budget. Quantization proved to be the decisive step that rescued the project, shrinking the model footprint significantly while bringing the latency safely back within the operational budget.

## 🚀 Reproduction Steps
This project follows strict engineering reproducibility standards (using a fixed `seed=0`, stratified splitting, and 40 training epochs). Ensuring the `NEU-DET` dataset is placed in the project root, please run the following steps:

1. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv .venv
   # For Windows:
   .venv\Scripts\activate
   # For Mac/Linux:
   source .venv/bin/activate
   ```

2. **Install Core Dependencies**:
   Since our deployment target is a CPU-only edge device, we recommend installing the CPU version of PyTorch to save space, followed by all project requirements:
   ```bash
   pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
   pip install -r requirements.txt
   ```

3. **Verify Environment**:
   ```bash
   python verify.py
   # Expected output at the end: PASS
   ```

4. **Launch the End-to-End Pipeline**:
   ```bash
   python pipeline_v5.py
   ```
   *Note: This single command sequentially executes Data Loading → Stratified Splitting → Model Training (40 Epochs) → ONNX Export → Calibration & INT8 Quantisation → Latency Benchmarking → Final Report Generation.*

## 📊 Core Empirical Metrics
The following are the core acceptance metrics empirically measured and averaged across multiple test machines (i7/i5 CPUs) for the V5 model:

| Metric | Float32 (Baseline) | INT8 (Quantised) | Delta / Verdict |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | 0.9287 | 0.9250 | Extreme robustness (-0.003) |
| **Model Disk Size** | 959.3 kB | 246.7 kB | **Shrunk by 3.9x** |
| **Latency Median** | 14.03 ms | 8.75 ms | - |
| **Latency p99** | 15.84 ms | 11.23 ms | - |
| **15 ms Budget** | ❌ **TOO SLOW** | ✅ **FITS** | INT8 is mandatory |
| **Worst Class Recall**| pitted_surface (0.75) | pitted_surface (0.76) | Confused with patches |

> **⚠️ Engineering Note on "The Latency Trap":**
> Multi-machine benchmarking revealed a critical hardware constraint: while widening the model to 241k parameters successfully pushed accuracy from 86% to 92.8%, it caused the Float32 p99 latency to hit 15.8 ~ 16.8 ms, violating the strict 15 ms production line budget. Therefore, **INT8 quantization is no longer just an optional optimization; it is the only mathematically viable deployment format** that guarantees operational safety (11.23 ms).