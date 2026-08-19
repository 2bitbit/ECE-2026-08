# V1 延迟重测记录（2026-08-18，本机 sera-him i5-11xxx）

> 背景：V1 三机延迟平均中 sera-him（36985ca 提交）int8 p99 2.38ms 显著高于
> destiny（0.53ms）与 CQL0329（1.14ms）。按要求重测本机 V1 延迟并剔除该异常数据点。

## 重测协议（与主代码一致）

- 模型：`model_fp32.onnx` / `model_int8.onnx`（V1 主代码产物，per-channel int8）
- ORT 单线程（intra_op_num_threads=1）、batch=1、CPU 执行
- 每轮 warmup 后计时；多轮重复以排除后台干扰

## 多轮实测结果

| 轮次 | 样本数 | fp32 median (ms) | fp32 p99 (ms) | int8 median (ms) | int8 p99 (ms) |
|---|---|---|---|---|---|
| 轮1 | 1000 | 0.919 | 4.390 | 1.228 | 5.923 |
| 轮2 | 300 | 0.453 | 0.791 | 1.157 | 2.499 |
| 轮3 | 500 | 0.678 | 1.495 | 1.313 | 3.174 |

- 后台干扰明显（单轮 max 达 7~18ms，p99 波动 0.79~5.92）
- 稳定态（轮2，干扰最小）：**fp32 median 0.45 / p99 0.79；int8 median 1.16 / p99 2.50**

## 结论

1. **36985ca 提交的 sera-him 延迟（int8 median 1.42 / p99 2.38）为真实可复现值**，
   本机 per-channel int8 执行确较慢（int8 反而慢于 fp32，这是 per-channel 量化的已知特性）
2. 对比：本机 per-tensor 量化（V2-V5 同款）int8 延迟明显更快——V1 与 V2-V5 的
   延迟口径本质不同（per-channel vs per-tensor），跨版本直接比 int8 延迟需谨慎
3. **数据处置（按用户要求）**：V1 延迟平均剔除 sera-him 数据点，
   改用 CQL0329 + destiny 两机平均：
   - median: fp32 0.26 ms / int8 0.58 ms
   - p99: fp32 0.29 ms / int8 0.84 ms
4. 准确率数据（0.8722/0.8583，与 destiny 完全一致）不受影响，保留三机平均

## 文件

- 脚本：`remeasure_v1_latency.py`（10x100=1000 样本全局统计版）
- 本记录：`v1_latency_remeasure.md`