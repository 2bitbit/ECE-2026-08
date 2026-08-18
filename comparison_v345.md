# 版本对比 — baseline / V3 / V4 / V5（本机实测）

运行时间：2026-08-18。数据来源：baseline 取自 `README.md` 本机数据；V3/V4/V5 由 `pipeline_v3/v4/v5.py` 本次运行生成（各版本输出目录 `v*_output/`）。

| 版本 | 思路 | fp32 acc | int8 acc | 体积 fp32/int8 (kB) | p99 延迟 fp32/int8 (ms) | 最差类 | fp32/int8 recall | verdict |
|---|---|---|---|---|---|---|---|---|
| baseline | 96² + 115k + Flatten | 0.8722 | 0.8583 | 461 / 124 | 0.44 / 1.48 | scratches | 0.650 / 0.650 | FITS |
| V3 | 128² + 412k + 4×TTA | 0.8639 | 0.8472 | 1619 / 408 | 7.55 / 8.26 | scratches | 0.733 / 0.717 | FITS |
| V4 | 96² + 60k + GAP + Cutout | 0.8861 | 0.8306 | 254 / 70 | 0.60 / 0.89 | pitted_surface | 0.633 / 0.467 | FITS |
| V5 | 128² + 240k + GAP + TTA | 0.9278 | 0.9194 | 959 / 247 | 14.40 / 12.40 | pitted_surface | 0.750 / 0.767 | FITS |

## median 延迟（补充）

| 版本 | fp32 median (ms) | int8 median (ms) |
|---|---|---|
| baseline | 0.31 | 0.65 |
| V3 | 6.09 | 6.96 |
| V4 | 0.58 | 0.79 |
| V5 | 13.77 | 11.26 |

## 结论

- **V5 精度最高**（fp32 0.9278 / int8 0.9194），且 int8 掉点仅 0.8 个点，量化鲁棒性最好；但 fp32 p99 延迟 14.40ms 仅剩 0.6ms 余量（4%），真机负载下可能超时。int8（12.40ms）才是实际部署形态。
- **V3 失败**：纯堆分辨率+宽度+TTA 反而过拟合（0.8639 < baseline 0.8722），且延迟 ~17×。
- **V4 效率甜点**：60k 参数拿到 0.8861 fp32、延迟/体积最优；但 int8 掉 5.5 个点（pitted_surface recall 0.467），量化鲁棒性最差。

## 注意

- V4 / V5 的 `write_results` 未写 `results_table.md`（仅 V3 有），本表数据取自各版本 `results_table.csv` 与控制台输出。
- 各版本完整 per-class 精度/召回见各自 `v*_output/per_class_metrics.csv`。
