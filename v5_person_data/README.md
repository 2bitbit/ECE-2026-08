# V5 各成员数据汇总（person-level data）

> 存放各成员 V5 模型实测数据（同代码 pipeline_v5.py，seed=0，40 epochs，per-tensor + QOperator）。
> 机器差异说明：fp32 精度跨机差 ≤0.3pt（≈1 张图），int8 差 ≤1.1pt（≈4 张图），
> 体积几乎一致，V5 跨机复现性良好（明显优于 V4，V4 int8 跨机差 4.4pt）。

## 汇总对比

| 指标 | CQL0329（i7-10xxx, 9a5d925） | sera-him（i5-11xxx, 本机重跑） | florchya（最新数据） | 平均 |
|---|---|---|---|---|
| fp32 测试准确率 | 0.9278 | 0.9306 | 0.9278 | 0.9287 |
| int8 测试准确率 | 0.9194 | 0.9306 | 0.9250 | 0.9250 |
| 最差类 recall (fp32/int8) | pitted 0.7500 / 0.7667 | pitted 0.7667 / 0.7667 | pitted 0.7500 / 0.7500 | 0.7556 / 0.7611 |
| 体积 fp32/int8 (kB) | 958.7 / 246.5 | 959.7 / 246.9 | 959.5 / 246.8 | 959.3 / 246.7 |
| 延迟中位 fp32/int8 (ms) | 13.77 / 11.26 | 14.78 / 8.10 | 13.53 / 6.88 | 14.03 / 8.75 |
| 延迟 p99 fp32/int8 (ms) | 14.40 / 12.40 | **16.83 (TOO SLOW)** / 10.33 | **16.30 (TOO SLOW)** / 10.95 | 15.84 / 11.23 |
| 15ms 判定 | FITS / FITS | TOO SLOW / FITS | TOO SLOW / FITS | 仅 int8 可靠 |

## 结论

1. **V5 跨机复现性极好**：三机实测 fp32 差 0.28pt、int8 差 1.12pt、最差类相同、crazing/patches/rolled-in_scale 等类别 P/R 几乎一致。
2. **部署形态必须是 int8**：三机 int8 p99 在 10.3~12.4 ms 之间（均 FITS）；而 fp32 在两台机器上均出现超预算现象（16.83ms 和 16.30ms），彻底证明 fp32 延迟不可靠。
3. **int8 平均 0.9250**（fp32 平均 0.9287），掉点极微（不到 0.4pt），量化鲁棒性优秀。
4. 建议答辩口径：报三机平均 int8 **0.9250**，并直接说明"多机实测证明 fp32 超出 15ms 预算，int8 是唯一可靠的部署形态"。

## 目录结构

```text
v5_person_data/
├── README.md              本文件
├── cql0329/               CQL0329 机器数据（commit 9a5d925）
│   ├── results_table.csv
│   └── per_class_metrics.csv
├── sera-him/              本机重跑数据
│   ├── results_table.csv
│   └── per_class_metrics.csv
└── florchya/              florchya 新加数据
    ├── results_table.csv
    └── per_class_metrics.csv
```

## 其他成员补充

- destiny / 其他成员若跑 V5，把 `v5_output/results_table.csv` 与 `per_class_metrics.csv`
  放进对应名字的目录并更新本表即可（或推送后由助手合并）。