# Final results (average across 2 machines, new identities only)

> 平均口径：仅统计新身份成员机器（CQL0329 + sera-him），忽略旧身份（ChenQile329）机器数据。
> 原因：旧身份提交（b89ad93）为 08-16 早期身份 ChenQile329 所产生，按团队要求不纳入最终平均。

| metric | float32 (mean ± sd) | int8 (mean ± sd) |
|---|---|---|
| test accuracy | 0.8695 ± 0.0027 | 0.8417 ± 0.0166 |
| model size (kB) | 460.5500 ± 0.2500 | 124.0000 ± 0.2000 |
| latency median (ms) | 0.3300 ± 0.0400 | 1.0700 ± 0.3500 |
| latency p99 (ms) | 0.6250 ± 0.2950 | 1.7600 ± 0.6200 |
| worst class recall | 0.6500 ± 0.0000 | 0.6583 ± 0.0083 |

| verdict vs 15 ms | FITS (all machines) | FITS (all machines) |

## Per-class precision/recall (mean across 2 machines)

| class | fp32 P | fp32 R | int8 P | int8 R |
|---|---|---|---|---|
| crazing | 0.9077 | 0.9833 | 0.8899 | 0.9416 |
| inclusion | 0.7532 | 0.9667 | 0.7291 | 0.9417 |
| patches | 0.9464 | 0.8833 | 0.8762 | 0.8750 |
| pitted_surface | 0.9160 | 0.8167 | 0.8655 | 0.7583 |
| rolled-in_scale | 0.9322 | 0.9166 | 0.9376 | 0.8750 |
| scratches | 0.7880 | 0.6500 | 0.7822 | 0.6583 |

## Machines (included)

- Windows 10, Intel i7-10xxx (1eb549e) (author CQL0329)
- Windows 10, Intel i5-11xxx (36985ca) (author sera-him)

## Machines (excluded)

- Linux WSL2 x86_64 (b89ad93) (author ChenQile329, old identity — excluded per team instruction)