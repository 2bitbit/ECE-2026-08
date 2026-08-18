# Final results (average across 3 machines)

> 平均口径：新身份成员机器（CQL0329 + sera-him + destiny）。
> 说明：destiny 数据为用户提供（未在 git 历史中，附于成员数据平均报告）。

| metric | float32 (mean ± sd) | int8 (mean ± sd) |
|---|---|---|
| test accuracy | 0.8704 ± 0.0026 | 0.8472 ± 0.0157 |
| model size (kB) | 460.5333 ± 0.2055 | 124.0000 ± 0.1633 |
| latency median (ms) | 0.2933 ± 0.0613 | 0.8600 ± 0.4121 |
| latency p99 (ms) | 0.4967 ± 0.3016 | 1.3500 ± 0.7697 |
| worst class recall | 0.6556 ± 0.0079 | 0.6611 ± 0.0079 |

| verdict vs 15 ms | FITS (all machines) | FITS (all machines) |

## Per-class precision/recall (mean across 3 machines)

| class | fp32 P | fp32 R | int8 P | int8 R |
|---|---|---|---|---|
| crazing | 0.9071 | 0.9779 | 0.8949 | 0.9444 |
| inclusion | 0.7565 | 0.9668 | 0.7278 | 0.9501 |
| patches | 0.9356 | 0.8832 | 0.8888 | 0.8777 |
| pitted_surface | 0.9136 | 0.8221 | 0.8733 | 0.7722 |
| rolled-in_scale | 0.9374 | 0.9168 | 0.9464 | 0.8777 |
| scratches | 0.7973 | 0.6557 | 0.7881 | 0.6612 |

## Machines (included)

- Windows 10, Intel i7-10xxx (1eb549e) (author CQL0329)
- Windows 10, Intel i5-11xxx (36985ca) (author sera-him)
- destiny (user-provided run, not in git history)

## Machines (excluded)

- Linux WSL2 x86_64 (b89ad93) (author ChenQile329, old identity — excluded per team instruction)