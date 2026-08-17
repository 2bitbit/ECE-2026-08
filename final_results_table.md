# Final results (average across 3 machines)

| metric | float32 (mean ± sd) | int8 (mean ± sd) |
|---|---|---|
| test accuracy | 0.8676 ± 0.0034 | 0.8454 ± 0.0146 |
| model size (kB) | 460.5667 ± 0.2055 | 124.0333 ± 0.1700 |
| latency median (ms) | 0.3233 ± 0.0340 | 0.8367 ± 0.4365 |
| latency p99 (ms) | 0.5400 ± 0.2692 | 1.3200 ± 0.8022 |
| worst class recall | 0.6500 ± 0.0000 | 0.6500 ± 0.0136 |

| verdict vs 15 ms | FITS (all machines) | FITS (all machines) |

## Per-class precision/recall (mean across machines)

| class | fp32 P | fp32 R | int8 P | int8 R |
|---|---|---|---|---|
| crazing | 0.9031 | 0.9833 | 0.8949 | 0.9444 |
| inclusion | 0.7532 | 0.9667 | 0.7219 | 0.9500 |
| patches | 0.9461 | 0.8778 | 0.8941 | 0.8778 |
| pitted_surface | 0.9131 | 0.8167 | 0.8733 | 0.7722 |
| rolled-in_scale | 0.9318 | 0.9111 | 0.9406 | 0.8778 |
| scratches | 0.7853 | 0.6500 | 0.7853 | 0.6500 |

## Machines

- Linux WSL2 x86_64 (b89ad93) (author ChenQile329)
- Windows 10, Intel i7-10xxx (1eb549e) (author CQL0329)
- Windows 10, Intel i5-11xxx (36985ca) (author sera-him)
