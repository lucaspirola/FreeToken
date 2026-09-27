## Part 1: probe, q8_0 KV, 262144 ceiling (pass 2; p1 in brackets)

### whole (p1-whole-1, p1-whole-2, p1-whole-3, p1-whole-4)

| prompt | p1-whole-1 prefill tok/s | p1-whole-2 prefill tok/s | p1-whole-3 prefill tok/s | p1-whole-4 prefill tok/s | p1-whole-1 TTFT s | p1-whole-2 TTFT s | p1-whole-3 TTFT s | p1-whole-4 TTFT s | p1-whole-1 decode tok/s | p1-whole-2 decode tok/s | p1-whole-3 decode tok/s | p1-whole-4 decode tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8K | 8498.0 (20610.0) | 8425.0 (20404.0) | 7648.0 (18287.0) | 7326.0 (18036.0) | 0.9 (0.4) | 1.0 (0.4) | 1.0 (0.4) | 1.1 (0.4) | 204.9 (209.8) | 204.9 (207.5) | 188.2 (186.9) | 182.1 (189.7) |
| 32K | 7880.0 (7632.0) | 7924.0 (7934.0) | 7215.0 (7191.0) | 7091.0 (7243.0) | 4.1 (4.2) | 4.0 (4.0) | 4.4 (4.5) | 4.5 (4.4) | 192.7 (196.5) | 195.3 (195.8) | 178.0 (178.3) | 174.4 (179.6) |
| 80K | 6304.0 (6327.0) | 6356.0 (6360.0) | 5662.0 (5772.0) | 5743.0 (5687.0) | 12.7 (12.6) | 12.6 (12.6) | 14.1 (13.9) | 13.9 (14.1) | 174.4 (174.9) | 175.7 (174.5) | 158.0 (159.0) | 159.0 (159.4) |
| 128K | 5218.0 (5213.0) | 5239.0 (5240.0) | 4746.0 (4736.0) | 4749.0 (4742.0) | 24.5 (24.6) | 24.4 (24.4) | 27.0 (27.0) | 27.0 (27.0) | 157.7 (157.3) | 158.5 (158.7) | 143.0 (143.2) | 144.6 (143.2) |
| 256K | 3578.0 (3574.0) | 3595.0 (3587.0) | 3256.0 (3243.0) | 3247.0 (3246.0) | 71.6 (71.6) | 71.2 (71.4) | 78.6 (79.0) | 78.8 (78.9) | 126.2 (125.9) | 126.6 (126.9) | 115.2 (115.2) | 115.0 (115.9) |

`p1-whole-1`: ram_gib 22.24, rss_ready_gib 22.96, gpu_mib 13978, coverage_faults None, starved None; expert slots at start 4928, at the 256K KV commit 3832 (KV 2.73 GiB); prefill transient 1.1 GiB; host load mean/max 2.1 / 4.9
`p1-whole-2`: ram_gib 21.16, rss_ready_gib 22.96, gpu_mib 13978, coverage_faults None, starved None; expert slots at start 4928, at the 256K KV commit 3832 (KV 2.73 GiB); prefill transient 1.1 GiB; host load mean/max 3.7 / 7.9
`p1-whole-3`: ram_gib 22.07, rss_ready_gib 22.99, gpu_mib 13978, coverage_faults None, starved None; expert slots at start 4928, at the 256K KV commit 3832 (KV 2.73 GiB); prefill transient 1.1 GiB; host load mean/max 6.3 / 10.5
`p1-whole-4`: ram_gib 19.87, rss_ready_gib 22.61, gpu_mib 13994, coverage_faults None, starved None; expert slots at start 4928, at the 256K KV commit 3832 (KV 2.73 GiB); prefill transient 1.1 GiB; host load mean/max 8.5 / 20.0

### saver (p1-saver-1, p1-saver-2, p1-saver-3, p1-saver-4)

| prompt | p1-saver-1 prefill tok/s | p1-saver-2 prefill tok/s | p1-saver-3 prefill tok/s | p1-saver-4 prefill tok/s | p1-saver-1 TTFT s | p1-saver-2 TTFT s | p1-saver-3 TTFT s | p1-saver-4 TTFT s | p1-saver-1 decode tok/s | p1-saver-2 decode tok/s | p1-saver-3 decode tok/s | p1-saver-4 decode tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8K | 8348.0 (24286.0) | 7960.0 (24614.0) | 7615.0 (22737.0) | 7296.0 (22085.0) | 1.0 (0.3) | 1.0 (0.3) | 1.1 (0.4) | 1.1 (0.4) | 197.3 (200.0) | 192.1 (192.4) | 174.8 (180.9) | 175.4 (181.5) |
| 32K | 7826.0 (7818.0) | 7640.0 (7709.0) | 6944.0 (7104.0) | 6996.0 (7096.0) | 4.1 (4.1) | 4.2 (4.2) | 4.6 (4.5) | 4.6 (4.5) | 186.0 (188.9) | 182.3 (182.5) | 167.3 (170.3) | 168.4 (170.4) |
| 80K | 6132.0 (6238.0) | 6140.0 (6146.0) | 5664.0 (5606.0) | 5653.0 (5578.0) | 13.1 (12.8) | 13.0 (13.0) | 14.1 (14.3) | 14.2 (14.3) | 153.2 (162.6) | 165.9 (164.9) | 153.3 (151.8) | 152.7 (152.5) |
| 128K | 5154.0 (5015.0) | 5178.0 (5108.0) | 4679.0 (4680.0) | 4661.0 (4674.0) | 24.8 (25.5) | 24.7 (25.1) | 27.4 (27.4) | 27.5 (27.4) | 153.2 (153.1) | 153.4 (149.6) | 139.5 (137.1) | 139.3 (138.2) |
| 256K | 3551.0 (3504.0) | 3554.0 (3502.0) | 3208.0 (3218.0) | 3212.0 (3216.0) | 72.1 (73.1) | 72.0 (73.1) | 79.8 (79.6) | 79.7 (79.6) | 123.2 (122.4) | 123.1 (119.2) | 110.3 (111.2) | 111.3 (108.6) |

`p1-saver-1`: ram_gib 16.67, rss_ready_gib 17.6, gpu_mib 13978, coverage_faults 0, starved 0; expert slots at start 4864, at the 256K KV commit 3768 (KV 2.73 GiB); prefill transient 1.1 GiB; host load mean/max 5.0 / 10.4
`p1-saver-2`: ram_gib 16.62, rss_ready_gib 17.42, gpu_mib 13978, coverage_faults 0, starved 0; expert slots at start 4864, at the 256K KV commit 3768 (KV 2.73 GiB); prefill transient 1.1 GiB; host load mean/max 6.5 / 11.4
`p1-saver-3`: ram_gib 16.64, rss_ready_gib 17.6, gpu_mib 13994, coverage_faults 0, starved 0; expert slots at start 4864, at the 256K KV commit 3768 (KV 2.73 GiB); prefill transient 1.1 GiB; host load mean/max 1.3 / 2.4
`p1-saver-4`: ram_gib 16.76, rss_ready_gib 17.61, gpu_mib 13994, coverage_faults 0, starved 0; expert slots at start 4864, at the 256K KV commit 3768 (KV 2.73 GiB); prefill transient 1.1 GiB; host load mean/max 1.2 / 2.4

### saver / whole, pass 2, mean of arms 1-2 (06:49-07:35) (whole: p1-whole-1, p1-whole-2; saver: p1-saver-1, p1-saver-2)

| prompt | prefill | TTFT | decode |
|---|---:|---:|---:|
| 8K | 8154 / 8462 = 96.4% | 0.98 / 0.95 s | 194.7 / 204.9 = 95.0% |
| 32K | 7733 / 7902 = 97.9% | 4.14 / 4.05 s | 184.2 / 194.0 = 94.9% |
| 80K | 6136 / 6330 = 96.9% | 13.04 / 12.64 s | 159.6 / 175.1 = 91.1% |
| 128K | 5166 / 5228 = 98.8% | 24.78 / 24.49 s | 153.3 / 158.1 = 97.0% |
| 256K | 3552 / 3586 = 99.1% | 72.07 / 71.39 s | 123.2 / 126.4 = 97.4% |

### saver / whole, pass 2, mean of arms 3-4 (10:22-10:54, slow period) (whole: p1-whole-3, p1-whole-4; saver: p1-saver-3, p1-saver-4)

| prompt | prefill | TTFT | decode |
|---|---:|---:|---:|
| 8K | 7456 / 7487 = 99.6% | 1.08 / 1.07 s | 175.1 / 185.1 = 94.6% |
| 32K | 6970 / 7153 = 97.4% | 4.59 / 4.48 s | 167.9 / 176.2 = 95.3% |
| 80K | 5658 / 5702 = 99.2% | 14.14 / 14.03 s | 153.0 / 158.5 = 96.5% |
| 128K | 4670 / 4748 = 98.4% | 27.42 / 26.97 s | 139.4 / 143.8 = 96.9% |
| 256K | 3210 / 3252 = 98.7% | 79.76 / 78.74 s | 110.8 / 115.1 = 96.3% |


### natural text (5 tasks, 3500 max tokens, 8K warm-up only)

| arm | decode tok/s | md5 same as p1nat-whole-1 | host load mean / max |
|---|---:|---|---|
| p1nat-saver-1 | 154.6 | 5/5 | 4.8 / 5.2 |
| p1nat-saver-2 | 154.5 | 5/5 | 7.7 / 10.6 |
| p1nat-saver-3 | 153.7 | 5/5 | 2.2 / 3.4 |
| p1nat-saver-4 | 153.6 | 5/5 | 2.2 / 3.4 |
| p1nat-whole-1 | 167.4 | 5/5 | 5.7 / 7.7 |
| p1nat-whole-2 | 168.2 | 5/5 | 12.5 / 23.2 |
| p1nat-whole-3 | 167.9 | 5/5 | 4.9 / 9.3 |
| p1nat-whole-4 | 168.3 | 5/5 | 4.2 / 9.1 |

## Part 2: KV lanes, saver

### ceiling 256k (262144)

q8q8 reference arms: `kv-q8q8-256k`, `kvnat-q8q8-256k`

| lane (arm) | 256K prefill tok/s | 256K TTFT s | 256K decode tok/s | 8K decode tok/s | natural tok/s | output vs q8q8 (probe 8K/256K p1,p2; natural) | KV GiB at 256K | expert slots start / at max KV | pool rows (GiB) | ram_gib / rss_ready_gib | gpu_mib | load probe / natural arm |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---|
| q8q8 (`kv-q8q8-256k`, `kvnat-q8q8-256k`) | 3569.0 (3553.0) | 71.74 | 123.1 (121.9) | 194.4 | 154.1 | ====; 5/5 | 2.73 | 4864 / 3768 | 7290 (13.45) | 17.68 / 17.6 | 13978 | 1.5 / 2.2; 4.3 / 9.4 |
| q8q6 (`kv-q8q6-256k`, `kvnat-q8q6-256k`) | 855.0 (865.0) | 299.53 | 107.3 (107.4) | 197.3 | 150.8 | ≠≠≠≠; 0/5 | 2.42 | 4912 / 3936 | 7118 (13.14) | 16.3 / 17.34 | 13990 | 1.9 / 5.6; 5.8 / 9.4 |
| q6q5 (`kv-q6q5-256k`, `kvnat-q6q5-256k`) | 1441.0 (1437.0) | 177.65 | 111.3 (111.5) | 197.3 | 152.1 | ≠≠≠≠; 0/5 | 1.95 | 4976 / 4192 | 6860 (12.66) | 15.8 / 16.84 | 13990 | 6.8 / 10.8; 2.1 / 4.1 |
| q4q4 (`kv-q4q4-256k`, `kvnat-q4q4-256k`) | 971.0 (994.0) | 263.59 | 114.3 (124.5) | 201.2 | 152.5 | ≠≠≠≠; 0/5 | 1.48 | 5040 / 4448 | 6602 (12.18) | 15.37 / 16.38 | 13990 | 1.2 / 3.4; 1.1 / 1.5 |

### ceiling 384k (--rope-yarn-factor 2, 393216)

q8q8 reference arms: `kv-q8q8-384k`, `kvnat-q8q8-384k`

| lane (arm) | 384K prefill tok/s | 384K TTFT s | 384K decode tok/s | 8K decode tok/s | natural tok/s | output vs q8q8 (probe 8K/384K p1,p2; natural) | KV GiB at 384K | expert slots start / at max KV | pool rows (GiB) | ram_gib / rss_ready_gib | gpu_mib | load probe / natural arm |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---|
| q8q8 (`kv-q8q8-384k`, `kvnat-q8q8-384k`) | 2466.0 (2551.0) | 155.73 | 90.3 (89.3) | 169.1 | 153.8 | ====; 5/5 | 4.06 | 4832 / 3024 | 8042 (14.84) | 18.16 / 19.07 | 13976 | 1.1 / 1.2; 1.2 / 2.3 |
| q8q6 (`kv-q8q6-384k`, `kvnat-q8q6-384k`) | 533.0 (533.0) | 719.89 | 77.9 (77.1) | 177.0 | 152.9 | ≠≠≠≠; 0/5 | 3.59 | 4888 / 3280 | 7784 (14.36) | 17.53 / 18.59 | 13984 | 4.0 / 11.7; 1.3 / 1.7 |
| q6q5 (`kv-q6q5-384k`, `kvnat-q6q5-384k`) | 919.0 (919.0) | 417.69 | 81.9 (82.7) | 176.8 | 152.9 | ≠≠≠≠; 0/5 | 2.89 | 4944 / 3656 | 7397 (13.65) | 17.09 / 17.88 | 13966 | 2.1 / 8.0; 1.0 / 2.3 |
| q4q4 (`kv-q4q4-384k`, `kvnat-q4q4-384k`) | 612.0 (612.0) | 627.90 | 94.4 (93.9) | 179.0 | 151.4 | ≠≠≠≠; 0/5 | 2.19 | 5008 / 4040 | 7010 (12.94) | 16.19 / 17.16 | 13968 | 3.8 / 10.9; 6.6 / 10.8 |

### every part-2 probe arm (pass 2 of the big prompt; load = window mean / max)

| arm | prefill tok/s | TTFT s | decode tok/s | 8K decode | out_sha1 (big p2) | load |
|---|---:|---:|---:|---:|---|---|
| kv-q4q4-256k | 971.0 | 263.59 | 114.3 | 201.2 | a9f3339637 | 1.2 / 3.4 |
| kv-q4q4-384k | 612.0 | 627.90 | 94.4 | 179.0 | a7c394def5 | 3.8 / 10.9 |
| kv-q4q4-384k-r2 | 627.0 | 612.75 | 95.7 | 183.4 | a7c394def5 | 5.4 / 11.8 |
| kv-q6q5-256k | 1441.0 | 177.65 | 111.3 | 197.3 | 8db5b7ed72 | 6.8 / 10.8 |
| kv-q6q5-256k-r2 | 1340.0 | 191.13 | 103.0 | 182.6 | 8db5b7ed72 | 1.4 / 1.9 |
| kv-q6q5-384k | 919.0 | 417.69 | 81.9 | 176.8 | 78496392aa | 2.1 / 8.0 |
| kv-q8q6-256k | 855.0 | 299.53 | 107.3 | 197.3 | deb037e24f | 1.9 / 5.6 |
| kv-q8q6-384k | 533.0 | 719.89 | 77.9 | 177.0 | 53192ab32a | 4.0 / 11.7 |
| kv-q8q6-384k-r2 | 546.0 | 703.23 | 79.7 | 179.2 | 53192ab32a | 3.6 / 10.7 |
| kv-q8q8-256k | 3569.0 | 71.74 | 123.1 | 194.4 | a983be5886 | 1.5 / 2.2 |
| kv-q8q8-256k-r2 | 3271.0 | 78.27 | 112.1 | 179.4 | a983be5886 | 1.5 / 2.3 |
| kv-q8q8-256k-rep | 3218.0 | 79.55 | 110.4 | 176.5 | a983be5886 | 8.0 / 10.5 |
| kv-q8q8-256k-triton | 1859.0 | 137.74 | 110.8 | 177.8 | a24504c26b | – |
| kv-q8q8-384k | 2466.0 | 155.73 | 90.3 | 169.1 | 450f1844c4 | 1.1 / 1.2 |
| kv-q8q8-384k-r2 | 2521.0 | 152.30 | 92.4 | 178.1 | 450f1844c4 | 1.5 / 2.5 |

| natural arm | tok/s | load |
|---|---:|---|
| kvnat-q4q4-256k | 152.5 | 1.1 / 1.5 |
| kvnat-q4q4-384k | 151.4 | 6.6 / 10.8 |
| kvnat-q4q4-384k-r2 | 150.6 | 8.2 / 11.6 |
| kvnat-q6q5-256k | 152.1 | 2.1 / 4.1 |
| kvnat-q6q5-384k | 152.9 | 1.0 / 2.3 |
| kvnat-q8q6-256k | 150.8 | 5.8 / 9.4 |
| kvnat-q8q6-256k-r2 | 151.4 | 2.3 / 4.7 |
| kvnat-q8q6-384k | 152.9 | 1.3 / 1.7 |
| kvnat-q8q8-256k | 154.1 | 4.3 / 9.4 |
| kvnat-q8q8-256k-r2 | 152.8 | 2.0 / 5.8 |
| kvnat-q8q8-384k | 153.8 | 1.2 / 2.3 |

Drift check, q8q8 256K repeated at the end: decode 123.1 -> 110.4, prefill 3569.0 -> 3218.0, output identical.
