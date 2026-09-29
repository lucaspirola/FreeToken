## Arms

| arm | start (load, GPU, avail, SM) | yarn | KV ceiling (GiB) | KV max committed (GiB) | slots start -> at max | idle GPU MiB / avail GiB | peak GPU MiB / min avail GiB | ram_gib / rss_ready / peak cgroup | cov faults / starved | R3 |
|---|---|---|---|---|---|---|---|---|---|---|
| c-sp | 0.50, 0MiB, 29.8, 277 MHz | None/None | 262144 (0.74) | 262144 (2.73) | 4952 -> 3856 | 14184 / 12.55 | 15098 / 10.22 | 17.29 / 17.43 / 18.79 | 0 / 0 | captures=1 kv_grows=10 tracebacks=0 |
| y-sp | 7.80, 0MiB, 27.5, 277 MHz | 2.0/262144 | 393216 (0.74) | 393216 (4.06) | 4912 -> 3088 | 14120 / 9.94 | 15102 / 8.57 | 17.84 / 18.92 / 19.84 | 0 / 0 | captures=1 kv_grows=36 tracebacks=0 |
| y-wp | 2.14, 0MiB, 29.1, 277 MHz | 2.0/262144 | 393216 (0.74) | 393216 (4.06) | 4984 -> 3168 | 14114 / 6.99 | 15098 / 4.7 | 21.97 / 22.82 / 25.85 | – / – | captures=1 kv_grows=36 tracebacks=0 |
| c-wp | 0.69, 0MiB, 30.2, 277 MHz | None/None | 262144 (0.74) | 262144 (2.73) | 5016 -> 3920 | 15078 / 8.2 | 15082 / 4.96 | 21.97 / 22.87 / 27.35 | – / – | captures=1 kv_grows=10 tracebacks=0 |
| y-sq | 1.37, 0MiB, 30.0, 277 MHz | 2.0/262144 | 393216 (0.74) | 393216 (4.06) | 4912 -> 3088 | 15080 / 11.93 | 15086 / 7.16 | 17.3 / 18.94 / 21.46 | 0 / 0 | captures=1 kv_grows=82 tracebacks=0 |
| y-wq | 0.84, 0MiB, 29.4, 277 MHz | 2.0/262144 | 393216 (0.74) | 393216 (4.06) | 4984 -> 3168 | 14114 / 7.36 | 15106 / 4.53 | 22.2 / 22.38 / 26.19 | – / – | captures=1 kv_grows=82 tracebacks=0 |
| c-sn | 5.56, 0MiB, 30.2, 277 MHz | None/None | 262144 (0.74) | – (–) | 4952 -> – | – / – | – / – | 16.52 / 17.52 / 21.33 | 0 / 0 | captures=1 kv_grows=0 tracebacks=0 |
| y-sn | 2.08, 0MiB, 30.2, 277 MHz | 2.0/262144 | 393216 (0.74) | – (–) | 4912 -> – | 15110 / 12.11 | 15110 / 11.74 | 17.94 / 18.96 / 18.89 | 0 / 0 | captures=1 kv_grows=0 tracebacks=0 |
| y-wn | 1.04, 0MiB, 30.3, 277 MHz | 2.0/262144 | 393216 (0.74) | – (–) | 4984 -> – | 14114 / 8.47 | 15076 / 8.17 | 21.99 / 23.0 / 25.21 | – / – | captures=1 kv_grows=0 tracebacks=0 |

* c-sp prediction checks: non-expert weights predicted 2.50 GiB, measured 2.69 GiB (-7%); CUDA graph pool (max bs 1) predicted 0.03 GiB, measured 0.04 GiB (-24%); prefill transient (8192-token chunk, peak layer gdn) predicted 0.88 GiB, measured 0.94 GiB (-7%); linear-state pool predicted 0.78 GiB, measured 0.78 GiB (+0%)
* y-sp prediction checks: non-expert weights predicted 2.50 GiB, measured 2.73 GiB (-8%); CUDA graph pool (max bs 1) predicted 0.03 GiB, measured 0.04 GiB (-27%); prefill transient (8192-token chunk, peak layer gdn) predicted 0.88 GiB, measured 0.96 GiB (-9%); linear-state pool predicted 0.78 GiB, measured 0.78 GiB (+0%)
* y-wp prediction checks: non-expert weights predicted 2.50 GiB, measured 2.73 GiB (-8%); CUDA graph pool (max bs 1) predicted 0.03 GiB, measured 0.04 GiB (-24%); prefill transient (8192-token chunk, peak layer gdn) predicted 0.88 GiB, measured 0.96 GiB (-9%); linear-state pool predicted 0.78 GiB, measured 0.78 GiB (+0%)
* c-wp prediction checks: non-expert weights predicted 2.50 GiB, measured 2.69 GiB (-7%); CUDA graph pool (max bs 1) predicted 0.03 GiB, measured 0.04 GiB (-20%); prefill transient (8192-token chunk, peak layer gdn) predicted 0.88 GiB, measured 0.94 GiB (-7%); linear-state pool predicted 0.78 GiB, measured 0.78 GiB (+0%)
* y-sq prediction checks: non-expert weights predicted 2.50 GiB, measured 2.73 GiB (-8%); CUDA graph pool (max bs 1) predicted 0.03 GiB, measured 0.04 GiB (-27%); prefill transient (8192-token chunk, peak layer gdn) predicted 0.88 GiB, measured 0.96 GiB (-9%); linear-state pool predicted 0.78 GiB, measured 0.78 GiB (+0%)
* y-wq prediction checks: non-expert weights predicted 2.50 GiB, measured 2.73 GiB (-8%); CUDA graph pool (max bs 1) predicted 0.03 GiB, measured 0.04 GiB (-24%); prefill transient (8192-token chunk, peak layer gdn) predicted 0.88 GiB, measured 0.96 GiB (-9%); linear-state pool predicted 0.78 GiB, measured 0.78 GiB (+0%)
* c-sn prediction checks: non-expert weights predicted 2.50 GiB, measured 2.69 GiB (-7%); CUDA graph pool (max bs 1) predicted 0.03 GiB, measured 0.04 GiB (-24%); prefill transient (8192-token chunk, peak layer gdn) predicted 0.88 GiB, measured 0.94 GiB (-7%); linear-state pool predicted 0.78 GiB, measured 0.78 GiB (+0%)
* y-sn prediction checks: non-expert weights predicted 2.50 GiB, measured 2.73 GiB (-8%); CUDA graph pool (max bs 1) predicted 0.03 GiB, measured 0.04 GiB (-27%); prefill transient (8192-token chunk, peak layer gdn) predicted 0.88 GiB, measured 0.96 GiB (-9%); linear-state pool predicted 0.78 GiB, measured 0.78 GiB (+0%)
* y-wn prediction checks: non-expert weights predicted 2.50 GiB, measured 2.73 GiB (-8%); CUDA graph pool (max bs 1) predicted 0.03 GiB, measured 0.04 GiB (-24%); prefill transient (8192-token chunk, peak layer gdn) predicted 0.88 GiB, measured 0.96 GiB (-9%); linear-state pool predicted 0.78 GiB, measured 0.78 GiB (+0%)

## Probe, YaRN 393216

| target | y-sp p1 | y-sp p2 | y-wp p1 | y-wp p2 |
|---|---|---|---|---|
| 1000 prompt tok | 1017 | 1019 | 1017 | 1019 |
| 8000 prompt tok | 8009 | 8011 | 8009 | 8011 |
| 32000 prompt tok | 32022 | 32024 | 32022 | 32024 |
| 80000 prompt tok | 80023 | 80025 | 80023 | 80025 |
| 128000 prompt tok | 128025 | 128027 | 128025 | 128027 |
| 256000 prompt tok | 256020 | 256022 | 256020 | 256022 |
| 262000 prompt tok | 262023 | 262025 | 262023 | 262025 |
| 300000 prompt tok | 300019 | 300021 | 300019 | 300021 |
| 380000 prompt tok | 380013 | 380015 | 380013 | 380015 |
| 1000 TTFT s | 0.35 | 0.51 | 0.26 | 0.41 |
| 8000 TTFT s | 0.32 | 0.90 | 0.24 | 0.82 |
| 32000 TTFT s | 4.02 | 4.01 | 3.57 | 3.69 |
| 80000 TTFT s | 12.79 | 12.73 | 12.32 | 11.93 |
| 128000 TTFT s | 24.74 | 24.77 | 22.95 | 23.42 |
| 256000 TTFT s | 72.98 | 72.75 | 70.22 | 69.83 |
| 262000 TTFT s | 75.55 | 75.48 | 72.69 | 72.77 |
| 300000 TTFT s | 94.99 | 94.44 | 91.54 | 91.20 |
| 380000 TTFT s | 141.50 | 140.65 | 137.51 | 137.04 |
| 1000 prefill tok/s | 2912 | 1989 | 3858 | 2502 |
| 8000 prefill tok/s | 25156 | 8861 | 32991 | 9793 |
| 32000 prefill tok/s | 7962 | 7994 | 8980 | 8683 |
| 80000 prefill tok/s | 6256 | 6286 | 6495 | 6710 |
| 128000 prefill tok/s | 5174 | 5168 | 5577 | 5467 |
| 256000 prefill tok/s | 3508 | 3519 | 3646 | 3666 |
| 262000 prefill tok/s | 3468 | 3471 | 3605 | 3601 |
| 300000 prefill tok/s | 3158 | 3177 | 3278 | 3290 |
| 380000 prefill tok/s | 2686 | 2702 | 2763 | 2773 |
| 1000 decode tok/s | 192.1 | 174.2 | 198.4 | 185.8 |
| 8000 decode tok/s | 190.3 | 178.8 | 205.1 | 186.9 |
| 32000 decode tok/s | 168.0 | 172.2 | 180.1 | 177.4 |
| 80000 decode tok/s | 152.5 | 152.6 | 147.5 | 158.4 |
| 128000 decode tok/s | 139.8 | 142.2 | 136.5 | 146.3 |
| 256000 decode tok/s | 112.4 | 112.0 | 116.6 | 116.7 |
| 262000 decode tok/s | 101.4 | 98.8 | 105.3 | 106.6 |
| 300000 decode tok/s | 104.0 | 97.2 | 107.7 | 104.2 |
| 380000 decode tok/s | 91.6 | 94.2 | 97.7 | 99.3 |

## Probe, control 262144

| target | c-sp p1 | c-sp p2 | c-wp p1 | c-wp p2 |
|---|---|---|---|---|
| 1000 prompt tok | 1017 | 1019 | 1017 | 1019 |
| 8000 prompt tok | 8009 | 8011 | 8009 | 8011 |
| 32000 prompt tok | 32022 | 32024 | 32022 | 32024 |
| 80000 prompt tok | 80023 | 80025 | 80023 | 80025 |
| 128000 prompt tok | 128025 | 128027 | 128025 | 128027 |
| 256000 prompt tok | 256020 | 256022 | 256020 | 256022 |
| 1000 TTFT s | 0.31 | 0.46 | 0.28 | 0.33 |
| 8000 TTFT s | 0.29 | 0.88 | 0.26 | 0.80 |
| 32000 TTFT s | 3.87 | 3.86 | 3.83 | 3.60 |
| 80000 TTFT s | 12.21 | 12.00 | 11.53 | 12.39 |
| 128000 TTFT s | 23.93 | 24.03 | 23.58 | 22.74 |
| 256000 TTFT s | 71.45 | 71.19 | 69.50 | 70.40 |
| 1000 prefill tok/s | 3293 | 2237 | 3669 | 3091 |
| 8000 prefill tok/s | 27446 | 9153 | 31325 | 10071 |
| 32000 prefill tok/s | 8270 | 8290 | 8357 | 8890 |
| 80000 prefill tok/s | 6553 | 6669 | 6942 | 6457 |
| 128000 prefill tok/s | 5350 | 5327 | 5430 | 5630 |
| 256000 prefill tok/s | 3583 | 3596 | 3684 | 3637 |
| 1000 decode tok/s | 191.8 | 179.9 | 193.6 | 193.9 |
| 8000 decode tok/s | 191.3 | 170.9 | 181.5 | 190.2 |
| 32000 decode tok/s | 174.8 | 176.4 | 177.0 | 183.6 |
| 80000 decode tok/s | 153.7 | 153.0 | 158.4 | 148.1 |
| 128000 decode tok/s | 139.6 | 139.8 | 151.3 | 142.3 |
| 256000 decode tok/s | 110.2 | 105.2 | 120.9 | 104.4 |

## saver / whole, YaRN

| target | saver/whole prefill p2 | saver/whole decode p2 | saver/whole decode p1 | out_sha1 p2 equal |
|---|---:|---:|---:|---|
| 1000 | 79.5% | 93.8% | 96.8% | yes |
| 8000 | 90.5% | 95.7% | 92.8% | yes |
| 32000 | 92.1% | 97.1% | 93.3% | yes |
| 80000 | 93.7% | 96.3% | 103.4% | yes |
| 128000 | 94.5% | 97.2% | 102.4% | yes |
| 256000 | 96.0% | 96.0% | 96.4% | yes |
| 262000 | 96.4% | 92.7% | 96.3% | yes |
| 300000 | 96.6% | 93.3% | 96.6% | yes |
| 380000 | 97.4% | 94.9% | 93.8% | yes |

## saver / whole, control

| target | saver/whole prefill p2 | saver/whole decode p2 | saver/whole decode p1 | out_sha1 p2 equal |
|---|---:|---:|---:|---|
| 1000 | 72.4% | 92.8% | 99.1% | yes |
| 8000 | 90.9% | 89.9% | 105.4% | yes |
| 32000 | 93.3% | 96.1% | 98.8% | yes |
| 80000 | 103.3% | 103.3% | 97.0% | yes |
| 128000 | 94.6% | 98.2% | 92.3% | yes |
| 256000 | 98.9% | 100.8% | 91.1% | yes |

## YaRN / control, saver

| target | yarn/ctrl prefill p2 | yarn/ctrl decode p2 | yarn/ctrl decode p1 | out_sha1 p2 equal |
|---|---:|---:|---:|---|
| 1000 | 88.9% | 96.8% | 100.2% | no |
| 8000 | 96.8% | 104.6% | 99.5% | no |
| 32000 | 96.4% | 97.6% | 96.1% | no |
| 80000 | 94.3% | 99.7% | 99.2% | no |
| 128000 | 97.0% | 101.7% | 100.1% | no |
| 256000 | 97.9% | 106.5% | 102.0% | no |

## YaRN / control, whole

| target | yarn/ctrl prefill p2 | yarn/ctrl decode p2 | yarn/ctrl decode p1 | out_sha1 p2 equal |
|---|---:|---:|---:|---|
| 1000 | 80.9% | 95.8% | 102.5% | no |
| 8000 | 97.2% | 98.3% | 113.0% | no |
| 32000 | 97.7% | 96.6% | 101.8% | no |
| 80000 | 103.9% | 107.0% | 93.1% | no |
| 128000 | 97.1% | 102.8% | 90.2% | no |
| 256000 | 100.8% | 111.8% | 96.4% | no |

## Natural text

* c-sn: 154.8 tok/s, md5 {'code': '8a3ec4b317c581a4c140b638ff8695b9', 'essay': 'ef957d712ca24ccf9e2df433fd333216', 'math': '1c460027072a785e8a4ca162e6f8f0f7', 'translate': 'e243f5b3a51207a118f15f84f66bbacc', 'summary': 'a450eb534235cee9a868bf8b572acc8e'}
* y-sn: 153.8 tok/s, md5 {'code': '186498b33d94f4d2043a83977b961e72', 'essay': 'e063bebf72648e2e29ac3b29e9313290', 'math': '20ce984fc16de1a3c790c726a439e451', 'translate': '38300dbf4ff230a4f77641754867420e', 'summary': 'a4f2137cc0ed726b3f3b2feb3dd65e46'}
* y-wn: 166.5 tok/s, md5 {'code': '186498b33d94f4d2043a83977b961e72', 'essay': 'e063bebf72648e2e29ac3b29e9313290', 'math': '20ce984fc16de1a3c790c726a439e451', 'translate': '38300dbf4ff230a4f77641754867420e', 'summary': 'a4f2137cc0ed726b3f3b2feb3dd65e46'}

## Depth needles

| kind | size | depth | planted | y-sq prompt tok / s / answer / pass | y-wq prompt tok / s / answer / pass | answers equal |
|---|---|---|---|---|---|---|
| multi | 300000 | [0.02, 0.5, 0.97] | INDIGO-HERON-4406, COBALT-LYNX-1793, VIOLET-LYNX-4431 | 299970 / 93.0 / `INDIGO-HERON-4406 COBALT-LYNX-1793 VIOLET-LYNX-4431` / PASS | 299970 / 91.3 / `INDIGO-HERON-4406 COBALT-LYNX-1793 VIOLET-LYNX-4431` / PASS | yes |
| multi | 380000 | [0.02, 0.5, 0.97] | VIOLET-HERON-6365, SAFFRON-HERON-4324, SAFFRON-LYNX-9979 | 379971 / 138.4 / `VIOLET-HERON-6365 SAFFRON-HERON-4324 SAFFRON-LYNX-9979` / PASS | 379971 / 136.0 / `VIOLET-HERON-6365 SAFFRON-HERON-4324 SAFFRON-LYNX-9979` / PASS | yes |
| multi | 390000 | [0.02, 0.5, 0.97] | SAFFRON-FALCON-5175, SAFFRON-FALCON-3130, SAFFRON-LYNX-7473 | 389971 / 145.0 / `SAFFRON-FALCON-5175 SAFFRON-FALCON-3130 SAFFRON-LYNX-7473` / PASS | 389971 / 142.4 / `SAFFRON-FALCON-5175 SAFFRON-FALCON-3130 SAFFRON-LYNX-7473` / PASS | yes |
| single | 300000 | 0.1 | AMBER-LYNX-3255 | 299965 / 93.9 / `AMBER-LYNX-3255` / PASS | 299965 / 91.3 / `AMBER-LYNX-3255` / PASS | yes |
| single | 300000 | 0.5 | SAFFRON-MARTEN-3242 | 299966 / 93.5 / `SAFFRON-MARTEN-3242` / PASS | 299966 / 91.0 / `SAFFRON-MARTEN-3242` / PASS | yes |
| single | 300000 | 0.85 | SAFFRON-MARTEN-4943 | 299966 / 93.4 / `SAFFRON-MARTEN-4943` / PASS | 299966 / 91.1 / `SAFFRON-MARTEN-4943` / PASS | yes |
| single | 300000 | 0.98 | AMBER-OTTER-7515 | 299965 / 93.1 / `AMBER-OTTER-7515` / PASS | 299965 / 91.3 / `AMBER-OTTER-7515` / PASS | yes |
| single | 380000 | 0.1 | INDIGO-OTTER-7382 | 379966 / 138.9 / `INDIGO-OTTER-7382` / PASS | 379966 / 136.9 / `INDIGO-OTTER-7382` / PASS | yes |
| single | 380000 | 0.5 | COBALT-MARTEN-6404 | 379966 / 139.0 / `COBALT-MARTEN-6404` / PASS | 379966 / 135.8 / `COBALT-MARTEN-6404` / PASS | yes |
| single | 380000 | 0.85 | COBALT-MARTEN-2636 | 379966 / 139.4 / `COBALT-MARTEN-2636` / PASS | 379966 / 137.1 / `COBALT-MARTEN-2636` / PASS | yes |
| single | 380000 | 0.98 | SAFFRON-MARTEN-8147 | 379966 / 139.3 / `SAFFRON-MARTEN-8147` / PASS | 379966 / 137.0 / `SAFFRON-MARTEN-8147` / PASS | yes |
| single | 390000 | 0.1 | INDIGO-FALCON-2515 | 389966 / 145.4 / `INDIGO-FALCON-2515` / PASS | 389966 / 143.1 / `INDIGO-FALCON-2515` / PASS | yes |
| single | 390000 | 0.5 | INDIGO-FALCON-8697 | 389966 / 145.5 / `INDIGO-FALCON-8697` / PASS | 389966 / 143.0 / `INDIGO-FALCON-8697` / PASS | yes |
| single | 390000 | 0.85 | SAFFRON-HERON-4658 | 389966 / 145.4 / `SAFFRON-HERON-4658` / PASS | 389966 / 143.0 / `SAFFRON-HERON-4658` / PASS | yes |
| single | 390000 | 0.98 | VIOLET-HERON-7299 | 389964 / 145.4 / `VIOLET-HERON-7299` / PASS | 389964 / 143.2 / `VIOLET-HERON-7299` / PASS | yes |
