# Ornith RTX 5080 cold-session spill/restore validation

Date: 2026-08-30  
Host: NVIDIA GeForce RTX 5080 (sm_120), 16 GiB  
Model: Ornith-1.5-35B Q4_K_M, INT4 KV, 65,536-token growth steps  
Host reserve: 3 GiB

An automatically derived Claude Code session ran a 70K synthetic needle request,
released its soft GPU lease, then resumed from the full client-resubmitted
conversation. The test forces a physical VMM transition, rather than accepting an
ordinary still-resident radix hit as evidence of restore correctness.

| Gate | Result |
|---|---:|
| First-turn input | 70,012 tokens |
| Retained checkpoint | 70,028 tokens / 0.44 GiB |
| First answer | exact passcode `5663623` |
| KV after idle release | 131,072 -> 65,536 tokens |
| VRAM physically returned | 0.35 GiB |
| MoE cache after release | 4,720 -> 4,918 slots |
| Restore source | disk (RAM admission preserved the 3 GiB reserve) |
| Second-turn cache hit | 70,028 tokens |
| New second-turn input | 27 tokens |
| Restore + graph recapture + reply | 1.70 s |
| Second answer | exact passcode `5663623` |
| Subsequent checkpoint | RAM / 0.44 GiB when safe headroom recovered |
| Shutdown cleanup | no GPU process or checkpoint directory remained |

The storage-level test separately round-trips both RAM and disk checkpoints after
complete radix eviction. It compares raw Q8-key/Q6-value payloads, both fp16 scale
slabs, and the final GDN conv/recurrent state byte-for-byte. The expanded
scheduler/KV/API gate passed 517 tests with one CUDA-only skip.

A second live gate covered Ornith Q6_K with asymmetric Q8-K/Q6-V storage. A 20,012-token
first turn recovered the same exact passcode, checkpointed 20,028 tokens (0.24 GiB) to
RAM, and eventually shrank the validation geometry from 32,768 to 16,384 physical KV
tokens, returning 0.14 GiB. A post-shrink third turn forced the actual RAM import,
reported a 20,062-token cache hit plus 21 new input tokens, and again answered exactly
`5663623`. Restore, decode-graph recapture, and reply took 6.02 s. The deliberately
small 16K growth step exists only to force this short validation transition; Q6
production keeps the measured 128K step.

Cold reuse is fail-closed. The client still sends the complete conversation. A saved
prefix is imported only if its int32 token sequence is an exact prefix of that
conversation and the live MHA storage fingerprint (K/V shapes, dtypes, and quantizers)
matches. Otherwise FreeToken deletes it and recomputes normally.
