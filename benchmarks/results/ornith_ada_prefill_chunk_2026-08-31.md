# Ornith RTX 2000 Ada prefill-chunk port

Date: 2026-08-31  
Host: NVIDIA RTX 2000 Ada Generation (sm_89), 16 GiB, 70 W, WSL 2  
Stack: Windows driver 595.95, CUDA toolkit 13.1, PyTorch 2.11.0+cu130,
Triton 3.6.0

The imported RTX 5080 machinery uses an 8,192-token maximum prefill chunk. On
this Ada host that size is a severe expert-prefill cliff, not merely a latency
tradeoff. Production auto-resolution therefore selects 4,096 only for
Qwen3.5-MoE GGUF on sm_89. An explicit `--max-prefill-length` remains
authoritative, and all other GPU/model combinations retain the 8K default.

## Single-request cold gates

Each pair used an identical synthetic prompt, one fresh server, 128 forced
output tokens, a 65,536-token context pool, LFU expert caching, and the requested
production KV type. Every run recovered exact passcode `5663623`.

| Model / KV | Prompt | Chunk | End-to-end prefill | Engine average | Decode | Peak VRAM | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| Q4_K_M / INT4 | 32,768 | 8,192 | 556.76 tok/s | 541.15 tok/s | 42.33 tok/s | 16.31 GiB | PASS |
| Q4_K_M / INT4 | 32,768 | 4,096 | 1,336.02 tok/s | 1,276.43 tok/s | 42.20 tok/s | 15.44 GiB | PASS |
| Q6_K / Q8_0 | 16,384 | 8,192 | 427.90 tok/s | 412.69 tok/s | 36.46 tok/s | 16.20 GiB | PASS |
| Q6_K / Q8_0 | 16,384 | 4,096 | 1,251.40 tok/s | 1,145.56 tok/s | 36.19 tok/s | 15.31 GiB | PASS |

That is +140.0% end-to-end prefill for Q4 and +192.5% for Q6, with decode
within 0.7%. The smaller temporary working set returned roughly 0.9 GiB of peak
VRAM in both pairs.

A narrow Q4 bracket rejected both neighboring directions: 2,048-token chunks
reached 1,210.21 tok/s end-to-end on a 16K prompt, while 6,144 reached only
779.64 tok/s on a 24K prompt. Both remained coherent, but neither displaced 4K.

The final no-override production gate logged
`Auto prefill chunk: 8192 -> 4096 ... on sm_89`, recovered the exact needle, and
reached 1,450.34 tok/s end-to-end / 1,260.33 tok/s engine average on 16K Q4.

## Mixed prefill/decode gate

One established Q4/INT4 agent decoded while a 16K helper arrived. All modes
kept both needles isolated and coherent.

| Policy | Helper TTFT | Helper wall | Main worst gap | Main tokens during helper |
|---|---:|---:|---:|---:|
| Fixed, 8K | 32.01 s | 35.85 s | 15.67 s | 193 |
| Adaptive, 8K | 26.83 s | 30.48 s | 14.54 s | 155 |
| Adaptive, 4K | 11.26 s | 14.96 s | 2.96 s | 167 |

The imported measured-time controller is beneficial on Ada, but the model/GPU
chunk adaptation supplies the larger gain: versus adaptive 8K it reduced helper
wall time by 50.9% and the established agent's worst pause by 79.7%.
