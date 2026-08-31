# Ornith RTX 2000 Ada multi-agent prefill tuning

Date: 2026-08-31  
Host: NVIDIA RTX 2000 Ada Generation (sm_89), 16 GiB, 70 W, WSL 2  
Stack: Windows driver 595.95, CUDA toolkit 13.1, PyTorch 2.11.0+cu130,
Triton 3.6.0

Growable GGUF serving defaults to one prefill lane because grouping independent
long prompts is expensive, but the RTX 5080 import also conditionally groups very
short fresh prompts. The grouping crossover is GPU-dependent, so it was swept on
Ada rather than copied unchanged.

All runs used four concurrent agents, 8K prefill chunks, a 65,536-token per-agent
context limit, a 262,144-token shared virtual KV ceiling, LFU expert caching, and
128 generated tokens. Every retained run recovered all four disjoint passcodes
without foreign-agent data. `Grouped` forces unrestricted prompt lanes;
`serialized` forces one lane.

| Model / KV | Prompt per agent | Grouped wall | Serialized wall | Winner |
|---|---:|---:|---:|---|
| Q4_K_M / INT4 | 1,024 | 7.861 s | 9.691 s | Grouped, 18.9% |
| Q4_K_M / INT4 | 1,536 | 11.425 s | 12.590 s | Grouped, 9.3% |
| Q4_K_M / INT4 | 2,048 | 17.151 s | 12.520 s | Serialized, 27.0% |
| Q4_K_M / INT4 | 4,096 | 25.202 s | 16.389 s | Serialized, 35.0% |
| Q6_K / Q8_0 | 1,536 | 14.381 s | 17.966 s | Grouped, 20.0% |

The accepted sm_89 crossover is therefore 1,536 templated tokens for both
production model/KV pairs. The existing 1,280-token fallback remains unchanged
for other GPU architectures, including the imported sm_120 policy. Continuations,
groups that do not fit one prefill budget, and explicit
`--max-prefill-sequences` settings retain their existing behavior.

The benchmark harness now permits smaller protected prefix, needle-context, and
tail regions for short synthetic prompts. The default long-context protection
sizes are unchanged.
